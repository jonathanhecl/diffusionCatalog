package bootstrap

import (
	"bufio"
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

//go:embed vision_worker/*
var embeddedVisionWorker embed.FS

type setupState struct {
	CheckedAt          time.Time `json:"checked_at"`
	VisionVenvReady    bool      `json:"vision_venv_ready"`
	AutoStartVision    bool      `json:"auto_start_vision"`
	DismissedOptional  bool      `json:"dismissed_optional"`
}

// Result holds any runtime endpoints or child processes started by the portable bootstrapper.
type Result struct {
	VisionWorkerURL string
	Cleanup         func()
}

// EnsureEnvironment makes the compiled binary portable and self-installing.
// On first execution (or when forceSetup is true), it checks for missing directories and
// optional runtime tools (Python 3 vision sidecar virtualenv, ffmpeg/ffprobe), interactively asking
// the user whether to install any missing component.
func EnsureEnvironment(ctx context.Context, mediaDir, cacheDir string, forceSetup, nonInteractive bool) Result {
	res := Result{Cleanup: func() {}}

	_ = os.MkdirAll(mediaDir, 0755)
	_ = os.MkdirAll(cacheDir, 0755)

	// Always extract the embedded Python vision worker into .cache/vision_worker so the binary is 100% standalone
	workerDir := filepath.Join(cacheDir, "vision_worker")
	_ = extractEmbeddedWorker(workerDir)

	stateFile := filepath.Join(cacheDir, ".setup_state.json")
	state := loadState(stateFile)

	venvPython := venvPythonPath(workerDir)
	if fileExists(venvPython) {
		state.VisionVenvReady = true
	}

	isTTY := isInteractiveTerminal() && !nonInteractive
	shouldPrompt := forceSetup || (isTTY && !state.DismissedOptional && !state.VisionVenvReady)

	if shouldPrompt && isTTY {
		reader := bufio.NewReader(os.Stdin)
		fmt.Println("================================================================")
		fmt.Println(" DiffusionCatalog — Portable Self-Installer & Dependency Check")
		fmt.Println("================================================================")

		// 1. Check ffmpeg / ffprobe (used for video container inspection)
		if _, err := exec.LookPath("ffprobe"); err != nil {
			if askYesNo(reader, "[?] Optional tool 'ffmpeg' (video metadata inspection) is missing. Install it now? [y/N]: ") {
				if err := installSystemPackage("ffmpeg"); err != nil {
					log.Printf("warning: could not install ffmpeg automatically: %v", err)
				} else {
					fmt.Println("[+] ffmpeg installed successfully.")
				}
			}
		}

		// 2. Check Python 3 for agnostic Face Detection & Extraction worker
		if !state.VisionVenvReady {
			if _, err := exec.LookPath("python3"); err != nil {
				if askYesNo(reader, "[?] 'python3' (required for local Face Detection worker) is missing. Install Python 3 now? [y/N]: ") {
					if err := installSystemPackage("python3"); err != nil {
						log.Printf("warning: could not install python3 automatically: %v", err)
					}
				}
			}

			if _, err := exec.LookPath("python3"); err == nil {
				if askYesNo(reader, "[?] Set up and install the local Python Face Detection & Cropping worker (.venv) now? [y/N]: ") {
					if err := setupVisionVirtualenv(workerDir); err != nil {
						log.Printf("warning: failed to set up Python vision worker virtualenv: %v", err)
					} else {
						state.VisionVenvReady = true
						state.AutoStartVision = true
						fmt.Println("[+] Python Vision Worker installed in", workerDir)
					}
				}
			}
		}

		state.CheckedAt = time.Now()
		state.DismissedOptional = true
		saveState(stateFile, state)
		fmt.Println("================================================================")
	}

	// Start the managed local Python vision sidecar if installed and enabled
	if state.VisionVenvReady && state.AutoStartVision && fileExists(venvPython) {
		url, cleanup, err := startManagedVisionWorker(ctx, workerDir, venvPython)
		if err != nil {
			log.Printf("warning: managed vision worker did not start: %v", err)
		} else {
			res.VisionWorkerURL = url
			res.Cleanup = cleanup
		}
	}

	return res
}

func extractEmbeddedWorker(targetDir string) error {
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return err
	}
	entries, err := embeddedVisionWorker.ReadDir("vision_worker")
	if err != nil {
		return err
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		data, err := embeddedVisionWorker.ReadFile("vision_worker/" + entry.Name())
		if err != nil {
			continue
		}
		_ = os.WriteFile(filepath.Join(targetDir, entry.Name()), data, 0644)
	}
	return nil
}

func setupVisionVirtualenv(workerDir string) error {
	fmt.Println("[*] Creating Python virtual environment in", filepath.Join(workerDir, ".venv"))
	venvCmd := exec.Command("python3", "-m", "venv", filepath.Join(workerDir, ".venv"))
	venvCmd.Stdout = os.Stdout
	venvCmd.Stderr = os.Stderr
	if err := venvCmd.Run(); err != nil {
		return err
	}

	pipBin := filepath.Join(workerDir, ".venv", "bin", "pip")
	if runtime.GOOS == "windows" {
		pipBin = filepath.Join(workerDir, ".venv", "Scripts", "pip.exe")
	}

	fmt.Println("[*] Installing FastAPI, Uvicorn, Pillow, and OpenCV Headless...")
	pipCmd := exec.Command(pipBin, "install", "--upgrade", "-r", filepath.Join(workerDir, "requirements.txt"))
	pipCmd.Stdout = os.Stdout
	pipCmd.Stderr = os.Stderr
	return pipCmd.Run()
}

func startManagedVisionWorker(ctx context.Context, workerDir, venvPython string) (string, func(), error) {
	addr := "127.0.0.1:8091"
	url := "http://" + addr

	cmd := exec.CommandContext(ctx, venvPython, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8091", "--log-level", "warning")
	cmd.Dir = workerDir
	if err := cmd.Start(); err != nil {
		return "", func() {}, err
	}

	cleanup := func() {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
	}

	// Wait briefly for readiness
	client := &http.Client{Timeout: 500 * time.Millisecond}
	for i := 0; i < 12; i++ {
		resp, err := client.Get(url + "/health")
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				log.Printf("started embedded Python vision worker at %s", url)
				return url, cleanup, nil
			}
		}
		time.Sleep(250 * time.Millisecond)
	}

	return url, cleanup, nil
}

func installSystemPackage(pkg string) error {
	switch runtime.GOOS {
	case "darwin":
		if _, err := exec.LookPath("brew"); err != nil {
			return fmt.Errorf("Homebrew ('brew') not found; please install %s manually or install Homebrew from https://brew.sh", pkg)
		}
		cmd := exec.Command("brew", "install", pkg)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		cmd.Stdin = os.Stdin
		return cmd.Run()
	case "linux":
		if _, err := exec.LookPath("apt-get"); err == nil {
			cmd := exec.Command("sudo", "apt-get", "install", "-y", pkg)
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			cmd.Stdin = os.Stdin
			return cmd.Run()
		}
		if _, err := exec.LookPath("dnf"); err == nil {
			cmd := exec.Command("sudo", "dnf", "install", "-y", pkg)
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			cmd.Stdin = os.Stdin
			return cmd.Run()
		}
		if _, err := exec.LookPath("pacman"); err == nil {
			cmd := exec.Command("sudo", "pacman", "-S", "--noconfirm", pkg)
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			cmd.Stdin = os.Stdin
			return cmd.Run()
		}
		return fmt.Errorf("no supported package manager (apt-get, dnf, pacman) found to install %s", pkg)
	default:
		return fmt.Errorf("automatic OS package installation not supported on %s", runtime.GOOS)
	}
}

func venvPythonPath(workerDir string) string {
	if runtime.GOOS == "windows" {
		return filepath.Join(workerDir, ".venv", "Scripts", "python.exe")
	}
	return filepath.Join(workerDir, ".venv", "bin", "python")
}

func askYesNo(r *bufio.Reader, prompt string) bool {
	fmt.Print(prompt)
	line, err := r.ReadString('\n')
	if err != nil {
		return false
	}
	ans := strings.ToLower(strings.TrimSpace(line))
	return ans == "y" || ans == "yes"
}

func isInteractiveTerminal() bool {
	fi, err := os.Stdin.Stat()
	if err != nil {
		return false
	}
	return (fi.Mode() & os.ModeCharDevice) != 0
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func loadState(path string) setupState {
	var st setupState
	data, err := os.ReadFile(path)
	if err == nil {
		_ = json.Unmarshal(data, &st)
	}
	return st
}

func saveState(path string, st setupState) {
	data, err := json.MarshalIndent(st, "", "  ")
	if err == nil {
		_ = os.WriteFile(path, data, 0644)
	}
}
