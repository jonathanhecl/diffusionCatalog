package config

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	DefaultConfigFile = "config.json"
	DefaultAddr       = ":1080"
	DefaultMediaDir   = "./media"
	DefaultCacheDir   = "./.cache"
	DefaultDBPath     = "./catalog.db"
	DefaultAdminUser  = "admin"
	DefaultAdminPass  = "admin123"
)

// Config holds application-level configuration resolved from config files, flags, and environment variables.
type Config struct {
	Addr            string `json:"addr"`
	MediaDir        string `json:"media_dir"`
	CacheDir        string `json:"cache_dir"`
	DBPath          string `json:"db_path"`
	AdminUser       string `json:"admin_user"`
	AdminPass       string `json:"admin_pass"`
	VisionWorkerURL string `json:"vision_url"`
	AutoScanOnBoot  bool   `json:"auto_scan"`
	SeedDemoIfEmpty bool   `json:"seed_demo"`
	ForceSetup      bool   `json:"-"`
	NonInteractive  bool   `json:"-"`
	ConfigFile      string `json:"config_file"`
}

// FileConfig models the JSON file structure for config.json.
type FileConfig struct {
	Port            any    `json:"port,omitempty"`
	Addr            string `json:"addr,omitempty"`
	MediaDir        string `json:"media_dir,omitempty"`
	CacheDir        string `json:"cache_dir,omitempty"`
	DBPath          string `json:"db_path,omitempty"`
	AdminUser       string `json:"admin_user,omitempty"`
	AdminPass       string `json:"admin_pass,omitempty"`
	VisionWorkerURL string `json:"vision_url,omitempty"`
	AutoScanOnBoot  *bool  `json:"auto_scan,omitempty"`
	SeedDemoIfEmpty *bool  `json:"seed_demo,omitempty"`
}

// Load parses configuration from config.json, environment variables, and os.Args.
func Load() Config {
	return LoadWithArgs(os.Args[1:])
}

// LoadWithArgs allows testing and programmatic configuration loading.
func LoadWithArgs(args []string) Config {
	loadDotEnv(".env")

	// Phase 1: Determine config file path from flags or environment
	configPath := envOrDefault("CATALOG_CONFIG", DefaultConfigFile)
	for i, arg := range args {
		if strings.HasPrefix(arg, "--config=") || strings.HasPrefix(arg, "-config=") {
			configPath = strings.SplitN(arg, "=", 2)[1]
		} else if (arg == "--config" || arg == "-config") && i+1 < len(args) {
			configPath = args[i+1]
		}
	}

	// Phase 2: If default config.json is requested and doesn't exist, create it
	if configPath == DefaultConfigFile {
		_ = EnsureDefaultConfigFile(configPath)
	}

	// Phase 3: Start with base defaults
	cfg := Config{
		Addr:            DefaultAddr,
		MediaDir:        DefaultMediaDir,
		CacheDir:        DefaultCacheDir,
		DBPath:          DefaultDBPath,
		AdminUser:       DefaultAdminUser,
		AdminPass:       DefaultAdminPass,
		VisionWorkerURL: "",
		AutoScanOnBoot:  true,
		SeedDemoIfEmpty: false,
		ForceSetup:      false,
		NonInteractive:  false,
		ConfigFile:      configPath,
	}

	// Phase 4: Apply config file if available
	var fileCfg FileConfig
	if data, err := os.ReadFile(configPath); err == nil {
		if err := json.Unmarshal(data, &fileCfg); err == nil {
			if fileCfg.Addr != "" || fileCfg.Port != nil {
				cfg.Addr = normalizeAddr(fileCfg.Addr, parsePortValue(fileCfg.Port))
			}
			if fileCfg.MediaDir != "" {
				cfg.MediaDir = fileCfg.MediaDir
			}
			if fileCfg.CacheDir != "" {
				cfg.CacheDir = fileCfg.CacheDir
			}
			if fileCfg.DBPath != "" {
				cfg.DBPath = fileCfg.DBPath
			}
			if fileCfg.AdminUser != "" {
				cfg.AdminUser = fileCfg.AdminUser
			}
			if fileCfg.AdminPass != "" {
				cfg.AdminPass = fileCfg.AdminPass
			}
			if fileCfg.VisionWorkerURL != "" {
				cfg.VisionWorkerURL = fileCfg.VisionWorkerURL
			}
			if fileCfg.AutoScanOnBoot != nil {
				cfg.AutoScanOnBoot = *fileCfg.AutoScanOnBoot
			}
			if fileCfg.SeedDemoIfEmpty != nil {
				cfg.SeedDemoIfEmpty = *fileCfg.SeedDemoIfEmpty
			}
		}
	}

	// Phase 5: Apply environment variable overrides
	envAddr := envOrDefault("CATALOG_ADDR", "")
	envPort := envOrDefault("CATALOG_PORT", envOrDefault("PORT", ""))
	if envAddr != "" || envPort != "" {
		cfg.Addr = normalizeAddr(envAddr, envPort)
	}
	if v := os.Getenv("CATALOG_MEDIA_DIR"); v != "" {
		cfg.MediaDir = v
	}
	if v := os.Getenv("CATALOG_CACHE_DIR"); v != "" {
		cfg.CacheDir = v
	}
	if v := os.Getenv("CATALOG_DB_PATH"); v != "" {
		cfg.DBPath = v
	}
	if v := os.Getenv("CATALOG_ADMIN_USER"); v != "" {
		cfg.AdminUser = v
	}
	if v := os.Getenv("CATALOG_ADMIN_PASS"); v != "" {
		cfg.AdminPass = v
	}
	if v := os.Getenv("CATALOG_VISION_URL"); v != "" {
		cfg.VisionWorkerURL = v
	}
	if v, ok := envBool("CATALOG_AUTO_SCAN"); ok {
		cfg.AutoScanOnBoot = v
	}
	if v, ok := envBool("CATALOG_SEED_DEMO"); ok {
		cfg.SeedDemoIfEmpty = v
	}
	if v, ok := envBool("CATALOG_NON_INTERACTIVE"); ok {
		cfg.NonInteractive = v
	}

	// Phase 6: Parse command-line flags
	fs := flag.NewFlagSet("catalogd", flag.ContinueOnError)

	flagConfig := fs.String("config", configPath, "Path to JSON configuration file")
	flagAddr := fs.String("addr", cfg.Addr, "HTTP listen address (e.g. :888 or 127.0.0.1:888)")
	flagPort := fs.String("port", "", "HTTP listen port (e.g. 888 or 8080)")
	flagMediaDir := fs.String("media-dir", cfg.MediaDir, "Root directory containing diffusion images and videos (read-only)")
	flagCacheDir := fs.String("cache-dir", cfg.CacheDir, "Directory for generated thumbnails and cropped face regions")
	flagDBPath := fs.String("db", cfg.DBPath, "Path to local SQLite database file")
	flagAdminUser := fs.String("admin-user", cfg.AdminUser, "Default administrator username")
	flagAdminPass := fs.String("admin-pass", cfg.AdminPass, "Default administrator password (used only when bootstrapping empty DB)")
	flagVisionURL := fs.String("vision-url", cfg.VisionWorkerURL, "Optional HTTP endpoint for external face detection & embedding worker")
	flagAutoScan := fs.Bool("auto-scan", cfg.AutoScanOnBoot, "Automatically scan media directory on startup")
	flagSeedDemo := fs.Bool("seed-demo", cfg.SeedDemoIfEmpty, "Generate sample diffusion images in media-dir if folder does not exist or is empty")
	flagSetup := fs.Bool("setup", false, "Run interactive dependency check and self-installer on startup")
	flagNonInteractive := fs.Bool("non-interactive", cfg.NonInteractive, "Disable interactive dependency installation prompts")

	explicitFlags := make(map[string]bool)
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			os.Exit(0)
		}
		os.Exit(2)
	}
	fs.Visit(func(f *flag.Flag) {
		explicitFlags[f.Name] = true
	})

	if explicitFlags["config"] {
		cfg.ConfigFile = *flagConfig
	}
	if explicitFlags["port"] {
		cfg.Addr = normalizeAddr("", *flagPort)
	}
	if explicitFlags["addr"] {
		cfg.Addr = normalizeAddr(*flagAddr, "")
	}
	if explicitFlags["media-dir"] {
		cfg.MediaDir = *flagMediaDir
	}
	if explicitFlags["cache-dir"] {
		cfg.CacheDir = *flagCacheDir
	}
	if explicitFlags["db"] {
		cfg.DBPath = *flagDBPath
	}
	if explicitFlags["admin-user"] {
		cfg.AdminUser = *flagAdminUser
	}
	if explicitFlags["admin-pass"] {
		cfg.AdminPass = *flagAdminPass
	}
	if explicitFlags["vision-url"] {
		cfg.VisionWorkerURL = *flagVisionURL
	}
	if explicitFlags["auto-scan"] {
		cfg.AutoScanOnBoot = *flagAutoScan
	}
	if explicitFlags["seed-demo"] {
		cfg.SeedDemoIfEmpty = *flagSeedDemo
	}
	if explicitFlags["setup"] {
		cfg.ForceSetup = *flagSetup
	}
	if explicitFlags["non-interactive"] {
		cfg.NonInteractive = *flagNonInteractive
	}

	if absMedia, err := filepath.Abs(cfg.MediaDir); err == nil {
		cfg.MediaDir = absMedia
	}
	if absCache, err := filepath.Abs(cfg.CacheDir); err == nil {
		cfg.CacheDir = absCache
	}

	return cfg
}

// EnsureDefaultConfigFile creates config.json with default values if it does not already exist.
func EnsureDefaultConfigFile(path string) error {
	if _, err := os.Stat(path); err == nil {
		return nil
	}
	defaultJSON := `{
  "port": 1080,
  "addr": ":1080",
  "media_dir": "./media",
  "cache_dir": "./.cache",
  "db_path": "./catalog.db",
  "admin_user": "admin",
  "admin_pass": "admin123",
  "vision_url": "",
  "auto_scan": true,
  "seed_demo": false
}
`
	return os.WriteFile(path, []byte(defaultJSON), 0644)
}

func normalizeAddr(addr, port string) string {
	addr = strings.TrimSpace(addr)
	port = strings.TrimSpace(port)

	if addr != "" {
		if !strings.Contains(addr, ":") {
			return ":" + addr
		}
		return addr
	}
	if port != "" {
		if strings.HasPrefix(port, ":") {
			return port
		}
		return ":" + port
	}
	return DefaultAddr
}

func parsePortValue(v any) string {
	switch val := v.(type) {
	case float64:
		return fmt.Sprintf("%d", int(val))
	case int:
		return fmt.Sprintf("%d", val)
	case int64:
		return fmt.Sprintf("%d", val)
	case string:
		return strings.TrimSpace(val)
	default:
		return ""
	}
}

func loadDotEnv(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	scanner := bufio.NewScanner(strings.NewReader(string(data)))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) == 2 {
			k := strings.TrimSpace(parts[0])
			v := strings.Trim(strings.TrimSpace(parts[1]), "\"'")
			if os.Getenv(k) == "" {
				_ = os.Setenv(k, v)
			}
		}
	}
}

func envOrDefault(key, fallback string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return fallback
}

func envBool(key string) (bool, bool) {
	val := os.Getenv(key)
	if val == "" {
		return false, false
	}
	val = strings.ToLower(val)
	if val == "true" || val == "1" || val == "yes" {
		return true, true
	}
	if val == "false" || val == "0" || val == "no" {
		return false, true
	}
	return false, false
}
