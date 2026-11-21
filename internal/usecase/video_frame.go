package usecase

import (
	"bytes"
	"context"
	"fmt"
	"image/gif"
	"image/jpeg"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// extractVideoFirstFrame extracts the first video frame as JPEG bytes for multimodal embedding.
// Animated GIFs are decoded in pure Go via image/gif first, falling back to ffmpeg for all other
// video containers (MP4, WebM, MOV) or misnamed files.
func extractVideoFirstFrame(ctx context.Context, absPath string) ([]byte, error) {
	if strings.EqualFold(filepath.Ext(absPath), ".gif") {
		if data, err := extractGIFFirstFrame(absPath); err == nil && len(data) > 0 {
			return data, nil
		}
	}

	ffmpegPath, err := findFFmpegBinary()
	if err != nil {
		return nil, err
	}

	frameCtx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()

	cmd := exec.CommandContext(frameCtx, ffmpegPath,
		"-hide_banner",
		"-loglevel", "error",
		"-nostdin",
		"-i", absPath,
		"-map", "0:v:0",
		"-an", "-sn", "-dn",
		"-frames:v", "1",
		"-vf", "format=yuvj420p",
		"-f", "image2pipe",
		"-vcodec", "mjpeg",
		"-q:v", "2",
		"-",
	)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		errMsg := strings.TrimSpace(stderr.String())
		if errMsg != "" {
			return nil, fmt.Errorf("ffmpeg failed extracting first frame: %w (%s)", err, errMsg)
		}
		return nil, fmt.Errorf("ffmpeg failed extracting first frame: %w", err)
	}

	out := stdout.Bytes()
	if len(out) < 2 || out[0] != 0xFF || out[1] != 0xD8 {
		return nil, fmt.Errorf("ffmpeg returned empty or invalid JPEG frame")
	}
	return out, nil
}

func extractGIFFirstFrame(absPath string) ([]byte, error) {
	f, err := os.Open(absPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	img, err := gif.Decode(f)
	if err != nil {
		return nil, err
	}

	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 90}); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func findFFmpegBinary() (string, error) {
	if p, err := exec.LookPath("ffmpeg"); err == nil {
		return p, nil
	}
	if runtime.GOOS != "windows" {
		for _, candidate := range []string{
			"/opt/homebrew/bin/ffmpeg",
			"/usr/local/bin/ffmpeg",
			"/usr/bin/ffmpeg",
		} {
			if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
				return candidate, nil
			}
		}
	}
	return "", fmt.Errorf("ffmpeg not found in PATH: %w", exec.ErrNotFound)
}
