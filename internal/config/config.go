package config

import (
	"flag"
	"os"
	"path/filepath"
)

// Config holds application-level configuration resolved from flags and environment variables.
type Config struct {
	Addr             string
	MediaDir         string
	CacheDir         string
	DBPath           string
	AdminUser        string
	AdminPass        string
	VisionWorkerURL  string
	AutoScanOnBoot   bool
	SeedDemoIfEmpty  bool
	ForceSetup       bool
	NonInteractive   bool
}

// Load parses command-line flags with environment variable fallbacks.
func Load() Config {
	var cfg Config

	flag.StringVar(&cfg.Addr, "addr", envOrDefault("CATALOG_ADDR", ":8080"), "HTTP listen address")
	flag.StringVar(&cfg.MediaDir, "media-dir", envOrDefault("CATALOG_MEDIA_DIR", "./media"), "Root directory containing diffusion images and videos (read-only)")
	flag.StringVar(&cfg.CacheDir, "cache-dir", envOrDefault("CATALOG_CACHE_DIR", "./.cache"), "Directory for generated thumbnails and cropped face regions")
	flag.StringVar(&cfg.DBPath, "db", envOrDefault("CATALOG_DB_PATH", "./catalog.db"), "Path to local SQLite database file")
	flag.StringVar(&cfg.AdminUser, "admin-user", envOrDefault("CATALOG_ADMIN_USER", "admin"), "Default administrator username")
	flag.StringVar(&cfg.AdminPass, "admin-pass", envOrDefault("CATALOG_ADMIN_PASS", "admin123"), "Default administrator password (used only when bootstrapping empty DB)")
	flag.StringVar(&cfg.VisionWorkerURL, "vision-url", envOrDefault("CATALOG_VISION_URL", ""), "Optional HTTP endpoint for external face detection & embedding worker")
	flag.BoolVar(&cfg.AutoScanOnBoot, "auto-scan", envOrDefaultBool("CATALOG_AUTO_SCAN", true), "Automatically scan media directory on startup")
	flag.BoolVar(&cfg.SeedDemoIfEmpty, "seed-demo", envOrDefaultBool("CATALOG_SEED_DEMO", true), "Generate sample diffusion images in media-dir if folder does not exist or is empty")
	flag.BoolVar(&cfg.ForceSetup, "setup", false, "Run interactive dependency check and self-installer on startup")
	flag.BoolVar(&cfg.NonInteractive, "non-interactive", envOrDefaultBool("CATALOG_NON_INTERACTIVE", false), "Disable interactive dependency installation prompts")

	flag.Parse()

	if absMedia, err := filepath.Abs(cfg.MediaDir); err == nil {
		cfg.MediaDir = absMedia
	}
	if absCache, err := filepath.Abs(cfg.CacheDir); err == nil {
		cfg.CacheDir = absCache
	}
	return cfg
}

func envOrDefault(key, fallback string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return fallback
}

func envOrDefaultBool(key string, fallback bool) bool {
	val := os.Getenv(key)
	if val == "true" || val == "1" || val == "yes" {
		return true
	}
	if val == "false" || val == "0" || val == "no" {
		return false
	}
	return fallback
}
