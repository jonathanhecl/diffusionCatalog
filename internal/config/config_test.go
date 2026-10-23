package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultConfigAddr(t *testing.T) {
	tmpDir := t.TempDir()
	cfgFile := filepath.Join(tmpDir, "non_existent_config.json")

	cfg := LoadWithArgs([]string{"--config=" + cfgFile})
	if cfg.Addr != ":1080" {
		t.Fatalf("expected default Addr :1080, got %s", cfg.Addr)
	}
}

func TestConfigFilePortNumber(t *testing.T) {
	tmpDir := t.TempDir()
	cfgFile := filepath.Join(tmpDir, "custom_config.json")
	content := `{
		"port": 8888,
		"admin_user": "custom_admin"
	}`
	if err := os.WriteFile(cfgFile, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	cfg := LoadWithArgs([]string{"--config=" + cfgFile})
	if cfg.Addr != ":8888" {
		t.Fatalf("expected Addr :8888 from config.json, got %s", cfg.Addr)
	}
	if cfg.AdminUser != "custom_admin" {
		t.Fatalf("expected admin_user custom_admin, got %s", cfg.AdminUser)
	}
}

func TestConfigFileAddrString(t *testing.T) {
	tmpDir := t.TempDir()
	cfgFile := filepath.Join(tmpDir, "custom_config.json")
	content := `{
		"addr": "127.0.0.1:9090"
	}`
	if err := os.WriteFile(cfgFile, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	cfg := LoadWithArgs([]string{"--config=" + cfgFile})
	if cfg.Addr != "127.0.0.1:9090" {
		t.Fatalf("expected Addr 127.0.0.1:9090 from config.json, got %s", cfg.Addr)
	}
}

func TestFlagOverridesConfigFile(t *testing.T) {
	tmpDir := t.TempDir()
	cfgFile := filepath.Join(tmpDir, "custom_config.json")
	content := `{"port": 8888}`
	if err := os.WriteFile(cfgFile, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	cfg := LoadWithArgs([]string{"--config=" + cfgFile, "--port=7070"})
	if cfg.Addr != ":7070" {
		t.Fatalf("expected flag --port to override config file, got %s", cfg.Addr)
	}
}

func TestEnsureDefaultConfigFile(t *testing.T) {
	tmpDir := t.TempDir()
	cfgFile := filepath.Join(tmpDir, "auto_config.json")

	if err := EnsureDefaultConfigFile(cfgFile); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(cfgFile); err != nil {
		t.Fatalf("expected auto_config.json to exist, err: %v", err)
	}

	cfg := LoadWithArgs([]string{"--config=" + cfgFile})
	if cfg.Addr != ":1080" {
		t.Fatalf("expected default :1080 from generated config, got %s", cfg.Addr)
	}
}
