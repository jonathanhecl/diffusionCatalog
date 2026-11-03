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
	if cfg.AutoScanInterval.Seconds() != 10 {
		t.Fatalf("expected default auto_scan_interval 10s, got %v", cfg.AutoScanInterval)
	}
}

func TestAutoScanIntervalConfig(t *testing.T) {
	tmpDir := t.TempDir()
	cfgFile := filepath.Join(tmpDir, "autoscan_config.json")
	content := `{
		"auto_scan": true,
		"auto_scan_interval": 15
	}`
	if err := os.WriteFile(cfgFile, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	cfg := LoadWithArgs([]string{"--config=" + cfgFile})
	if cfg.AutoScanInterval.Seconds() != 15 {
		t.Fatalf("expected auto_scan_interval 15s from int, got %v", cfg.AutoScanInterval)
	}

	// Override with flag
	cfg2 := LoadWithArgs([]string{"--config=" + cfgFile, "--auto-scan-interval=30s"})
	if cfg2.AutoScanInterval.Seconds() != 30 {
		t.Fatalf("expected auto_scan_interval 30s from flag, got %v", cfg2.AutoScanInterval)
	}

	// Disable auto-scan
	cfg3 := LoadWithArgs([]string{"--config=" + cfgFile, "--auto-scan=false"})
	if cfg3.AutoScanInterval != 0 {
		t.Fatalf("expected auto_scan_interval 0 when auto_scan=false, got %v", cfg3.AutoScanInterval)
	}
}

func TestConfigHTTPAddr(t *testing.T) {
	tests := []struct {
		addr string
		want string
	}{
		{addr: ":1080", want: "127.0.0.1:1080"},
		{addr: "0.0.0.0:1080", want: "127.0.0.1:1080"},
		{addr: "[::]:1080", want: "127.0.0.1:1080"},
		{addr: "127.0.0.1:1080", want: "127.0.0.1:1080"},
		{addr: "192.168.1.50:1080", want: "192.168.1.50:1080"},
		{addr: "localhost:1080", want: "localhost:1080"},
		{addr: "", want: "127.0.0.1:80"},
	}

	for _, tc := range tests {
		cfg := Config{Addr: tc.addr}
		if got := cfg.HTTPAddr(); got != tc.want {
			t.Errorf("HTTPAddr() for addr %q = %q, want %q", tc.addr, got, tc.want)
		}
	}
}
