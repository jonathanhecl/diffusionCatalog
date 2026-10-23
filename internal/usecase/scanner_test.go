package usecase

import (
	"context"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/metadata"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/storage/sqlite"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/vision"
)

func TestFormatETA(t *testing.T) {
	tests := []struct {
		secs     int64
		expected string
	}{
		{-1, "< 5s"},
		{0, "< 5s"},
		{5, "5s"},
		{59, "59s"},
		{60, "1m"},
		{75, "1m 15s"},
		{3600, "1h 00m"},
		{3665, "1h 01m"},
	}

	for _, tc := range tests {
		actual := formatETA(tc.secs)
		if actual != tc.expected {
			t.Errorf("formatETA(%d) = %q; want %q", tc.secs, actual, tc.expected)
		}
	}
}

func TestScannerLiveProgressAndETA(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	mediaDir := filepath.Join(tmpDir, "media")
	cacheDir := filepath.Join(tmpDir, "cache")
	dbPath := filepath.Join(tmpDir, "test.db")

	_ = os.MkdirAll(mediaDir, 0755)
	_ = os.MkdirAll(cacheDir, 0755)

	// Create 5 synthetic image files
	for i := 1; i <= 5; i++ {
		imgPath := filepath.Join(mediaDir, itoa64(int64(i))+".png")
		img := renderSyntheticArtwork(100, 100, color.RGBA{10, 20, 30, 255}, color.RGBA{40, 50, 60, 255}, i)
		f, err := os.Create(imgPath)
		if err != nil {
			t.Fatalf("os.Create error: %v", err)
		}
		if err := png.Encode(f, img); err != nil {
			f.Close()
			t.Fatalf("png.Encode error: %v", err)
		}
		f.Close()
	}

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("sqlite.Open error: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db, dbPath)
	extractor := metadata.NewExtractor()
	scanner := NewScannerService(mediaDir, cacheDir, repo, extractor, vision.NewNoopEngine())

	// Test progress recording unit logic
	scanner.status.TotalFiles = 100
	scanner.status.IsScanning = true
	scanner.status.Phase = "indexing"
	startTime := time.Now().Add(-2 * time.Second)

	scanned, total, pct := scanner.recordProgress("test.png", "media", false, startTime)
	if scanned != 1 || total != 100 || pct != 1 {
		t.Errorf("unexpected recordProgress output: scanned=%d, total=%d, pct=%d", scanned, total, pct)
	}

	st := scanner.Status()
	if st.Speed <= 0 {
		t.Errorf("expected positive speed, got %f", st.Speed)
	}
	if st.EtaSeconds <= 0 {
		t.Errorf("expected positive eta_seconds, got %d", st.EtaSeconds)
	}
	if st.EtaFormatted == "" {
		t.Errorf("expected non-empty eta_formatted")
	}

	// Now run full Scan
	report, err := scanner.Scan(ctx)
	if err != nil {
		t.Fatalf("scanner.Scan failed: %v", err)
	}
	if report.ScannedFiles != 5 {
		t.Errorf("expected 5 scanned files, got %d", report.ScannedFiles)
	}

	finalStatus := scanner.Status()
	if finalStatus.IsScanning {
		t.Errorf("expected IsScanning to be false after completion")
	}
	if finalStatus.Phase != "completed" {
		t.Errorf("expected Phase to be 'completed', got %q", finalStatus.Phase)
	}
	if finalStatus.ProgressPct != 100 {
		t.Errorf("expected ProgressPct to be 100, got %d", finalStatus.ProgressPct)
	}
	if finalStatus.EtaSeconds != 0 {
		t.Errorf("expected EtaSeconds to be 0 after completion, got %d", finalStatus.EtaSeconds)
	}
}
