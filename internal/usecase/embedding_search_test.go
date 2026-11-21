package usecase_test

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/gif"
	"image/jpeg"
	"image/png"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/metadata"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/storage/sqlite"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/vision"
	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
	"github.com/jonathanhecl/diffusionCatalog/internal/usecase"
)

type embeddingMockOllama struct {
	embedImageCalls    int
	embedTextCalls     int
	lastImageModel     string
	lastTextModel      string
	receivedImageBytes [][]byte
	embedImageFn       func(imageBytes []byte) ([]float32, error)
}

func (m *embeddingMockOllama) CheckStatus(ctx context.Context, baseURL string) (string, bool) {
	return "1.0", true
}
func (m *embeddingMockOllama) ListModels(ctx context.Context, baseURL string) ([]domain.OllamaModelInfo, error) {
	return nil, nil
}
func (m *embeddingMockOllama) CheckOMLXStatus(ctx context.Context, baseURL, apiKey string) (bool, int) {
	return false, 0
}
func (m *embeddingMockOllama) ListOMLXModels(ctx context.Context, baseURL, apiKey string) ([]domain.OllamaModelInfo, error) {
	return nil, nil
}
func (m *embeddingMockOllama) ImageToPrompt(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt, imageSource, userHint, thinkLevel string) (string, error) {
	return "", nil
}
func (m *embeddingMockOllama) ImagesToPrompt(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt string, imageSources []string, userHint, thinkLevel string) (string, error) {
	return "", nil
}
func (m *embeddingMockOllama) TextToPrompt(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt, inputConcept, thinkLevel string) (string, error) {
	return "", nil
}
func (m *embeddingMockOllama) AnalyzeImage(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt, imageSource, userPrompt, thinkLevel string) (string, error) {
	return "", nil
}
func (m *embeddingMockOllama) AnalyzeImages(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt string, imageSources []string, userPrompt, thinkLevel string) (string, error) {
	return "", nil
}

func (m *embeddingMockOllama) EmbedText(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, text string) ([]float32, error) {
	m.embedTextCalls++
	m.lastTextModel = model
	if text == "gato" {
		return []float32{1.0, 0.0, 0.0}, nil
	}
	return []float32{0.0, 1.0, 0.0}, nil
}

func (m *embeddingMockOllama) EmbedImage(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model string, imageBytes []byte) ([]float32, error) {
	m.embedImageCalls++
	m.lastImageModel = model
	cp := make([]byte, len(imageBytes))
	copy(cp, imageBytes)
	m.receivedImageBytes = append(m.receivedImageBytes, cp)
	if m.embedImageFn != nil {
		return m.embedImageFn(imageBytes)
	}
	// Mock returns vector pointing to [1.0, 0.0, 0.0]
	return []float32{0.99, 0.01, 0.0}, nil
}

func TestScannerAndCatalogEmbeddingIntegration(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	mediaDir := filepath.Join(tmpDir, "media")
	cacheDir := filepath.Join(tmpDir, ".cache")
	dbPath := filepath.Join(tmpDir, "catalog.db")

	_ = os.MkdirAll(mediaDir, 0755)
	_ = os.MkdirAll(cacheDir, 0755)

	// Create test image
	testImgPath := filepath.Join(mediaDir, "sample_feline.png")
	_ = os.WriteFile(testImgPath, []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n', 0, 0, 0, 0}, 0644)

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("failed to open sqlite: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db, dbPath)
	mockOllama := &embeddingMockOllama{}

	scannerSvc := usecase.NewScannerService(mediaDir, cacheDir, repo, metadata.NewExtractor(), vision.NewNoopEngine())
	scannerSvc.SetOllamaProvider(mockOllama)
	catalogSvc := usecase.NewCatalogService(mediaDir, repo, mockOllama)

	// 1. Scan without embedding configured (should NOT embed)
	_, err = scannerSvc.Scan(ctx)
	if err != nil {
		t.Fatalf("initial scan failed: %v", err)
	}
	if mockOllama.embedImageCalls != 0 {
		t.Fatalf("expected 0 EmbedImage calls when disabled, got %d", mockOllama.embedImageCalls)
	}

	// 2. Enable embedding in settings
	_ = repo.SaveSettings(ctx, domain.AppSettings{
		OllamaEnabled:    true,
		OllamaBaseURL:    "http://127.0.0.1:11434",
		EmbeddingEnabled: true,
		EmbeddingModel:   "embeddinggemma-2",
	})

	// 3. Scan with embedding enabled: should embed the image
	_, err = scannerSvc.Scan(ctx)
	if err != nil {
		t.Fatalf("scan with embedding failed: %v", err)
	}
	if mockOllama.embedImageCalls != 1 {
		t.Fatalf("expected 1 EmbedImage call, got %d", mockOllama.embedImageCalls)
	}

	// 4. Scan again without model change: should SKIP already embedded image
	_, err = scannerSvc.Scan(ctx)
	if err != nil {
		t.Fatalf("second scan failed: %v", err)
	}
	if mockOllama.embedImageCalls != 1 {
		t.Fatalf("expected EmbedImage calls to stay 1 (skipped), got %d", mockOllama.embedImageCalls)
	}

	// 5. Query catalog by term "gato": should match via vector similarity
	page, err := catalogSvc.Query(ctx, domain.CatalogFilter{
		Query: "gato",
	})
	if err != nil {
		t.Fatalf("catalog query failed: %v", err)
	}
	if mockOllama.embedTextCalls != 1 {
		t.Fatalf("expected 1 EmbedText call for query, got %d", mockOllama.embedTextCalls)
	}
	if len(page.Assets) != 1 {
		t.Fatalf("expected 1 asset returned by embedding match, got %d", len(page.Assets))
	}

	// 6. Change embedding model: existing embedding must be invalidated and re-embedded
	_ = repo.SaveSettings(ctx, domain.AppSettings{
		OllamaEnabled:    true,
		OllamaBaseURL:    "http://127.0.0.1:11434",
		EmbeddingEnabled: true,
		EmbeddingModel:   "embeddinggemma-2-new",
	})
	_, err = scannerSvc.Scan(ctx)
	if err != nil {
		t.Fatalf("scan after model change failed: %v", err)
	}
	if mockOllama.embedImageCalls != 2 {
		t.Fatalf("expected EmbedImage calls to increment to 2 upon model change, got %d", mockOllama.embedImageCalls)
	}
	if mockOllama.lastImageModel != "embeddinggemma-2-new" {
		t.Errorf("expected lastImageModel embeddinggemma-2-new, got %s", mockOllama.lastImageModel)
	}
}

func setupEmbeddingTestEnv(t *testing.T) (context.Context, string, domain.CatalogRepository, *usecase.ScannerService, *usecase.CatalogService, *embeddingMockOllama) {
	t.Helper()
	ctx := context.Background()
	tmpDir := t.TempDir()
	mediaDir := filepath.Join(tmpDir, "media")
	cacheDir := filepath.Join(tmpDir, ".cache")
	dbPath := filepath.Join(tmpDir, "catalog.db")

	_ = os.MkdirAll(mediaDir, 0755)
	_ = os.MkdirAll(cacheDir, 0755)

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("failed to open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	repo := sqlite.NewRepository(db, dbPath)
	_ = repo.SaveSettings(ctx, domain.AppSettings{
		OllamaEnabled:    true,
		OllamaBaseURL:    "http://127.0.0.1:11434",
		EmbeddingEnabled: true,
		EmbeddingModel:   "embeddinggemma-2",
	})

	mockOllama := &embeddingMockOllama{}
	scannerSvc := usecase.NewScannerService(mediaDir, cacheDir, repo, metadata.NewExtractor(), vision.NewNoopEngine())
	scannerSvc.SetOllamaProvider(mockOllama)
	catalogSvc := usecase.NewCatalogService(mediaDir, repo, mockOllama)
	return ctx, mediaDir, repo, scannerSvc, catalogSvc, mockOllama
}

func assertRedJPEGFrame(t *testing.T, data []byte, expectedW, expectedH int) {
	t.Helper()
	if len(data) == 0 {
		t.Fatalf("expected non-empty JPEG frame bytes")
	}
	img, err := jpeg.Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("expected valid JPEG bytes for extracted first frame, got error: %v", err)
	}
	b := img.Bounds()
	if b.Dx() != expectedW || b.Dy() != expectedH {
		t.Fatalf("expected decoded frame size %dx%d, got %dx%d", expectedW, expectedH, b.Dx(), b.Dy())
	}
	r32, g32, b32, _ := img.At(b.Min.X+b.Dx()/2, b.Min.Y+b.Dy()/2).RGBA()
	r, g, bl := uint8(r32>>8), uint8(g32>>8), uint8(b32>>8)
	if r <= 200 || g >= 50 || bl >= 50 {
		t.Fatalf("expected center pixel of first frame to be RED, got RGB(%d, %d, %d)", r, g, bl)
	}
}

func TestScannerVideoFirstFrameEmbeddingGIF(t *testing.T) {
	ctx, mediaDir, repo, scannerSvc, catalogSvc, mockOllama := setupEmbeddingTestEnv(t)

	// Build a 2-frame 32x32 animated GIF: frame 0 solid RED, frame 1 solid BLUE
	palette := color.Palette{
		color.RGBA{R: 255, G: 0, B: 0, A: 255},
		color.RGBA{R: 0, G: 0, B: 255, A: 255},
	}
	frame0 := image.NewPaletted(image.Rect(0, 0, 32, 32), palette)
	frame1 := image.NewPaletted(image.Rect(0, 0, 32, 32), palette)
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			frame0.SetColorIndex(x, y, 0)
			frame1.SetColorIndex(x, y, 1)
		}
	}

	gifPath := filepath.Join(mediaDir, "animated_clip.gif")
	f, err := os.Create(gifPath)
	if err != nil {
		t.Fatalf("failed to create test gif: %v", err)
	}
	err = gif.EncodeAll(f, &gif.GIF{
		Image: []*image.Paletted{frame0, frame1},
		Delay: []int{10, 10},
	})
	_ = f.Close()
	if err != nil {
		t.Fatalf("failed to encode test gif: %v", err)
	}

	// 1. Initial scan: must extract frame 0 (RED) and embed it
	report, err := scannerSvc.Scan(ctx)
	if err != nil {
		t.Fatalf("scan failed: %v", err)
	}
	if report.VideosFound != 1 {
		t.Fatalf("expected 1 video found, got %d", report.VideosFound)
	}
	if mockOllama.embedImageCalls != 1 {
		t.Fatalf("expected 1 EmbedImage call for GIF video, got %d", mockOllama.embedImageCalls)
	}
	assertRedJPEGFrame(t, mockOllama.receivedImageBytes[0], 32, 32)

	// 2. Second scan without model change: must skip already-embedded video
	_, err = scannerSvc.Scan(ctx)
	if err != nil {
		t.Fatalf("second scan failed: %v", err)
	}
	if mockOllama.embedImageCalls != 1 {
		t.Fatalf("expected EmbedImage calls to remain 1 on unchanged rescan, got %d", mockOllama.embedImageCalls)
	}

	// 3. Semantic search (both general and media_type=video) must find the GIF asset
	page, err := catalogSvc.Query(ctx, domain.CatalogFilter{Query: "gato"})
	if err != nil {
		t.Fatalf("catalog query failed: %v", err)
	}
	if len(page.Assets) != 1 || page.Assets[0].MediaType != domain.MediaTypeVideo {
		t.Fatalf("expected 1 video asset from semantic search, got %+v", page.Assets)
	}

	videoPage, err := catalogSvc.Query(ctx, domain.CatalogFilter{
		Query:     "gato",
		MediaType: string(domain.MediaTypeVideo),
	})
	if err != nil {
		t.Fatalf("catalog video query failed: %v", err)
	}
	if len(videoPage.Assets) != 1 {
		t.Fatalf("expected 1 video asset with MediaType=video filter, got %d", len(videoPage.Assets))
	}

	// 4. Switching embedding model invalidates and re-embeds the video
	_ = repo.SaveSettings(ctx, domain.AppSettings{
		OllamaEnabled:    true,
		OllamaBaseURL:    "http://127.0.0.1:11434",
		EmbeddingEnabled: true,
		EmbeddingModel:   "embeddinggemma-2-v2",
	})
	_, err = scannerSvc.Scan(ctx)
	if err != nil {
		t.Fatalf("scan after model change failed: %v", err)
	}
	if mockOllama.embedImageCalls != 2 {
		t.Fatalf("expected EmbedImage calls to increment to 2 on model change, got %d", mockOllama.embedImageCalls)
	}
	if mockOllama.lastImageModel != "embeddinggemma-2-v2" {
		t.Errorf("expected lastImageModel embeddinggemma-2-v2, got %s", mockOllama.lastImageModel)
	}
}

func TestScannerVideoFirstFrameEmbeddingMP4WithFFmpeg(t *testing.T) {
	ffmpegPath, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Skip("ffmpeg not found in PATH; skipping MP4 first-frame integration test")
	}

	ctx, mediaDir, _, scannerSvc, catalogSvc, mockOllama := setupEmbeddingTestEnv(t)

	mp4Path := filepath.Join(mediaDir, "two_frame_clip.mp4")
	cmd := exec.CommandContext(ctx, ffmpegPath,
		"-y",
		"-hide_banner",
		"-loglevel", "error",
		"-f", "lavfi", "-i", "color=c=red:s=64x64:r=1:d=1",
		"-f", "lavfi", "-i", "color=c=blue:s=64x64:r=1:d=1",
		"-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
		"-map", "[v]",
		"-pix_fmt", "yuv420p",
		"-r", "1",
		mp4Path,
	)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("failed generating synthetic 2-frame MP4: %v (%s)", err, string(out))
	}

	report, err := scannerSvc.Scan(ctx)
	if err != nil {
		t.Fatalf("scan failed: %v", err)
	}
	if report.VideosFound != 1 {
		t.Fatalf("expected 1 video found, got %d", report.VideosFound)
	}
	if mockOllama.embedImageCalls != 1 {
		t.Fatalf("expected 1 EmbedImage call for MP4 video, got %d", mockOllama.embedImageCalls)
	}
	assertRedJPEGFrame(t, mockOllama.receivedImageBytes[0], 64, 64)

	// Verify semantic search returns the MP4 asset
	page, err := catalogSvc.Query(ctx, domain.CatalogFilter{
		Query:     "gato",
		MediaType: string(domain.MediaTypeVideo),
	})
	if err != nil {
		t.Fatalf("catalog query failed: %v", err)
	}
	if len(page.Assets) != 1 || page.Assets[0].FileName != "two_frame_clip.mp4" {
		t.Fatalf("expected two_frame_clip.mp4 from semantic search, got %+v", page.Assets)
	}

	// Rescan must skip already-embedded MP4
	_, err = scannerSvc.Scan(ctx)
	if err != nil {
		t.Fatalf("rescan failed: %v", err)
	}
	if mockOllama.embedImageCalls != 1 {
		t.Fatalf("expected EmbedImage calls to stay 1 on rescan, got %d", mockOllama.embedImageCalls)
	}
}

func TestScannerVideoFirstFrameExtractionFailureDoesNotAbortScan(t *testing.T) {
	ctx, mediaDir, _, scannerSvc, _, mockOllama := setupEmbeddingTestEnv(t)

	// Corrupt MP4
	corruptMP4Path := filepath.Join(mediaDir, "corrupt_video.mp4")
	if err := os.WriteFile(corruptMP4Path, []byte("not-a-valid-mp4-stream"), 0644); err != nil {
		t.Fatalf("failed writing corrupt mp4: %v", err)
	}

	// Valid PNG
	validPNGPath := filepath.Join(mediaDir, "valid_image.png")
	img := image.NewRGBA(image.Rect(0, 0, 16, 16))
	f, err := os.Create(validPNGPath)
	if err != nil {
		t.Fatalf("failed creating valid png: %v", err)
	}
	if err := png.Encode(f, img); err != nil {
		_ = f.Close()
		t.Fatalf("failed encoding valid png: %v", err)
	}
	_ = f.Close()

	report, err := scannerSvc.Scan(ctx)
	if err != nil {
		t.Fatalf("expected scan to continue past corrupt video without error, got: %v", err)
	}
	if report.VideosFound != 1 || report.ImagesFound != 1 {
		t.Fatalf("expected 1 video and 1 image indexed, got %d videos, %d images", report.VideosFound, report.ImagesFound)
	}
	if mockOllama.embedImageCalls != 1 {
		t.Fatalf("expected exactly 1 EmbedImage call (valid PNG only), got %d", mockOllama.embedImageCalls)
	}
	if len(mockOllama.receivedImageBytes[0]) == 0 {
		t.Fatalf("expected non-empty bytes for valid PNG embedding")
	}
}
