package usecase

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/metadata"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/storage/sqlite"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/vision"
	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

func TestEndToEndSeedScanAndCatalogQuery(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	mediaDir := filepath.Join(tmpDir, "media")
	cacheDir := filepath.Join(tmpDir, ".cache")
	dbPath := filepath.Join(tmpDir, "test_catalog.db")

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("failed to open sqlite db: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db)

	authSvc, err := NewAuthService(ctx, repo, "admin", "admin123")
	if err != nil {
		t.Fatalf("failed to bootstrap default admin: %v", err)
	}
	token, err := authSvc.Authenticate(ctx, "admin", "admin123")
	if err != nil || token == "" {
		t.Fatalf("expected default admin login to succeed, got err=%v", err)
	}

	extractor := metadata.NewExtractor()
	scannerSvc := NewScannerService(mediaDir, cacheDir, repo, extractor, vision.NewNoopEngine())
	catalogSvc := NewCatalogService(mediaDir, repo, nil)

	if err := SeedDemoFolderIfEmpty(ctx, mediaDir, cacheDir, repo, scannerSvc); err != nil {
		t.Fatalf("failed to seed demo folder: %v", err)
	}

	// Verify seeded Settings and Generator System Prompt Profiles (flux, krea2, moss, minimaxh3, sdxl)
	settings, err := catalogSvc.GetSettings(ctx)
	if err != nil {
		t.Fatalf("failed to get settings: %v", err)
	}
	if settings.DefaultTargetGenerator != "flux" {
		t.Errorf("expected default generator 'flux', got %q", settings.DefaultTargetGenerator)
	}

	profiles, err := catalogSvc.ListPromptProfiles(ctx)
	if err != nil {
		t.Fatalf("failed to list prompt profiles: %v", err)
	}
	if len(profiles) < 6 {
		t.Fatalf("expected at least 6 seeded generator profiles (flux, krea2, qwen-image, moss, minimaxh3, sdxl), got %d", len(profiles))
	}

	loras, err := catalogSvc.ListModelLoRAs(ctx, "")
	if err != nil {
		t.Fatalf("failed to list model LoRAs: %v", err)
	}
	if len(loras) < 4 {
		t.Fatalf("expected at least 4 seeded model-specific LoRAs, got %d", len(loras))
	}
	ctxBlock := catalogSvc.buildLoRAContextBlock(ctx, loras[0].ProfileSlug, []int64{loras[0].ID})
	if ctxBlock == "" {
		t.Errorf("expected non-empty LoRA context block for prompt synthesis")
	}

	page, err := catalogSvc.Query(ctx, domain.CatalogFilter{PerPage: 50})
	if err != nil {
		t.Fatalf("failed to query catalog: %v", err)
	}
	if page.TotalCount != 6 {
		t.Fatalf("expected 6 seeded diffusion assets, got %d", page.TotalCount)
	}
	if len(page.Characters) != 3 {
		t.Fatalf("expected 3 character folders, got %d", len(page.Characters))
	}
	if len(page.Groups) != 3 {
		t.Fatalf("expected 3 auto-indexed folder groups, got %d", len(page.Groups))
	}

	// Verify Face Explorer mode
	facePage, err := catalogSvc.Query(ctx, domain.CatalogFilter{FaceMode: true, PerPage: 50})
	if err != nil {
		t.Fatalf("failed to query face explorer mode: %v", err)
	}
	if facePage.TotalCount != 6 {
		t.Fatalf("expected 6 extracted face crops, got %d", facePage.TotalCount)
	}
}

type mockOllamaProvider struct {
	lastCall string
}

func (m *mockOllamaProvider) CheckStatus(ctx context.Context, baseURL string) (string, bool) {
	return "1.0", true
}
func (m *mockOllamaProvider) ListModels(ctx context.Context, baseURL string) ([]domain.OllamaModelInfo, error) {
	return nil, nil
}
func (m *mockOllamaProvider) CheckOMLXStatus(ctx context.Context, baseURL, apiKey string) (bool, int) {
	return false, 0
}
func (m *mockOllamaProvider) ListOMLXModels(ctx context.Context, baseURL, apiKey string) ([]domain.OllamaModelInfo, error) {
	return nil, nil
}
func (m *mockOllamaProvider) ImageToPrompt(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt, absImagePath, userHint, thinkLevel string) (string, error) {
	m.lastCall = "ImageToPrompt"
	return "vision prompt", nil
}
func (m *mockOllamaProvider) TextToPrompt(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt, inputConcept, thinkLevel string) (string, error) {
	m.lastCall = "TextToPrompt"
	return "remix text prompt", nil
}

func TestRemixAndVisionOperationModeAndTargetGeneratorPersistence(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	mediaDir := filepath.Join(tmpDir, "media")
	cacheDir := filepath.Join(tmpDir, ".cache")
	dbPath := filepath.Join(tmpDir, "test_remix.db")

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("failed to open sqlite db: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db)
	mockOllama := &mockOllamaProvider{}
	catalogSvc := NewCatalogService(mediaDir, repo, mockOllama)

	// 1. Verify SetDefaultTargetGenerator persists target generator across calls
	if err := catalogSvc.SetDefaultTargetGenerator(ctx, "krea2"); err != nil {
		t.Fatalf("failed to set default target generator: %v", err)
	}
	settings, err := catalogSvc.GetSettings(ctx)
	if err != nil {
		t.Fatalf("failed to get settings: %v", err)
	}
	if settings.DefaultTargetGenerator != "krea2" {
		t.Errorf("expected default target generator 'krea2', got %q", settings.DefaultTargetGenerator)
	}

	// 2. Insert test asset
	rootID, _ := repo.UpsertMediaRoot(ctx, &domain.MediaRoot{Path: mediaDir, Label: "TestRoot"})
	assetID, err := repo.UpsertAsset(ctx, &domain.MediaAsset{
		RootID:    rootID,
		RelPath:   "test_video.mp4",
		FileName:  "test_video.mp4",
		MediaType: domain.MediaTypeVideo,
		Metadata: domain.GenerationParams{
			PositivePrompt: "cyberpunk skyline",
		},
	})
	if err != nil {
		t.Fatalf("failed to insert asset: %v", err)
	}

	// 3. Test mode == "remix" on a video asset: MUST use TextToPrompt, NOT ImageToPrompt
	out, err := catalogSvc.RunImage2Prompt(ctx, assetID, "flux", "remix", "make it neon", 0, nil, domain.PromptExecutionOverride{}, false, "")
	if err != nil {
		t.Fatalf("RunImage2Prompt remix mode failed: %v", err)
	}
	if out != "remix text prompt" {
		t.Errorf("unexpected output: %q", out)
	}
	if mockOllama.lastCall != "TextToPrompt" {
		t.Errorf("expected mode 'remix' to call TextToPrompt, got %q", mockOllama.lastCall)
	}

	// 4. Test mode == "vision" on an attached frame: MUST use ImageToPrompt
	outVision, err := catalogSvc.RunImage2Prompt(ctx, assetID, "flux", "vision", "inspect visual details", 0, nil, domain.PromptExecutionOverride{}, false, filepath.Join(cacheDir, "fake_frame.jpg"))
	if err != nil {
		t.Fatalf("RunImage2Prompt vision mode failed: %v", err)
	}
	if outVision != "vision prompt" {
		t.Errorf("unexpected output: %q", outVision)
	}
	if mockOllama.lastCall != "ImageToPrompt" {
		t.Errorf("expected mode 'vision' to call ImageToPrompt, got %q", mockOllama.lastCall)
	}
}
