package usecase_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/metadata"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/storage/sqlite"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/vision"
	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
	"github.com/jonathanhecl/diffusionCatalog/internal/usecase"
)

type embeddingMockOllama struct {
	embedImageCalls int
	embedTextCalls  int
	lastImageModel  string
	lastTextModel   string
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
