package web

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/metadata"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/storage/sqlite"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/vision"
	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
	"github.com/jonathanhecl/diffusionCatalog/internal/usecase"
)

func TestCharacterEndpointsLifecycle(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	mediaDir := filepath.Join(tmpDir, "media")
	cacheDir := filepath.Join(tmpDir, "cache")
	dbPath := filepath.Join(tmpDir, "test.db")
	_ = os.MkdirAll(mediaDir, 0755)
	_ = os.MkdirAll(cacheDir, 0755)

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("failed to open sqlite: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db)
	extractor := metadata.NewExtractor()
	scannerSvc := usecase.NewScannerService(mediaDir, cacheDir, repo, extractor, vision.NewNoopEngine())
	authSvc, _ := usecase.NewAuthService(ctx, repo, "admin", "admin123")
	catalogSvc := usecase.NewCatalogService(mediaDir, repo, nil)

	srv, err := NewServer(mediaDir, cacheDir, "NoopVision", catalogSvc, scannerSvc, authSvc)
	if err != nil {
		t.Fatalf("failed to create server: %v", err)
	}

	handler := srv.Handler()

	// 1. Create Character with 2 base64 reference images
	tinyPng := "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
	createForm := url.Values{
		"name":             {"Elena Rostova"},
		"description":      {"Pale skin, silver shoulder-length straight hair, sharp hazel eyes, athletic build, small scar on left eyebrow."},
		"ref_image_data_0": {tinyPng},
		"ref_image_data_1": {tinyPng},
	}

	req := httptest.NewRequest("POST", "/admin/characters/create", strings.NewReader(createForm.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusSeeOther {
		t.Fatalf("expected 303 SeeOther on create character, got %d", rec.Code)
	}

	chars, err := repo.ListCharacters(ctx)
	if err != nil || len(chars) != 1 {
		t.Fatalf("expected 1 character in repo, got %d (err: %v)", len(chars), err)
	}
	ch := chars[0]
	if ch.Name != "Elena Rostova" {
		t.Errorf("expected name 'Elena Rostova', got %q", ch.Name)
	}
	if len(ch.ReferenceImages) != 2 {
		t.Fatalf("expected 2 reference images saved, got %d", len(ch.ReferenceImages))
	}

	// Verify files actually exist on disk in cacheDir/characters
	for _, refPath := range ch.ReferenceImages {
		if !strings.HasPrefix(refPath, "/cache/characters/") {
			t.Errorf("expected refPath to start with /cache/characters/, got %q", refPath)
		}
		rel := strings.TrimPrefix(refPath, "/cache/characters/")
		diskFile := filepath.Join(cacheDir, "characters", rel)
		if _, err := os.Stat(diskFile); os.IsNotExist(err) {
			t.Errorf("expected file on disk %s, but does not exist", diskFile)
		}
	}

	// 2. Update Character: update description and keep 1 reference image
	updateForm := url.Values{
		"id":                   {strconv.FormatInt(ch.ID, 10)},
		"name":                 {"Elena Rostova (Updated)"},
		"description":          {"Updated physical description: deep hazel eyes, braided silver hair."},
		"existing_ref_image_0": {ch.ReferenceImages[0]},
	}
	reqUpdate := httptest.NewRequest("POST", "/admin/characters/update", strings.NewReader(updateForm.Encode()))
	reqUpdate.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	recUpdate := httptest.NewRecorder()
	handler.ServeHTTP(recUpdate, reqUpdate)

	if recUpdate.Code != http.StatusSeeOther {
		t.Fatalf("expected 303 SeeOther on update character, got %d", recUpdate.Code)
	}

	updatedCh, err := repo.GetCharacterByID(ctx, ch.ID)
	if err != nil {
		t.Fatalf("failed to retrieve updated character: %v", err)
	}
	if updatedCh.Name != "Elena Rostova (Updated)" {
		t.Errorf("expected updated name, got %q", updatedCh.Name)
	}
	if !strings.Contains(updatedCh.Description, "braided silver hair") {
		t.Errorf("expected updated description, got %q", updatedCh.Description)
	}
	if len(updatedCh.ReferenceImages) != 1 {
		t.Errorf("expected 1 reference image after update, got %d", len(updatedCh.ReferenceImages))
	}

	// 3. Delete Character: should delete character and clean up reference files
	delForm := url.Values{
		"id": {strconv.FormatInt(ch.ID, 10)},
	}
	reqDel := httptest.NewRequest("POST", "/admin/characters/delete", strings.NewReader(delForm.Encode()))
	reqDel.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	recDel := httptest.NewRecorder()
	handler.ServeHTTP(recDel, reqDel)

	if recDel.Code != http.StatusSeeOther {
		t.Fatalf("expected 303 SeeOther on delete character, got %d", recDel.Code)
	}

	charsAfter, err := repo.ListCharacters(ctx)
	if err != nil || len(charsAfter) != 0 {
		t.Errorf("expected 0 characters after delete, got %d", len(charsAfter))
	}
}

type mockMultiVisionOllama struct {
	lastCall       string
	lastSources    []string
	lastUserPrompt string
}

func (m *mockMultiVisionOllama) CheckStatus(ctx context.Context, baseURL string) (string, bool) {
	return "1.0", true
}
func (m *mockMultiVisionOllama) ListModels(ctx context.Context, baseURL string) ([]domain.OllamaModelInfo, error) {
	return nil, nil
}
func (m *mockMultiVisionOllama) CheckOMLXStatus(ctx context.Context, baseURL, apiKey string) (bool, int) {
	return false, 0
}
func (m *mockMultiVisionOllama) ListOMLXModels(ctx context.Context, baseURL, apiKey string) ([]domain.OllamaModelInfo, error) {
	return nil, nil
}
func (m *mockMultiVisionOllama) ImageToPrompt(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt, imageSource, userHint, thinkLevel string) (string, error) {
	m.lastCall = "ImageToPrompt"
	m.lastSources = []string{imageSource}
	m.lastUserPrompt = userHint
	return "single image prompt", nil
}
func (m *mockMultiVisionOllama) ImagesToPrompt(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt string, imageSources []string, userHint, thinkLevel string) (string, error) {
	m.lastCall = "ImagesToPrompt"
	m.lastSources = imageSources
	m.lastUserPrompt = userHint
	return "multi image prompt with character replacement", nil
}
func (m *mockMultiVisionOllama) TextToPrompt(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt, inputConcept, thinkLevel string) (string, error) {
	m.lastCall = "TextToPrompt"
	m.lastUserPrompt = inputConcept
	return "text prompt with character replacement", nil
}
func (m *mockMultiVisionOllama) AnalyzeImage(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt, imageSource, userPrompt, thinkLevel string) (string, error) {
	return "", nil
}
func (m *mockMultiVisionOllama) AnalyzeImages(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt string, imageSources []string, userPrompt, thinkLevel string) (string, error) {
	return "", nil
}
func (m *mockMultiVisionOllama) EmbedText(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, text string) ([]float32, error) {
	return nil, nil
}
func (m *mockMultiVisionOllama) EmbedImage(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model string, imageBytes []byte) ([]float32, error) {
	return nil, nil
}

func TestCharacterPromptReplacementDirective(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	mediaDir := filepath.Join(tmpDir, "media")
	cacheDir := filepath.Join(tmpDir, "cache")
	dbPath := filepath.Join(tmpDir, "test_prompt.db")
	_ = os.MkdirAll(mediaDir, 0755)
	_ = os.MkdirAll(cacheDir, 0755)

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("failed to open sqlite: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db)
	mockAI := &mockMultiVisionOllama{}
	catalogSvc := usecase.NewCatalogService(mediaDir, repo, mockAI)
	catalogSvc.SetCacheDir(cacheDir)

	// Create a reference character with 1 image
	refFile1 := filepath.Join(cacheDir, "characters", "char1_ref1.png")
	_ = os.MkdirAll(filepath.Dir(refFile1), 0755)
	_ = os.WriteFile(refFile1, []byte("fake image 1"), 0644)

	charID, err := catalogSvc.CreateCharacter(ctx, &domain.Character{
		Name:            "Vesper-9",
		Description:     "Silver prosthetic cyber-optic right eye, cropped dark buzzcut, sharp angular jawline, tall lean build.",
		ReferenceImages: []string{"/cache/characters/char1_ref1.png"},
	}, 0)
	if err != nil {
		t.Fatalf("failed to create character: %v", err)
	}

	// 1. Test Vision mode: with base scene image + character reference image -> calls ImagesToPrompt
	sceneFile := filepath.Join(tmpDir, "base_scene.jpg")
	_ = os.WriteFile(sceneFile, []byte("fake scene"), 0644)

	out, _, err := catalogSvc.RunImage2Prompt(ctx, 0, "flux", "vision", "cyberpunk rain street", charID, nil, nil, domain.PromptExecutionOverride{}, false, sceneFile)
	if err != nil {
		t.Fatalf("RunImage2Prompt failed: %v", err)
	}
	if out != "multi image prompt with character replacement" {
		t.Errorf("unexpected out prompt: %s", out)
	}
	if mockAI.lastCall != "ImagesToPrompt" {
		t.Errorf("expected ImagesToPrompt to be called, got %s", mockAI.lastCall)
	}
	if len(mockAI.lastSources) != 2 {
		t.Errorf("expected 2 image sources (base scene + char ref), got %d: %v", len(mockAI.lastSources), mockAI.lastSources)
	}

	// Verify directives inside prompt
	if !strings.Contains(mockAI.lastUserPrompt, "PRESERVE THE SCENE") {
		t.Errorf("expected prompt directive to include 'PRESERVE THE SCENE', got:\n%s", mockAI.lastUserPrompt)
	}
	if !strings.Contains(mockAI.lastUserPrompt, "PRESERVE WARDROBE") {
		t.Errorf("expected prompt directive to include 'PRESERVE WARDROBE', got:\n%s", mockAI.lastUserPrompt)
	}
	if !strings.Contains(mockAI.lastUserPrompt, "Vesper-9") {
		t.Errorf("expected character name in prompt, got:\n%s", mockAI.lastUserPrompt)
	}

	// 2. Test Text-only remix mode: calls TextToPrompt with character sheet and directives
	outText, _, err := catalogSvc.RunImage2Prompt(ctx, 0, "flux", "remix", "cyberpunk rain street", charID, nil, nil, domain.PromptExecutionOverride{}, false, "")
	if err != nil {
		t.Fatalf("RunImage2Prompt (remix) failed: %v", err)
	}
	if outText != "text prompt with character replacement" {
		t.Errorf("unexpected outText: %s", outText)
	}
	if mockAI.lastCall != "TextToPrompt" {
		t.Errorf("expected TextToPrompt call, got %s", mockAI.lastCall)
	}
	if !strings.Contains(mockAI.lastUserPrompt, "PRESERVE WARDROBE") {
		t.Errorf("expected remix prompt to contain wardrobe preservation directive, got:\n%s", mockAI.lastUserPrompt)
	}
}
