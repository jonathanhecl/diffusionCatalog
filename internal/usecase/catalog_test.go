package usecase

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
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
	lastCall        string
	analyzeResponse string
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
func (m *mockOllamaProvider) ImagesToPrompt(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt string, imageSources []string, userHint, thinkLevel string) (string, error) {
	m.lastCall = "ImageToPrompt"
	return "vision prompt", nil
}
func (m *mockOllamaProvider) TextToPrompt(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt, inputConcept, thinkLevel string) (string, error) {
	m.lastCall = "TextToPrompt"
	return "remix text prompt", nil
}
func (m *mockOllamaProvider) AnalyzeImage(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt, imageSource, userPrompt, thinkLevel string) (string, error) {
	m.lastCall = "AnalyzeImage"
	if m.analyzeResponse != "" {
		return m.analyzeResponse, nil
	}
	return `{"aspectRatio":"16:9","canvasWidth":1024,"canvasHeight":576,"elements":[{"id":"el_1","type":"character","name":"Hero","x":500,"y":300,"poseName":"Standing Idle"}]}`, nil
}
func (m *mockOllamaProvider) AnalyzeImages(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt string, imageSources []string, userPrompt, thinkLevel string) (string, error) {
	m.lastCall = "AnalyzeImages"
	return "", nil
}
func (m *mockOllamaProvider) EmbedText(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, text string) ([]float32, error) {
	return nil, nil
}
func (m *mockOllamaProvider) EmbedImage(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model string, imageBytes []byte) ([]float32, error) {
	return nil, nil
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
	out, _, err := catalogSvc.RunImage2Prompt(ctx, assetID, "flux", "remix", "make it neon", 0, nil, domain.PromptExecutionOverride{}, false, "")
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
	outVision, _, err := catalogSvc.RunImage2Prompt(ctx, assetID, "flux", "vision", "inspect visual details", 0, nil, domain.PromptExecutionOverride{}, false, filepath.Join(cacheDir, "fake_frame.jpg"))
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

func TestSceneDirectorStagingAndCompositionCRUD(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	mediaDir := filepath.Join(tmpDir, "media")
	dbPath := filepath.Join(tmpDir, "test_director.db")

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("failed to open sqlite db: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db)
	mockOllama := &mockOllamaProvider{}
	catalogSvc := NewCatalogService(mediaDir, repo, mockOllama)

	// 1. Verify default seeded poses are populated
	poses, err := catalogSvc.ListStudioPoses(ctx)
	if err != nil {
		t.Fatalf("failed to list studio poses: %v", err)
	}
	if len(poses) < 5 {
		t.Errorf("expected at least 5 default seeded poses, got %d", len(poses))
	}

	// 2. Test saving and retrieving a custom pose
	customPose := &domain.StudioPose{
		Name:        "Ninja Stance",
		Category:    "Action",
		PoseJSON:    `{"nose":{"x":45,"y":20}}`,
		Description: "Low crouched guard",
	}
	poseID, err := catalogSvc.UpsertStudioPose(ctx, customPose)
	if err != nil || poseID <= 0 {
		t.Fatalf("failed to save custom pose: %v", err)
	}

	// 3. Test CompileSceneDirectorManifest
	sampleSceneJSON := `{
		"aspectRatio": "16:9 widescreen",
		"canvasWidth": 1280,
		"canvasHeight": 720,
		"elements": [
			{
				"id": "char-1",
				"type": "character",
				"name": "Detective K",
				"features": "40yo weathered face, stubble beard",
				"hair": "messy dark hair with grey temples",
				"clothing": "classic beige trench coat over a rumpled suit",
				"details": "smoking a cigarette, fedora hat",
				"x": 200,
				"y": 550,
				"scale": 1.3,
				"facing": "three_quarter_left",
				"poseName": "Walking / Striding"
			},
			{
				"id": "prop-1",
				"type": "prop",
				"label": "Vintage Armchair",
				"subtype": "chair",
				"x": 950,
				"y": 360,
				"scale": 1.0
			},
			{
				"id": "light-1",
				"type": "light",
				"lightType": "Key Light",
				"x": 150,
				"y": 120,
				"rotation": 45,
				"color": "#fbbf24"
			},
			{
				"id": "note-1",
				"type": "note",
				"text": "Moody film noir, neon reflections on wet asphalt",
				"x": 100,
				"y": 50
			}
		]
	}`

	manifest, err := catalogSvc.CompileSceneDirectorManifest(ctx, sampleSceneJSON)
	if err != nil {
		t.Fatalf("CompileSceneDirectorManifest failed: %v", err)
	}

	if !strings.Contains(manifest, "Detective K") || !strings.Contains(manifest, "Foreground-Left") {
		t.Errorf("expected character in Foreground-Left, manifest:\n%s", manifest)
	}
	if !strings.Contains(manifest, "features: 40yo weathered face") || !strings.Contains(manifest, "clothing: classic beige trench coat") {
		t.Errorf("expected character features and clothing in manifest:\n%s", manifest)
	}
	if !strings.Contains(manifest, "hair: messy dark hair") || !strings.Contains(manifest, "details: smoking a cigarette") {
		t.Errorf("expected character hair and details in manifest:\n%s", manifest)
	}
	if !strings.Contains(manifest, "Vintage Armchair") || !strings.Contains(manifest, "Midground-Right") {
		t.Errorf("expected prop in Midground-Right, manifest:\n%s", manifest)
	}
	if !strings.Contains(manifest, "Moody film noir") {
		t.Errorf("expected director note in manifest:\n%s", manifest)
	}

	// 3b. Test character with separate anatomy, expression, and custom pose name
	anatomySceneJSON := `{
		"aspectRatio": "4:3",
		"elements": [
			{
				"id": "char-2",
				"type": "character",
				"name": "Warrior Queen",
				"x": 200,
				"y": 400,
				"facing": "front",
				"poseName": "High Kick (Martial Arts)",
				"anatomy": "tall athletic muscular build, battle scar across left cheek",
				"expression": "fierce determined combat glare with clenched jaw",
				"hair": "long fiery red braids",
				"clothing": "ornate dragon plate armor"
			}
		]
	}`
	anatomyManifest, err := catalogSvc.CompileSceneDirectorManifest(ctx, anatomySceneJSON)
	if err != nil {
		t.Fatalf("CompileSceneDirectorManifest with anatomy and expression failed: %v", err)
	}
	if !strings.Contains(anatomyManifest, "anatomy: tall athletic muscular build") {
		t.Errorf("expected anatomy in manifest:\n%s", anatomyManifest)
	}
	if !strings.Contains(anatomyManifest, "expression: fierce determined combat glare") {
		t.Errorf("expected expression in manifest:\n%s", anatomyManifest)
	}
	if !strings.Contains(anatomyManifest, "High Kick (Martial Arts)") {
		t.Errorf("expected poseName in manifest:\n%s", anatomyManifest)
	}

	// 3c. Test scene with environment, situation, and time of day
	envSceneJSON := `{
		"aspectRatio": "16:9",
		"environmentType": "indoor",
		"environment": "hospital emergency room with medical equipment and sterile lighting",
		"timeOfDay": "dramatic golden hour sunset, warm fiery orange-gold glow",
		"elements": [
			{
				"id": "char-env",
				"type": "character",
				"name": "Doctor Ramirez",
				"x": 500,
				"y": 300,
				"facing": "front",
				"clothing": "medical scrubs uniform in hospital ceil blue with stethoscope"
			}
		]
	}`
	envManifest, err := catalogSvc.CompileSceneDirectorManifest(ctx, envSceneJSON)
	if err != nil {
		t.Fatalf("CompileSceneDirectorManifest with environment failed: %v", err)
	}
	if !strings.Contains(envManifest, "Setting & Environment: [INDOOR] hospital emergency room") {
		t.Errorf("expected indoor hospital setting in manifest:\n%s", envManifest)
	}
	if !strings.Contains(envManifest, "Time of Day & Atmosphere: dramatic golden hour sunset") {
		t.Errorf("expected golden hour atmosphere in manifest:\n%s", envManifest)
	}

	// 3d. Test character feature marker (body part modification / annotation like robotic arm)
	markerSceneJSON := `{
		"aspectRatio": "16:9",
		"elements": [
			{
				"id": "char-cyborg",
				"type": "character",
				"name": "Cyborg Agent",
				"x": 300,
				"y": 250,
				"facing": "front",
				"poseName": "Standing Idle"
			},
			{
				"id": "marker-arm",
				"type": "prop",
				"subtype": "circle",
				"markerRole": "character_feature",
				"targetCharacterId": "char-cyborg",
				"targetBodyPart": "Left Arm",
				"description": "advanced cybernetic prosthetic arm with chrome plating and exposed blue fiber optics",
				"x": 380,
				"y": 280
			}
		]
	}`
	markerManifest, err := catalogSvc.CompileSceneDirectorManifest(ctx, markerSceneJSON)
	if err != nil {
		t.Fatalf("CompileSceneDirectorManifest with feature marker failed: %v", err)
	}
	if !strings.Contains(markerManifest, "targeted body feature [Left Arm]: advanced cybernetic prosthetic arm") {
		t.Errorf("expected character traits to include targeted body feature on Left Arm, got:\n%s", markerManifest)
	}
	if !strings.Contains(markerManifest, "Character Targeted Features & Body Modifications:") {
		t.Errorf("expected section 'Character Targeted Features & Body Modifications:', got:\n%s", markerManifest)
	}
	if !strings.Contains(markerManifest, "[Cyborg Agent] -> [Left Arm]: advanced cybernetic prosthetic arm") {
		t.Errorf("expected feature line linking Cyborg Agent to Left Arm, got:\n%s", markerManifest)
	}
	if strings.Contains(markerManifest, "Environment, Props & Staging Elements:") {
		t.Errorf("expected feature marker to NOT be listed under Environment, Props & Staging Elements, got:\n%s", markerManifest)
	}

	// 4. Test SynthesizeScenePrompt (without canvas capture -> TextToPrompt)
	compiledManifest, generatedPrompt, _, err := catalogSvc.SynthesizeScenePrompt(ctx, sampleSceneJSON, "flux", "cyberpunk aesthetic", nil, domain.PromptExecutionOverride{}, "")
	if err != nil {
		t.Fatalf("SynthesizeScenePrompt failed: %v", err)
	}
	if compiledManifest == "" || generatedPrompt == "" {
		t.Errorf("expected non-empty manifest and prompt, got manifest=%q prompt=%q", compiledManifest, generatedPrompt)
	}
	if mockOllama.lastCall != "TextToPrompt" {
		t.Errorf("expected SynthesizeScenePrompt to invoke TextToPrompt, got %q", mockOllama.lastCall)
	}

	// 4b. Test SynthesizeScenePrompt with visual canvas capture -> ImageToPrompt
	dummyCanvasPath := filepath.Join(tmpDir, "dummy_canvas.png")
	if err := os.WriteFile(dummyCanvasPath, []byte("fake png content"), 0644); err != nil {
		t.Fatalf("failed to write dummy canvas: %v", err)
	}
	_, visionPrompt, _, err := catalogSvc.SynthesizeScenePrompt(ctx, sampleSceneJSON, "flux", "cyberpunk aesthetic", nil, domain.PromptExecutionOverride{}, dummyCanvasPath)
	if err != nil {
		t.Fatalf("SynthesizeScenePrompt with canvas image failed: %v", err)
	}
	if visionPrompt == "" {
		t.Errorf("expected non-empty prompt with canvas vision")
	}
	if mockOllama.lastCall != "ImageToPrompt" {
		t.Errorf("expected SynthesizeScenePrompt with canvasImagePath to invoke ImageToPrompt, got %q", mockOllama.lastCall)
	}

	// 4c. Test SynthesizeScenePrompt with in-memory canvas data URL (no disk temporary files)
	mockOllama.lastCall = ""
	inMemoryDataURL := "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
	_, inMemoryVisionPrompt, _, err := catalogSvc.SynthesizeScenePrompt(ctx, sampleSceneJSON, "flux", "cyberpunk aesthetic", nil, domain.PromptExecutionOverride{}, inMemoryDataURL)
	if err != nil {
		t.Fatalf("SynthesizeScenePrompt with in-memory data URL failed: %v", err)
	}
	if inMemoryVisionPrompt == "" {
		t.Errorf("expected non-empty prompt with in-memory canvas vision")
	}
	if mockOllama.lastCall != "ImageToPrompt" {
		t.Errorf("expected in-memory data URL to invoke ImageToPrompt, got %q", mockOllama.lastCall)
	}

	// 5. Test StudioComposition CRUD
	comp := &domain.StudioComposition{
		Name:          "Rainy Street Scene",
		Description:   "Film noir staging with Detective K",
		SceneJSON:     sampleSceneJSON,
		ThumbnailData: "data:image/png;base64,fake",
		PromptOutput:  generatedPrompt,
	}
	compID, err := catalogSvc.SaveStudioComposition(ctx, comp)
	if err != nil || compID <= 0 {
		t.Fatalf("failed to save composition: %v", err)
	}

	saved, err := catalogSvc.GetStudioComposition(ctx, compID)
	if err != nil || saved == nil {
		t.Fatalf("failed to retrieve saved composition: %v", err)
	}
	if saved.Name != "Rainy Street Scene" {
		t.Errorf("expected composition name 'Rainy Street Scene', got %q", saved.Name)
	}

	list, err := catalogSvc.ListStudioCompositions(ctx)
	if err != nil || len(list) != 1 {
		t.Errorf("expected 1 saved composition in list, got %d", len(list))
	}

	if err := catalogSvc.DeleteStudioComposition(ctx, compID); err != nil {
		t.Fatalf("failed to delete composition: %v", err)
	}
	listAfter, _ := catalogSvc.ListStudioCompositions(ctx)
	if len(listAfter) != 0 {
		t.Errorf("expected 0 compositions after deletion, got %d", len(listAfter))
	}
}

func TestDeleteAsset(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	mediaDir := filepath.Join(tmpDir, "media")
	_ = os.MkdirAll(mediaDir, 0755)
	dbPath := filepath.Join(tmpDir, "test_delete.db")

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("failed to open sqlite: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db)
	catalogSvc := NewCatalogService(mediaDir, repo, nil)

	// Create test file on disk and companion .txt
	filePath := filepath.Join(mediaDir, "cyberpunk_art.png")
	txtPath := filepath.Join(mediaDir, "cyberpunk_art.txt")
	if err := os.WriteFile(filePath, []byte("fake image data"), 0644); err != nil {
		t.Fatalf("failed to write test image: %v", err)
	}
	if err := os.WriteFile(txtPath, []byte("cyberpunk prompt"), 0644); err != nil {
		t.Fatalf("failed to write test txt: %v", err)
	}

	asset := &domain.MediaAsset{
		RelPath:   "cyberpunk_art.png",
		FileName:  "cyberpunk_art.png",
		FileHash:  "aabbcc112233",
		MediaType: domain.MediaTypeImage,
		Width:     1024,
		Height:    1024,
		FileSize:  15,
	}
	assetID, err := repo.UpsertAsset(ctx, asset)
	if err != nil || assetID <= 0 {
		t.Fatalf("failed to insert asset: %v", err)
	}

	// Verify asset exists in DB and files exist on disk
	retrieved, err := catalogSvc.GetAsset(ctx, assetID)
	if err != nil || retrieved == nil {
		t.Fatalf("failed to retrieve asset before delete")
	}

	// Delete asset via catalog service
	if err := catalogSvc.DeleteAsset(ctx, assetID); err != nil {
		t.Fatalf("DeleteAsset failed: %v", err)
	}

	// Verify file is gone from disk
	if _, err := os.Stat(filePath); !os.IsNotExist(err) {
		t.Errorf("expected media file to be deleted from disk")
	}

	// Verify companion .txt is gone from disk
	if _, err := os.Stat(txtPath); !os.IsNotExist(err) {
		t.Errorf("expected companion .txt to be deleted from disk")
	}

	// Verify record is gone from DB
	after, err := repo.GetAssetByID(ctx, assetID)
	if err == nil && after != nil {
		t.Errorf("expected asset to be deleted from database, but found: %+v", after)
	}
}

func TestResolveExecutionRoute(t *testing.T) {
	catalogSvc := &CatalogService{}
	cfg := domain.AppSettings{
		OllamaBaseURL: "http://127.0.0.1:11434",
		OMLXBaseURL:   "http://127.0.0.1:8000",
	}

	// 1. Standard Ollama model
	p, m, url, th := catalogSvc.ResolveExecutionRoute(cfg, domain.ProviderOllama, "llama3:latest", "default", domain.PromptExecutionOverride{})
	if p != domain.ProviderOllama || m != "llama3:latest" || url != "http://127.0.0.1:11434" || th != "default" {
		t.Errorf("unexpected standard ollama resolution: %v, %v, %v, %v", p, m, url, th)
	}

	// 2. MLX model configured with ollama provider (should auto-detect OpenAI/oMLX provider)
	p, m, url, th = catalogSvc.ResolveExecutionRoute(cfg, domain.ProviderOllama, "SuperQwen3.8-27b-abliterated-MLX-4bit", "default", domain.PromptExecutionOverride{})
	if p != domain.ProviderOpenAI {
		t.Errorf("expected auto-detected ProviderOpenAI for MLX model, got %v", p)
	}
	if url != "http://127.0.0.1:8000" {
		t.Errorf("expected OMLXBaseURL http://127.0.0.1:8000, got %v", url)
	}

	// 3. Override provider & model & thinking
	ov := domain.PromptExecutionOverride{
		Provider:   domain.ProviderOpenAI,
		Model:      "custom-mlx-model",
		ThinkLevel: "medium",
	}
	p, m, url, th = catalogSvc.ResolveExecutionRoute(cfg, domain.ProviderOllama, "llama3:latest", "off", ov)
	if p != domain.ProviderOpenAI || m != "custom-mlx-model" || url != "http://127.0.0.1:8000" || th != "medium" {
		t.Errorf("unexpected override resolution: %v, %v, %v, %v", p, m, url, th)
	}
}

type countingMockOllamaProvider struct {
	checkStatusCalls int
	mu               sync.Mutex
}

func (m *countingMockOllamaProvider) CheckStatus(ctx context.Context, baseURL string) (string, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.checkStatusCalls++
	return "0.1.32", true
}

func (m *countingMockOllamaProvider) ListModels(ctx context.Context, baseURL string) ([]domain.OllamaModelInfo, error) {
	return []domain.OllamaModelInfo{
		{Name: "llama3:latest", Provider: domain.ProviderOllama},
	}, nil
}

func (m *countingMockOllamaProvider) CheckOMLXStatus(ctx context.Context, baseURL, apiKey string) (bool, int) {
	return true, 1
}

func (m *countingMockOllamaProvider) ListOMLXModels(ctx context.Context, baseURL, apiKey string) ([]domain.OllamaModelInfo, error) {
	return []domain.OllamaModelInfo{
		{Name: "mlx-qwen:latest", Provider: domain.ProviderOpenAI},
	}, nil
}

func (m *countingMockOllamaProvider) ImageToPrompt(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt, absImagePath, userHint, thinkLevel string) (string, error) {
	return "", nil
}
func (m *countingMockOllamaProvider) ImagesToPrompt(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt string, imageSources []string, userHint, thinkLevel string) (string, error) {
	return "", nil
}

func (m *countingMockOllamaProvider) TextToPrompt(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt, inputConcept, thinkLevel string) (string, error) {
	return "", nil
}

func (m *countingMockOllamaProvider) AnalyzeImage(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt, imageSource, userPrompt, thinkLevel string) (string, error) {
	return "", nil
}
func (m *countingMockOllamaProvider) AnalyzeImages(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt string, imageSources []string, userPrompt, thinkLevel string) (string, error) {
	return "", nil
}
func (m *countingMockOllamaProvider) EmbedText(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, text string) ([]float32, error) {
	return nil, nil
}
func (m *countingMockOllamaProvider) EmbedImage(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model string, imageBytes []byte) ([]float32, error) {
	return nil, nil
}

func TestInspectAIBackendsCachingAndInvalidation(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	mediaDir := filepath.Join(tmpDir, "media")
	dbPath := filepath.Join(tmpDir, "test_cache.db")

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("failed to open sqlite: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db)
	mock := &countingMockOllamaProvider{}

	// Configure settings to enable Ollama
	_ = repo.SaveSettings(ctx, domain.AppSettings{
		OllamaEnabled: true,
		OllamaBaseURL: "http://127.0.0.1:11434",
		OpenAIEnabled: true,
		OpenAIBaseURL: "http://127.0.0.1:8000",
	})

	catalogSvc := NewCatalogService(mediaDir, repo, mock)

	// First call should probe servers
	st1 := catalogSvc.InspectAIBackendsFresh(ctx)
	if !st1.OllamaOnline || len(st1.OllamaModels) != 1 {
		t.Fatalf("expected st1 to have 1 ollama model and be online, got: %+v", st1)
	}
	mock.mu.Lock()
	initialCalls := mock.checkStatusCalls
	mock.mu.Unlock()

	if initialCalls == 0 {
		t.Fatalf("expected at least 1 CheckStatus call, got 0")
	}

	// Second call should hit the 1-minute cache without probing again
	st2 := catalogSvc.InspectAIBackends(ctx)
	if !st2.OllamaOnline || len(st2.AllModels) != 2 {
		t.Fatalf("expected cached status with 2 all models, got %+v", st2)
	}

	mock.mu.Lock()
	callsAfterCacheHit := mock.checkStatusCalls
	mock.mu.Unlock()

	if callsAfterCacheHit != initialCalls {
		t.Errorf("expected cache hit to NOT increment checkStatusCalls: before=%d, after=%d", initialCalls, callsAfterCacheHit)
	}

	// Invalidate cache
	catalogSvc.InvalidateAICache()

	// Third call should re-probe
	st3 := catalogSvc.InspectAIBackendsFresh(ctx)
	if !st3.OllamaOnline {
		t.Fatalf("expected st3 to be online after re-probe")
	}

	mock.mu.Lock()
	callsAfterInvalidate := mock.checkStatusCalls
	mock.mu.Unlock()

	if callsAfterInvalidate <= callsAfterCacheHit {
		t.Errorf("expected checkStatusCalls to increase after cache invalidation: was %d, now %d", callsAfterCacheHit, callsAfterInvalidate)
	}
}

func TestImageToScene(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	mediaDir := filepath.Join(tmpDir, "media")
	dbPath := filepath.Join(tmpDir, "test_i2s.db")

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("failed to open sqlite: %v", err)
	}
	defer db.Close()

	mockOllama := &mockOllamaProvider{
		analyzeResponse: `
		` + "```json" + `
		{
			"aspectRatio": "16:9",
			"canvasWidth": 1024,
			"canvasHeight": 576,
			"perspective": "front",
			"environmentType": "outdoor",
			"environment": "Rainy cyberpunk alleyway with neon signs",
			"timeOfDay": "night",
			"elements": [
				{
					"type": "character",
					"subtype": "female",
					"name": "Detective",
					"x": 400,
					"y": 300,
					"poseName": "Walking / Striding",
					"facing": "three_quarter_left"
				},
				{
					"type": "prop",
					"subtype": "car",
					"label": "Hovercar",
					"x": 800,
					"y": 450
				},
				{
					"type": "light",
					"lightType": "Key Light",
					"color": "#38bdf8",
					"x": 200,
					"y": 150,
					"rotation": 45
				}
			]
		}
		` + "```",
	}

	repo := sqlite.NewRepository(db)
	catalogSvc := NewCatalogService(mediaDir, repo, mockOllama)
	_ = catalogSvc.SaveSettings(ctx, domain.AppSettings{
		RemixDefaultProvider: domain.ProviderOllama,
		VisionModel:          "llava:latest",
		OllamaDefaultModel:   "llava:latest",
	})

	dummyImage := "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
	sceneJSON, scene, err := catalogSvc.ImageToScene(ctx, dummyImage, "cyberpunk tone", "16:9", 1024, 576, domain.PromptExecutionOverride{})
	if err != nil {
		t.Fatalf("ImageToScene failed: %v", err)
	}

	if mockOllama.lastCall != "AnalyzeImage" {
		t.Errorf("expected AnalyzeImage to be called, got %q", mockOllama.lastCall)
	}

	if scene == nil {
		t.Fatalf("expected non-nil scene")
	}

	if scene.AspectRatio != "16:9" || scene.CanvasWidth != 1024 || scene.CanvasHeight != 576 {
		t.Errorf("expected enforced aspect ratio 16:9 (1024x576), got %s (%fx%f)", scene.AspectRatio, scene.CanvasWidth, scene.CanvasHeight)
	}

	if scene.EnvironmentType != "outdoor" || scene.TimeOfDay != "night" {
		t.Errorf("unexpected scene environment: %+v", scene)
	}

	if len(scene.Elements) != 3 {
		t.Fatalf("expected 3 elements, got %d", len(scene.Elements))
	}

	char := scene.Elements[0]
	if char.Type != "character" || char.Subtype != "female" || char.PoseName != "Walking / Striding" {
		t.Errorf("unexpected character element: %+v", char)
	}
	if char.ID == "" {
		t.Errorf("expected auto-generated element ID")
	}

	if !strings.Contains(sceneJSON, "Detective") {
		t.Errorf("expected sceneJSON to contain 'Detective', got %s", sceneJSON)
	}
}

