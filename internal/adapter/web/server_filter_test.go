package web_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/metadata"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/storage/sqlite"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/vision"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/web"
	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
	"github.com/jonathanhecl/diffusionCatalog/internal/usecase"
)

func setupTestServer(t *testing.T) (*web.Server, *sqlite.Repository, int64, int64) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	mediaDir := filepath.Join(tmpDir, "media")
	cacheDir := filepath.Join(tmpDir, "cache")
	dbPath := filepath.Join(tmpDir, "test_web.db")

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("sqlite.Open error: %v", err)
	}

	repo := sqlite.NewRepository(db)
	authSvc, _ := usecase.NewAuthService(ctx, repo, "admin", "admin123")
	extractor := metadata.NewExtractor()
	scannerSvc := usecase.NewScannerService(mediaDir, cacheDir, repo, extractor, vision.NewNoopEngine())
	catalogSvc := usecase.NewCatalogService(mediaDir, repo, nil)

	now := time.Now().UTC()
	a1 := &domain.MediaAsset{
		RootID:    1,
		RelPath:   "cyber_a.png",
		FileName:  "cyber_a.png",
		MediaType: domain.MediaTypeImage,
		Category:  "Cyberpunk",
		Rating:    9,
		FileMTime: now.Add(-10 * time.Minute),
	}
	a2 := &domain.MediaAsset{
		RootID:    1,
		RelPath:   "cyber_b.png",
		FileName:  "cyber_b.png",
		MediaType: domain.MediaTypeImage,
		Category:  "Cyberpunk",
		Rating:    5,
		FileMTime: now.Add(-20 * time.Minute),
	}
	id1, _ := repo.UpsertAsset(ctx, a1)
	id2, _ := repo.UpsertAsset(ctx, a2)

	server, err := web.NewServer(mediaDir, cacheDir, "noop", catalogSvc, scannerSvc, authSvc)
	if err != nil {
		t.Fatalf("web.NewServer error: %v", err)
	}

	return server, repo, id1, id2
}

func TestCatalogFilterPersistenceAndMediaNavigation(t *testing.T) {
	server, _, id1, id2 := setupTestServer(t)
	handler := server.Handler()

	// 1. Visit catalog with filter: category=Cyberpunk
	req := httptest.NewRequest("GET", "/?category=Cyberpunk&sort=rating", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	// Verify cookie dc_catalog_filter was set
	cookies := w.Result().Cookies()
	var filterCookie *http.Cookie
	for _, c := range cookies {
		if c.Name == "dc_catalog_filter" {
			filterCookie = c
			break
		}
	}
	if filterCookie == nil {
		t.Fatalf("expected cookie dc_catalog_filter to be set")
	}
	if !strings.Contains(filterCookie.Value, "category=Cyberpunk") {
		t.Errorf("cookie does not contain category=Cyberpunk: %s", filterCookie.Value)
	}

	// 2. Return to root "/" with that cookie -> should redirect to saved filter
	reqReturn := httptest.NewRequest("GET", "/", nil)
	reqReturn.AddCookie(filterCookie)
	wReturn := httptest.NewRecorder()
	handler.ServeHTTP(wReturn, reqReturn)

	if wReturn.Code != http.StatusFound {
		t.Fatalf("expected redirect 302, got %d", wReturn.Code)
	}
	loc := wReturn.Header().Get("Location")
	if !strings.Contains(loc, "category=Cyberpunk") {
		t.Errorf("expected redirect location to contain category=Cyberpunk, got %q", loc)
	}

	// 3. Reset filter via /?reset=1 -> cookie cleared and redirects to /
	reqReset := httptest.NewRequest("GET", "/?reset=1", nil)
	reqReset.AddCookie(filterCookie)
	wReset := httptest.NewRecorder()
	handler.ServeHTTP(wReset, reqReset)

	if wReset.Code != http.StatusFound {
		t.Fatalf("expected reset redirect 302, got %d", wReset.Code)
	}
	var resetCookie *http.Cookie
	for _, c := range wReset.Result().Cookies() {
		if c.Name == "dc_catalog_filter" {
			resetCookie = c
			break
		}
	}
	if resetCookie == nil || resetCookie.MaxAge > 0 {
		t.Errorf("expected cookie to be cleared (MaxAge < 0)")
	}

	// 4. Inspect media detail navigation: view id1 with category=Cyberpunk
	reqDetail := httptest.NewRequest("GET", "/?asset_id="+itoa(id1)+"&category=Cyberpunk", nil)
	wDetail := httptest.NewRecorder()
	handler.ServeHTTP(wDetail, reqDetail)

	if wDetail.Code != http.StatusOK {
		t.Fatalf("expected detail view status 200, got %d", wDetail.Code)
	}
	body := wDetail.Body.String()

	// Verify "Prev" and "Next" navigation buttons exist
	if !strings.Contains(body, "Prev") {
		t.Errorf("expected body to contain 'Prev' button")
	}
	if !strings.Contains(body, "Next") {
		t.Errorf("expected body to contain 'Next' button")
	}

	// Next media in category Cyberpunk is id2
	expectedNextLink := "asset_id=" + itoa(id2)
	if !strings.Contains(body, expectedNextLink) {
		t.Errorf("expected body to contain Next link pointing to %s", expectedNextLink)
	}

	// Back to Catalog should preserve category=Cyberpunk
	if !strings.Contains(body, "category=Cyberpunk") {
		t.Errorf("expected body to preserve category=Cyberpunk in links")
	}
}

func itoa(v int64) string {
	return strconv.FormatInt(v, 10)
}

func TestSynthesizeSceneInMemoryNoTempFiles(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	mediaDir := filepath.Join(tmpDir, "media")
	cacheDir := filepath.Join(tmpDir, "cache")
	dbPath := filepath.Join(tmpDir, "test_synth.db")

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("sqlite.Open error: %v", err)
	}

	repo := sqlite.NewRepository(db)
	authSvc, _ := usecase.NewAuthService(ctx, repo, "admin", "admin123")
	extractor := metadata.NewExtractor()
	scannerSvc := usecase.NewScannerService(mediaDir, cacheDir, repo, extractor, vision.NewNoopEngine())
	catalogSvc := usecase.NewCatalogService(mediaDir, repo, nil)

	server, err := web.NewServer(mediaDir, cacheDir, "noop", catalogSvc, scannerSvc, authSvc)
	if err != nil {
		t.Fatalf("web.NewServer error: %v", err)
	}
	catalogSvc.InspectAIBackendsFresh(ctx)
	handler := server.Handler()

	payload := `{
		"scene_json": "{\"elements\":[]}",
		"canvas_image_data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
		"target_profile": "flux",
		"user_hint": "test scene"
	}`

	req := httptest.NewRequest("POST", "/admin/compositions/synthesize", strings.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	// Verify no temporary files were written to uploadsDir
	uploadsDir := filepath.Join(cacheDir, "uploads")
	matches, _ := filepath.Glob(filepath.Join(uploadsDir, "scene_canvas_*"))
	if len(matches) > 0 {
		t.Errorf("expected 0 scene_canvas temporary files in %s, found %d: %v", uploadsDir, len(matches), matches)
	}
}

type testWebOllamaMock struct {
	lastCall string
}

func (m *testWebOllamaMock) CheckStatus(ctx context.Context, baseURL string) (string, bool) {
	return "1.0", true
}
func (m *testWebOllamaMock) ListModels(ctx context.Context, baseURL string) ([]domain.OllamaModelInfo, error) {
	return nil, nil
}
func (m *testWebOllamaMock) CheckOMLXStatus(ctx context.Context, baseURL, apiKey string) (bool, int) {
	return false, 0
}
func (m *testWebOllamaMock) ListOMLXModels(ctx context.Context, baseURL, apiKey string) ([]domain.OllamaModelInfo, error) {
	return nil, nil
}
func (m *testWebOllamaMock) ImageToPrompt(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt, imageSource, userHint, thinkLevel string) (string, error) {
	return "", nil
}
func (m *testWebOllamaMock) TextToPrompt(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt, inputConcept, thinkLevel string) (string, error) {
	return "", nil
}
func (m *testWebOllamaMock) AnalyzeImage(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt, imageSource, userPrompt, thinkLevel string) (string, error) {
	m.lastCall = "AnalyzeImage"
	return `{"aspectRatio":"16:9","canvasWidth":1024,"canvasHeight":576,"environmentType":"outdoor","elements":[{"type":"character","name":"Runner","poseName":"Running / Sprint"}]}`, nil
}
func (m *testWebOllamaMock) EmbedText(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, text string) ([]float32, error) {
	return nil, nil
}
func (m *testWebOllamaMock) EmbedImage(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model string, imageBytes []byte) ([]float32, error) {
	return nil, nil
}

func TestHandleImageToSceneEndpoint(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	mediaDir := filepath.Join(tmpDir, "media")
	cacheDir := filepath.Join(tmpDir, "cache")
	dbPath := filepath.Join(tmpDir, "test_i2s_web.db")

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("sqlite.Open error: %v", err)
	}

	repo := sqlite.NewRepository(db)
	mockOllama := &testWebOllamaMock{}
	catalogSvc := usecase.NewCatalogService(mediaDir, repo, mockOllama)
	_ = catalogSvc.SaveSettings(ctx, domain.AppSettings{
		RemixDefaultProvider: domain.ProviderOllama,
		VisionModel:          "llava:latest",
		OllamaDefaultModel:   "llava:latest",
	})

	authSvc, _ := usecase.NewAuthService(ctx, repo, "admin", "admin123")
	extractor := metadata.NewExtractor()
	scannerSvc := usecase.NewScannerService(mediaDir, cacheDir, repo, extractor, vision.NewNoopEngine())

	server, err := web.NewServer(mediaDir, cacheDir, "noop", catalogSvc, scannerSvc, authSvc)
	if err != nil {
		t.Fatalf("web.NewServer error: %v", err)
	}
	catalogSvc.InspectAIBackendsFresh(ctx)
	handler := server.Handler()

	// 1. Test empty body -> BadRequest
	reqEmpty := httptest.NewRequest("POST", "/admin/compositions/image-to-scene", strings.NewReader(`{}`))
	reqEmpty.Header.Set("Content-Type", "application/json")
	wEmpty := httptest.NewRecorder()
	handler.ServeHTTP(wEmpty, reqEmpty)
	if wEmpty.Code != http.StatusBadRequest {
		t.Errorf("expected 400 Bad Request for empty image, got %d", wEmpty.Code)
	}

	// 2. Test valid JSON payload with base64 image data and aspect ratio
	payload := `{
		"image_data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
		"user_hint": "cyberpunk style",
		"aspect_ratio": "3:4",
		"canvas_width": 768,
		"canvas_height": 1024
	}`
	req := httptest.NewRequest("POST", "/admin/compositions/image-to-scene", strings.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", w.Code, w.Body.String())
	}

	var resp struct {
		Status    string                 `json:"status"`
		SceneJSON string                 `json:"scene_json"`
		Scene     map[string]interface{} `json:"scene"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode JSON response: %v", err)
	}

	if resp.Status != "ok" {
		t.Errorf("expected status 'ok', got %q", resp.Status)
	}
	if resp.Scene["aspectRatio"] != "3:4" {
		t.Errorf("expected aspectRatio '3:4', got %v", resp.Scene["aspectRatio"])
	}
	if !strings.Contains(resp.SceneJSON, "Runner") {
		t.Errorf("expected scene_json to contain Runner, got %s", resp.SceneJSON)
	}
	if mockOllama.lastCall != "AnalyzeImage" {
		t.Errorf("expected AnalyzeImage called on Ollama mock, got %q", mockOllama.lastCall)
	}
}

func TestCollectionsTabRendering(t *testing.T) {
	server, repo, id1, id2 := setupTestServer(t)
	ctx := context.Background()
	handler := server.Handler()

	// Create a collection and link assets
	gid, err := repo.CreateGroup(ctx, &domain.Group{Name: "Retro Classics", Description: "Vintage arcade art"})
	if err != nil {
		t.Fatalf("CreateGroup error: %v", err)
	}
	_ = repo.AttachAssetToGroup(ctx, gid, id1)
	_ = repo.AttachAssetToGroup(ctx, gid, id2)

	// Visit /?tab=collections
	req := httptest.NewRequest("GET", "/?tab=collections", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	body := w.Body.String()
	if !strings.Contains(body, "Collections") {
		t.Errorf("expected page to contain 'Collections'")
	}
	if !strings.Contains(body, "Retro Classics") {
		t.Errorf("expected page to contain collection name 'Retro Classics'")
	}
	if !strings.Contains(body, "collections-grid") {
		t.Errorf("expected page to contain 'collections-grid' class")
	}
	if !strings.Contains(body, "collection-cover-duo") && !strings.Contains(body, "collection-cover-trio") {
		t.Errorf("expected page to render multi-cover preview container")
	}
	if strings.Contains(body, "Agnostic Groups") {
		t.Errorf("expected 'Agnostic Groups' text to be completely removed")
	}
}

func TestUpdateAndDeleteCollectionEndpoints(t *testing.T) {
	server, repo, id1, _ := setupTestServer(t)
	ctx := context.Background()
	handler := server.Handler()

	gid, err := repo.CreateGroup(ctx, &domain.Group{Name: "Initial Collection", Description: "Initial desc"})
	if err != nil {
		t.Fatalf("CreateGroup error: %v", err)
	}
	_ = repo.AttachAssetToGroup(ctx, gid, id1)

	// 1. Test POST /admin/groups/update
	updateForm := "id=" + strconv.FormatInt(gid, 10) + "&name=Updated+Collection+Title&description=Updated+desc"
	reqUpdate := httptest.NewRequest("POST", "/admin/groups/update", strings.NewReader(updateForm))
	reqUpdate.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	wUpdate := httptest.NewRecorder()
	handler.ServeHTTP(wUpdate, reqUpdate)

	if wUpdate.Code != http.StatusSeeOther {
		t.Errorf("expected 303 redirect, got %d", wUpdate.Code)
	}
	loc := wUpdate.Header().Get("Location")
	if !strings.Contains(loc, "tab=collections") {
		t.Errorf("expected redirect to tab=collections, got %s", loc)
	}

	// Verify update in SQLite
	groups, _ := repo.ListGroups(ctx)
	var found bool
	for _, g := range groups {
		if g.ID == gid && g.Name == "Updated Collection Title" && g.Description == "Updated desc" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected group to be updated in SQLite")
	}

	// 2. Test POST /admin/groups/delete
	deleteForm := "id=" + strconv.FormatInt(gid, 10)
	reqDelete := httptest.NewRequest("POST", "/admin/groups/delete", strings.NewReader(deleteForm))
	reqDelete.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	wDelete := httptest.NewRecorder()
	handler.ServeHTTP(wDelete, reqDelete)

	if wDelete.Code != http.StatusSeeOther {
		t.Errorf("expected 303 redirect, got %d", wDelete.Code)
	}

	// Verify group is deleted
	groupsAfter, _ := repo.ListGroups(ctx)
	for _, g := range groupsAfter {
		if g.ID == gid {
			t.Errorf("group %d still exists after delete", gid)
		}
	}
}

func TestActiveCollectionBannerAndFilterIndicator(t *testing.T) {
	server, repo, id1, _ := setupTestServer(t)
	ctx := context.Background()
	handler := server.Handler()

	gid, err := repo.CreateGroup(ctx, &domain.Group{Name: "Astral Archivists", Description: "Ancient astronomical archives"})
	if err != nil {
		t.Fatalf("CreateGroup error: %v", err)
	}
	_ = repo.AttachAssetToGroup(ctx, gid, id1)

	// Browse with ?group_id=...
	req := httptest.NewRequest("GET", "/?group_id="+strconv.FormatInt(gid, 10), nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	body := w.Body.String()
	if !strings.Contains(body, "active-collection-banner") {
		t.Errorf("expected active-collection-banner to be rendered")
	}
	if !strings.Contains(body, "active-filters-bar") {
		t.Errorf("expected active-filters-bar to be rendered")
	}
	if !strings.Contains(body, "Astral Archivists") {
		t.Errorf("expected collection name 'Astral Archivists' in body")
	}
	if !strings.Contains(body, "Exit Collection") {
		t.Errorf("expected 'Exit Collection' button in collection banner")
	}

	// Exit collection and chip remove button should link to /?reset=1 when group_id is the only filter
	if !strings.Contains(body, `href="/?reset=1" class="btn-ghost active-coll-exit"`) {
		t.Errorf("expected Exit Collection button to link to /?reset=1, got body:\n%s", body)
	}
	if !strings.Contains(body, `href="/?reset=1" class="chip-remove"`) {
		t.Errorf("expected collection chip remove button to link to /?reset=1")
	}

	// Verify that clicking Exit Collection clears the filter cookie and redirects cleanly
	var filterCookie *http.Cookie
	for _, c := range w.Result().Cookies() {
		if c.Name == "dc_catalog_filter" {
			filterCookie = c
			break
		}
	}
	if filterCookie == nil {
		t.Fatalf("expected cookie dc_catalog_filter to be set")
	}

	reqExit := httptest.NewRequest("GET", "/?reset=1", nil)
	reqExit.AddCookie(filterCookie)
	wExit := httptest.NewRecorder()
	handler.ServeHTTP(wExit, reqExit)

	if wExit.Code != http.StatusFound {
		t.Fatalf("expected status 302 from /?reset=1, got %d", wExit.Code)
	}
	var resetCookie *http.Cookie
	for _, c := range wExit.Result().Cookies() {
		if c.Name == "dc_catalog_filter" {
			resetCookie = c
			break
		}
	}
	if resetCookie == nil || resetCookie.MaxAge > 0 {
		t.Errorf("expected cookie to be cleared (MaxAge < 0)")
	}

	// Now verify multi-filter scenario: group_id AND category
	reqMulti := httptest.NewRequest("GET", "/?category=Cyberpunk&group_id="+strconv.FormatInt(gid, 10), nil)
	wMulti := httptest.NewRecorder()
	handler.ServeHTTP(wMulti, reqMulti)
	if wMulti.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", wMulti.Code)
	}
	bodyMulti := wMulti.Body.String()
	// When removing group_id while category is active, it should keep category
	if !strings.Contains(bodyMulti, `href="/?category=Cyberpunk" class="btn-ghost active-coll-exit"`) {
		t.Errorf("expected Exit Collection to link to /?category=Cyberpunk when category is also active")
	}
	if !strings.Contains(bodyMulti, `href="/?category=Cyberpunk" class="chip-remove"`) {
		t.Errorf("expected collection chip remove to link to /?category=Cyberpunk")
	}
}

type modelFilterTestOllamaMock struct {
	testWebOllamaMock
}

func (m *modelFilterTestOllamaMock) CheckStatus(ctx context.Context, baseURL string) (string, bool) {
	return "0.40.0", true
}

func (m *modelFilterTestOllamaMock) ListModels(ctx context.Context, baseURL string) ([]domain.OllamaModelInfo, error) {
	return []domain.OllamaModelInfo{
		{
			Provider:      domain.ProviderOllama,
			Name:          "nemotron-vl:latest",
			HasCompletion: true,
			HasVision:     true,
			Capabilities:  []string{"completion", "vision"},
		},
		{
			Provider:      domain.ProviderOllama,
			Name:          "mistral-chat:latest",
			HasCompletion: true,
			HasVision:     false,
			Capabilities:  []string{"completion"},
		},
		{
			Provider:      domain.ProviderOllama,
			Name:          "strands-decider:latest",
			HasCompletion: false,
			Capabilities:  []string{"decision"},
		},
		{
			Provider:      domain.ProviderOllama,
			Name:          "embeddinggemma:latest",
			HasCompletion: false,
			HasEmbedding:  true,
			Capabilities:  []string{"embedding"},
		},
		{
			Provider:      domain.ProviderOllama,
			Name:          "embeddinggemma-2:latest",
			HasCompletion: false,
			HasVision:     true,
			HasEmbedding:  true,
			Capabilities:  []string{"embedding", "vision", "audio"},
		},
	}, nil
}

func TestPromptStudioModelFiltering(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	mediaDir := filepath.Join(tmpDir, "media")
	cacheDir := filepath.Join(tmpDir, "cache")
	dbPath := filepath.Join(tmpDir, "test_filter_models.db")

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("sqlite.Open error: %v", err)
	}

	repo := sqlite.NewRepository(db)
	_ = repo.SaveSettings(ctx, domain.AppSettings{
		OllamaEnabled:         true,
		OllamaBaseURL:         "http://127.0.0.1:11434",
		OllamaDefaultModel:    "nemotron-vl:latest",
		RemixDefaultProvider:  domain.ProviderOllama,
		StudioDefaultProvider: domain.ProviderOllama,
	})

	authSvc, _ := usecase.NewAuthService(ctx, repo, "admin", "admin123")
	extractor := metadata.NewExtractor()
	scannerSvc := usecase.NewScannerService(mediaDir, cacheDir, repo, extractor, vision.NewNoopEngine())
	mockOllama := &modelFilterTestOllamaMock{}
	catalogSvc := usecase.NewCatalogService(mediaDir, repo, mockOllama)

	server, err := web.NewServer(mediaDir, cacheDir, "noop", catalogSvc, scannerSvc, authSvc)
	if err != nil {
		t.Fatalf("web.NewServer error: %v", err)
	}
	catalogSvc.InspectAIBackendsFresh(ctx)
	handler := server.Handler()

	// 1. In Vision mode: Only vision completion models must be present in remixModelSelect
	reqVision := httptest.NewRequest("GET", "/?tab=studio&mode=vision", nil)
	wVision := httptest.NewRecorder()
	handler.ServeHTTP(wVision, reqVision)
	if wVision.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", wVision.Code)
	}
	bodyVision := wVision.Body.String()

	// Locate remixModelSelect
	remixStart := strings.Index(bodyVision, `id="remixModelSelect"`)
	if remixStart == -1 {
		t.Fatalf("remixModelSelect not found in body")
	}
	remixEnd := strings.Index(bodyVision[remixStart:], `</select>`)
	remixBlockVision := bodyVision[remixStart : remixStart+remixEnd]

	if !strings.Contains(remixBlockVision, "nemotron-vl:latest") {
		t.Errorf("expected nemotron-vl:latest to be in vision options")
	}
	if strings.Contains(remixBlockVision, "mistral-chat:latest") {
		t.Errorf("mistral-chat:latest is text-only and must NOT appear in vision mode")
	}
	if strings.Contains(remixBlockVision, "strands-decider:latest") {
		t.Errorf("strands-decider:latest is decision-only and must NOT appear in prompt generation")
	}
	if strings.Contains(remixBlockVision, "embeddinggemma:latest") {
		t.Errorf("embeddinggemma:latest is embedding-only and must NOT appear in prompt generation")
	}
	if strings.Contains(remixBlockVision, "embeddinggemma-2:latest") {
		t.Errorf("embeddinggemma-2:latest is embedding-only and must NOT appear in prompt generation")
	}

	// 2. In Text Remix mode: Completion models (both text and vision) must appear, but NOT decision or embedding
	reqRemix := httptest.NewRequest("GET", "/?tab=studio&mode=remix", nil)
	wRemix := httptest.NewRecorder()
	handler.ServeHTTP(wRemix, reqRemix)
	if wRemix.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", wRemix.Code)
	}
	bodyRemix := wRemix.Body.String()

	remixStart = strings.Index(bodyRemix, `id="remixModelSelect"`)
	if remixStart == -1 {
		t.Fatalf("remixModelSelect not found in body")
	}
	remixEnd = strings.Index(bodyRemix[remixStart:], `</select>`)
	remixBlockRemix := bodyRemix[remixStart : remixStart+remixEnd]

	if !strings.Contains(remixBlockRemix, "nemotron-vl:latest") {
		t.Errorf("expected nemotron-vl:latest to be in remix options")
	}
	if !strings.Contains(remixBlockRemix, "mistral-chat:latest") {
		t.Errorf("expected mistral-chat:latest to be in remix options, got:\n%s", remixBlockRemix)
	}
	if strings.Contains(remixBlockRemix, "strands-decider:latest") {
		t.Errorf("strands-decider:latest is decision-only and must NOT appear in prompt generation")
	}
	if strings.Contains(remixBlockRemix, "embeddinggemma:latest") {
		t.Errorf("embeddinggemma:latest is embedding-only and must NOT appear in prompt generation")
	}
	if strings.Contains(remixBlockRemix, "embeddinggemma-2:latest") {
		t.Errorf("embeddinggemma-2:latest is embedding-only and must NOT appear in prompt generation")
	}
}
