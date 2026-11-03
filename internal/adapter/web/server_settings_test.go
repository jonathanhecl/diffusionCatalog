package web_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

func TestSimplifiedLLMSettingsWorkflow(t *testing.T) {
	server, repo, _, _ := setupTestServer(t)
	handler := server.Handler()
	ctx := context.Background()

	// 1. Initial settings inspection
	initial, err := repo.GetSettings(ctx)
	if err != nil {
		t.Fatalf("GetSettings error: %v", err)
	}
	if !initial.OllamaEnabled || !initial.OpenAIEnabled {
		t.Errorf("expected both providers enabled by default")
	}

	// 2. Submit new simplified settings
	form := url.Values{
		"ollama_enabled":           {"1"},
		"ollama_base_url":          {"http://192.168.1.100:11434"},
		"ollama_default_model":     {"qwen2.5-vl:7b"},
		"ollama_default_think":     {"low"},
		"openai_enabled":           {"1"},
		"openai_base_url":          {"http://192.168.1.100:8000"},
		"openai_api_key":           {"test-secret-key"},
		"openai_default_model":     {"SuperQwen3.8-27b-vision"},
		"openai_default_think":     {"medium"},
		"remix_default_provider":   {"ollama"},
		"studio_default_provider":  {"openai"},
		"default_target_generator": {"flux"},
	}

	reqSave := httptest.NewRequest("POST", "/admin/settings/save", strings.NewReader(form.Encode()))
	reqSave.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	wSave := httptest.NewRecorder()
	handler.ServeHTTP(wSave, reqSave)

	if wSave.Code != http.StatusSeeOther {
		t.Fatalf("expected redirect 303 after save, got %d", wSave.Code)
	}

	// 3. Verify settings persisted in repository
	saved, err := repo.GetSettings(ctx)
	if err != nil {
		t.Fatalf("GetSettings after save error: %v", err)
	}

	if saved.OllamaBaseURL != "http://192.168.1.100:11434" {
		t.Errorf("expected ollama_base_url 'http://192.168.1.100:11434', got %q", saved.OllamaBaseURL)
	}
	if saved.OllamaDefaultModel != "qwen2.5-vl:7b" {
		t.Errorf("expected ollama_default_model 'qwen2.5-vl:7b', got %q", saved.OllamaDefaultModel)
	}
	if saved.OllamaDefaultThink != "low" {
		t.Errorf("expected ollama_default_think 'low', got %q", saved.OllamaDefaultThink)
	}
	if saved.OpenAIBaseURL != "http://192.168.1.100:8000" {
		t.Errorf("expected openai_base_url 'http://192.168.1.100:8000', got %q", saved.OpenAIBaseURL)
	}
	if saved.OpenAIAPIKey != "test-secret-key" {
		t.Errorf("expected openai_api_key 'test-secret-key', got %q", saved.OpenAIAPIKey)
	}
	if saved.OpenAIDefaultModel != "SuperQwen3.8-27b-vision" {
		t.Errorf("expected openai_default_model 'SuperQwen3.8-27b-vision', got %q", saved.OpenAIDefaultModel)
	}
	if saved.OpenAIDefaultThink != "medium" {
		t.Errorf("expected openai_default_think 'medium', got %q", saved.OpenAIDefaultThink)
	}
	if saved.RemixDefaultProvider != domain.ProviderOllama {
		t.Errorf("expected remix_default_provider 'ollama', got %q", saved.RemixDefaultProvider)
	}
	if saved.StudioDefaultProvider != domain.ProviderOpenAI {
		t.Errorf("expected studio_default_provider 'openai', got %q", saved.StudioDefaultProvider)
	}

	// 4. Verify helper methods
	remixProv, remixMod, remixThk := saved.GetRemixDefaults()
	if remixProv != domain.ProviderOllama || remixMod != "qwen2.5-vl:7b" || remixThk != "low" {
		t.Errorf("unexpected RemixDefaults: %v, %v, %v", remixProv, remixMod, remixThk)
	}

	studioProv, studioMod, studioThk := saved.GetStudioDefaults()
	if studioProv != domain.ProviderOpenAI || studioMod != "SuperQwen3.8-27b-vision" || studioThk != "medium" {
		t.Errorf("unexpected StudioDefaults: %v, %v, %v", studioProv, studioMod, studioThk)
	}

	// 5. Verify Settings view rendering
	reqView := httptest.NewRequest("GET", "/?tab=settings", nil)
	wView := httptest.NewRecorder()
	handler.ServeHTTP(wView, reqView)

	if wView.Code != http.StatusOK {
		t.Fatalf("expected status 200 on settings view, got %d", wView.Code)
	}
	body := wView.Body.String()
	if !strings.Contains(body, "1. Ollama Provider") {
		t.Errorf("expected '1. Ollama Provider' in settings view")
	}
	if !strings.Contains(body, "2. OpenAI Compatible") {
		t.Errorf("expected '2. OpenAI Compatible' in settings view")
	}
	if !strings.Contains(body, "3. Remix Default Provider") {
		t.Errorf("expected '3. Remix Default Provider' in settings view")
	}
	if !strings.Contains(body, "4. Studio 2D / General Default Provider") {
		t.Errorf("expected '4. Studio 2D / General Default Provider' in settings view")
	}
	if !strings.Contains(body, "Vision Required 👁️") {
		t.Errorf("expected 'Vision Required 👁️' note in model selectors")
	}
}

func TestSettingsVerticalNavbarAndSystemPrompts(t *testing.T) {
	server, repo, _, _ := setupTestServer(t)
	handler := server.Handler()
	ctx := context.Background()

	// 1. /?tab=generators redirects directly to /?tab=settings&section=generators
	reqGen := httptest.NewRequest("GET", "/?tab=generators&gen=flux", nil)
	wGen := httptest.NewRecorder()
	handler.ServeHTTP(wGen, reqGen)
	if wGen.Code != http.StatusFound {
		t.Fatalf("expected redirect 302 from /?tab=generators, got %d", wGen.Code)
	}
	loc := wGen.Header().Get("Location")
	if !strings.Contains(loc, "tab=settings") || !strings.Contains(loc, "section=generators") {
		t.Errorf("expected location to redirect to tab=settings&section=generators, got %q", loc)
	}

	// 2. /?tab=settings renders vertical navigation sidebar
	reqSettings := httptest.NewRequest("GET", "/?tab=settings", nil)
	wSettings := httptest.NewRecorder()
	handler.ServeHTTP(wSettings, reqSettings)
	if wSettings.Code != http.StatusOK {
		t.Fatalf("expected 200 OK for /?tab=settings, got %d", wSettings.Code)
	}
	settingsBody := wSettings.Body.String()
	if !strings.Contains(settingsBody, "AI &amp; ROUTING") {
		t.Errorf("expected 'AI & ROUTING' in settings vertical navbar")
	}
	if !strings.Contains(settingsBody, "SYSTEM PROMPTS") {
		t.Errorf("expected 'SYSTEM PROMPTS' in settings vertical navbar")
	}
	if !strings.Contains(settingsBody, "STORAGE &amp; DATA") {
		t.Errorf("expected 'STORAGE & DATA' in settings vertical navbar")
	}
	if !strings.Contains(settingsBody, "+ Add New System Prompt") {
		t.Errorf("expected '+ Add New System Prompt' in settings vertical navbar")
	}

	// 3. Upsert a new system prompt profile
	formProfile := url.Values{
		"slug":            {"wan21"},
		"name":            {"Wan 2.1 Video"},
		"target_modality": {"video"},
		"description":     {"Wan video diffusion model family"},
		"system_prompt":   {"You are an expert prompt engineer for Wan 2.1 video generation."},
	}
	reqUpsert := httptest.NewRequest("POST", "/admin/profiles/upsert", strings.NewReader(formProfile.Encode()))
	reqUpsert.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	wUpsert := httptest.NewRecorder()
	handler.ServeHTTP(wUpsert, reqUpsert)
	if wUpsert.Code != http.StatusSeeOther {
		t.Fatalf("expected 303 redirect after profile upsert, got %d", wUpsert.Code)
	}
	upsertLoc := wUpsert.Header().Get("Location")
	if !strings.Contains(upsertLoc, "tab=settings") || !strings.Contains(upsertLoc, "section=generators") || !strings.Contains(upsertLoc, "gen=wan21") {
		t.Errorf("expected upsert redirect to settings generators section, got %q", upsertLoc)
	}

	// 4. Verify persisted in DB
	prof, err := repo.GetPromptProfileBySlug(ctx, "wan21")
	if err != nil || prof == nil {
		t.Fatalf("expected profile wan21 to exist in repo, err: %v", err)
	}
	if prof.SystemPrompt != "You are an expert prompt engineer for Wan 2.1 video generation." {
		t.Errorf("unexpected system prompt: %q", prof.SystemPrompt)
	}

	// 5. GET /?tab=settings&section=generators&gen=wan21 renders system prompt and file loader button
	reqGenView := httptest.NewRequest("GET", "/?tab=settings&section=generators&gen=wan21", nil)
	wGenView := httptest.NewRecorder()
	handler.ServeHTTP(wGenView, reqGenView)
	if wGenView.Code != http.StatusOK {
		t.Fatalf("expected 200 OK for generators view in settings, got %d", wGenView.Code)
	}
	genViewBody := wGenView.Body.String()
	if !strings.Contains(genViewBody, "Wan 2.1 Video") {
		t.Errorf("expected 'Wan 2.1 Video' in view body")
	}
	if !strings.Contains(genViewBody, "Load from .txt / .md") {
		t.Errorf("expected 'Load from .txt / .md' button in system prompt editor")
	}
	if !strings.Contains(genViewBody, "You are an expert prompt engineer for Wan 2.1 video generation.") {
		t.Errorf("expected system prompt content in textarea")
	}
}

func TestModelNameQuoteSanitization(t *testing.T) {
	server, repo, _, _ := setupTestServer(t)
	handler := server.Handler()
	ctx := context.Background()

	// 1. Save settings with quotes around default model names
	form := url.Values{
		"ollama_enabled":           {"1"},
		"ollama_base_url":          {"http://mac-mini.local:11434"},
		"ollama_default_model":     {`"huggingface.co/grumd0/Nemotron-3.5-Bolt-E4B-Heretic-GGUF:Q4_K_M"`},
		"ollama_default_think":     {"off"},
		"openai_enabled":           {"1"},
		"openai_base_url":          {"http://mac-mini.local:8000"},
		"openai_default_model":     {`"bonsai-2-27b"`},
		"openai_default_think":     {"low"},
		"remix_default_provider":   {"ollama"},
		"studio_default_provider":  {"openai"},
		"default_target_generator": {"flux"},
	}

	reqSave := httptest.NewRequest("POST", "/admin/settings/save", strings.NewReader(form.Encode()))
	reqSave.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	wSave := httptest.NewRecorder()
	handler.ServeHTTP(wSave, reqSave)

	if wSave.Code != http.StatusSeeOther {
		t.Fatalf("expected redirect 303 after save, got %d", wSave.Code)
	}

	saved, err := repo.GetSettings(ctx)
	if err != nil {
		t.Fatalf("GetSettings error: %v", err)
	}
	if saved.OllamaDefaultModel != "huggingface.co/grumd0/Nemotron-3.5-Bolt-E4B-Heretic-GGUF:Q4_K_M" {
		t.Errorf("expected sanitized model without quotes, got: %q", saved.OllamaDefaultModel)
	}
	if saved.OpenAIDefaultModel != "bonsai-2-27b" {
		t.Errorf("expected sanitized model without quotes, got: %q", saved.OpenAIDefaultModel)
	}

	// 2. Verify rendered JS in catalog.html doesn't contain double escaped quotes
	reqView := httptest.NewRequest("GET", "/?tab=director", nil)
	wView := httptest.NewRecorder()
	handler.ServeHTTP(wView, reqView)

	if wView.Code != http.StatusOK {
		t.Fatalf("expected status 200 on director view, got %d", wView.Code)
	}
	body := wView.Body.String()
	// Must NOT contain escaped quotes in string literals like: defaultModel: "\"bonsai-2-27b\""
	if strings.Contains(body, `defaultModel: "\"`) || strings.Contains(body, `name: "\"`) {
		t.Errorf("catalog.html contains double-quoted/escaped model names in JS config")
	}
}

func TestStudioAttachedImageClientSessionStorage(t *testing.T) {
	server, _, _, _ := setupTestServer(t)
	handler := server.Handler()

	reqView := httptest.NewRequest("GET", "/?tab=studio", nil)
	wView := httptest.NewRecorder()
	handler.ServeHTTP(wView, reqView)

	if wView.Code != http.StatusOK {
		t.Fatalf("expected 200 OK on studio view, got %d", wView.Code)
	}
	body := wView.Body.String()
	if !strings.Contains(body, "sessionStorage.setItem('dc_studio_attached_image'") {
		t.Errorf("expected sessionStorage caching for attached image in studio")
	}
	if !strings.Contains(body, "restoreStudioAttachedImage") {
		t.Errorf("expected restoreStudioAttachedImage function in studio")
	}
	if !strings.Contains(body, "removeStudioAttachedImage") {
		t.Errorf("expected removeStudioAttachedImage function in studio")
	}
}

func TestDirectorNewPosesPresetsRendering(t *testing.T) {
	server, _, _, _ := setupTestServer(t)
	handler := server.Handler()

	reqView := httptest.NewRequest("GET", "/?tab=director", nil)
	wView := httptest.NewRecorder()
	handler.ServeHTTP(wView, reqView)

	if wView.Code != http.StatusOK {
		t.Fatalf("expected 200 OK on director view, got %d", wView.Code)
	}
	body := wView.Body.String()

	expectedSubstrings := []string{
		"Leaning Forward (Balcony)",
		"Pushing Heavy Load",
		"Leaning Forward (Balcony Side)",
		"Pushing Heavy Load (Side)",
		"Asomado Balcón",
		"Empujar Carga",
		`data-subtype="marker"`,
		"Body Feature",
		"propRoleSelect",
		"propFeatureControls",
		"propBodyPartInput",
		"Half Forward Fold (90°)",
		"Downward-Facing Dog (Yoga)",
		"Warrior II (Yoga Pose)",
		"Cobra Pose (Upward Backbend)",
		"Model Cross-Legged (Runway)",
		"Sensual Cat Crawl (To Camera)",
		"Model Lounging Recline (Glamour)",
		"Torso 90",
		"Perro Abajo",
		"Guerrero II",
		"Cobra Yoga",
		"Modelo Cruzada",
		"Gateo Cámara",
		"Recostada Glam",
	}

	for _, substr := range expectedSubstrings {
		if !strings.Contains(body, substr) {
			t.Errorf("expected director view to contain %q, but not found", substr)
		}
	}

	// Ensure no duplicate closing block or syntax error around updateSpatialManifest
	if strings.Contains(body, "viewer.textContent = manifest.trim();\n  }\n    if (lights.length)") {
		t.Errorf("found duplicate corrupted updateSpatialManifest closure")
	}
}

func TestKnowledgeBaseSettingsAndTokenBudgetLifecycle(t *testing.T) {
	server, repo, _, _ := setupTestServer(t)
	handler := server.Handler()
	ctx := context.Background()

	// 1. GET /?tab=settings renders KNOWLEDGE BASE in vertical sidebar
	reqSettings := httptest.NewRequest("GET", "/?tab=settings", nil)
	wSettings := httptest.NewRecorder()
	handler.ServeHTTP(wSettings, reqSettings)
	if wSettings.Code != http.StatusOK {
		t.Fatalf("expected 200 OK for settings, got %d", wSettings.Code)
	}
	settingsBody := wSettings.Body.String()
	if !strings.Contains(settingsBody, "KNOWLEDGE BASE") {
		t.Errorf("expected 'KNOWLEDGE BASE' in settings vertical navbar")
	}
	if !strings.Contains(settingsBody, "+ Add New Knowledge") {
		t.Errorf("expected '+ Add New Knowledge' in settings vertical navbar")
	}

	// 2. Upsert a new knowledge snippet
	formKnowledge := url.Values{
		"slug":        {"anamorphic-optics"},
		"title":       {"Anamorphic Optics & Flares"},
		"description": {"Oval bokeh and streak flare rules"},
		"content":     {"Ensure subtle horizontal cyan streaks and oval bokeh across all specular highlights."},
	}
	reqUpsert := httptest.NewRequest("POST", "/admin/knowledge/upsert", strings.NewReader(formKnowledge.Encode()))
	reqUpsert.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	wUpsert := httptest.NewRecorder()
	handler.ServeHTTP(wUpsert, reqUpsert)
	if wUpsert.Code != http.StatusSeeOther {
		t.Fatalf("expected 303 redirect after knowledge upsert, got %d", wUpsert.Code)
	}
	upsertLoc := wUpsert.Header().Get("Location")
	if !strings.Contains(upsertLoc, "section=knowledge") || !strings.Contains(upsertLoc, "slug=anamorphic-optics") {
		t.Errorf("expected upsert redirect to knowledge section, got %q", upsertLoc)
	}

	// 3. Verify in DB
	snippets, err := repo.ListKnowledgeSnippets(ctx)
	if err != nil {
		t.Fatalf("failed to list knowledge snippets: %v", err)
	}
	var found *domain.KnowledgeSnippet
	for i := range snippets {
		if snippets[i].Slug == "anamorphic-optics" {
			found = &snippets[i]
			break
		}
	}
	if found == nil {
		t.Fatalf("expected anamorphic-optics to be saved in DB")
	}
	if found.Title != "Anamorphic Optics & Flares" {
		t.Errorf("expected title 'Anamorphic Optics & Flares', got %q", found.Title)
	}

	// 4. GET /?tab=settings&section=knowledge&slug=anamorphic-optics renders knowledge content & token count badge
	reqView := httptest.NewRequest("GET", "/?tab=settings&section=knowledge&slug=anamorphic-optics", nil)
	wView := httptest.NewRecorder()
	handler.ServeHTTP(wView, reqView)
	if wView.Code != http.StatusOK {
		t.Fatalf("expected 200 OK for knowledge view, got %d", wView.Code)
	}
	viewBody := wView.Body.String()
	if !strings.Contains(viewBody, "Anamorphic Optics &amp; Flares") && !strings.Contains(viewBody, "Anamorphic Optics & Flares") {
		t.Errorf("expected knowledge title in view body")
	}
	if !strings.Contains(viewBody, "Ensure subtle horizontal cyan streaks and oval bokeh") {
		t.Errorf("expected knowledge content in textarea")
	}
	if !strings.Contains(viewBody, "Load from .txt / .md") {
		t.Errorf("expected 'Load from .txt / .md' file picker button")
	}

	// 5. GET /?tab=studio renders knowledge chips and Context Weight budget bar
	reqStudio := httptest.NewRequest("GET", "/?tab=studio", nil)
	wStudio := httptest.NewRecorder()
	handler.ServeHTTP(wStudio, reqStudio)
	if wStudio.Code != http.StatusOK {
		t.Fatalf("expected 200 OK for studio, got %d", wStudio.Code)
	}
	studioBody := wStudio.Body.String()
	if !strings.Contains(studioBody, "Contextual Knowledge &amp; Directives") && !strings.Contains(studioBody, "Incorporate Contextual Knowledge") {
		t.Errorf("expected contextual knowledge section in studio")
	}
	if !strings.Contains(studioBody, "Anamorphic Optics &amp; Flares") && !strings.Contains(studioBody, "Anamorphic Optics & Flares") {
		t.Errorf("expected knowledge chip in studio")
	}
	if !strings.Contains(studioBody, "studio-context-budget-bar") {
		t.Errorf("expected studio context budget bar")
	}

	// 6. Delete knowledge snippet
	formDel := url.Values{"id": {strconv.FormatInt(found.ID, 10)}}
	reqDel := httptest.NewRequest("POST", "/admin/knowledge/delete", strings.NewReader(formDel.Encode()))
	reqDel.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	wDel := httptest.NewRecorder()
	handler.ServeHTTP(wDel, reqDel)
	if wDel.Code != http.StatusSeeOther {
		t.Fatalf("expected 303 redirect after delete, got %d", wDel.Code)
	}
	delSnippet, _ := repo.GetKnowledgeSnippetByID(ctx, found.ID)
	if delSnippet != nil {
		t.Errorf("expected deleted snippet to be nil")
	}
}
