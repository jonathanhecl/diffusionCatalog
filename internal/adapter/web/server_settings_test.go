package web_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
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
