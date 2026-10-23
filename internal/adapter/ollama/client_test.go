package ollama

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

type roundTripFunc func(req *http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestNormalizeURLs(t *testing.T) {
	if got := NormalizeOllamaURL("http://mac-mini.local"); got != "http://mac-mini.local:11434" {
		t.Errorf("expected http://mac-mini.local:11434, got %s", got)
	}
	if got := NormalizeOMLXURL("http://mac-mini.local"); got != "http://mac-mini.local:8000" {
		t.Errorf("expected http://mac-mini.local:8000, got %s", got)
	}
	if got := NormalizeOMLXURL("http://mac-mini.local:8000/v1"); got != "http://mac-mini.local:8000" {
		t.Errorf("expected http://mac-mini.local:8000, got %s", got)
	}
}

func TestOMLXListModelsAndTextToPrompt(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer omlx" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"object": "list",
			"data": [
				{"id": "SuperQwen3.8-27b-abliterated-MLX-4bit", "max_model_len": 262144},
				{"id": "Qwen2.5-VL-7B-Instruct-MLX-4bit", "max_model_len": 32768}
			]
		}`))
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"choices": [{"message": {"role": "assistant", "content": "Cinematic wide shot, neon rain reflections, 35mm anamorphic lens."}}]
		}`))
	})

	client := &Client{
		httpClient: &http.Client{
			Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				rec := httptest.NewRecorder()
				mux.ServeHTTP(rec, req)
				return rec.Result(), nil
			}),
		},
	}
	ctx := context.Background()

	models, err := client.ListOMLXModels(ctx, "http://mac-mini.local:8000", "omlx")
	if err != nil {
		t.Fatalf("expected ListOMLXModels to succeed, got %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("expected 2 oMLX models, got %d", len(models))
	}
	if models[0].ContextLength != 262144 {
		t.Errorf("expected max_model_len 262144, got %d", models[0].ContextLength)
	}
	if !models[0].HasVision || !models[0].HasThinking || !models[0].HasTools {
		t.Errorf("expected SuperQwen3.8 to report vision, thinking, and tools capabilities, got %+v", models[0].Capabilities)
	}
	if !models[1].HasVision {
		t.Errorf("expected Qwen2.5-VL to be flagged HasVision=true")
	}

	out, err := client.TextToPrompt(ctx, domain.ProviderOMLX, "http://mac-mini.local:8000", "omlx", models[0].Name, "You are a Flux prompt specialist.", "cyberpunk street", "low")
	if err != nil {
		t.Fatalf("expected oMLX TextToPrompt to succeed, got %v", err)
	}
	if out == "" {
		t.Errorf("expected non-empty synthesized prompt")
	}
}
