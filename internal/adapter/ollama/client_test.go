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
	if got := NormalizeOllamaURL("http://127.0.0.1"); got != "http://127.0.0.1:11434" {
		t.Errorf("expected http://127.0.0.1:11434, got %s", got)
	}
	if got := NormalizeOMLXURL("http://127.0.0.1"); got != "http://127.0.0.1:8000" {
		t.Errorf("expected http://127.0.0.1:8000, got %s", got)
	}
	if got := NormalizeOMLXURL("http://127.0.0.1:8000/v1"); got != "http://127.0.0.1:8000" {
		t.Errorf("expected http://127.0.0.1:8000, got %s", got)
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

	models, err := client.ListOMLXModels(ctx, "http://127.0.0.1:8000", "omlx")
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

	out, err := client.TextToPrompt(ctx, domain.ProviderOMLX, "http://127.0.0.1:8000", "omlx", models[0].Name, "You are a Flux prompt specialist.", "cyberpunk street", "low")
	if err != nil {
		t.Fatalf("expected oMLX TextToPrompt to succeed, got %v", err)
	}
	if out == "" {
		t.Errorf("expected non-empty synthesized prompt")
	}
}

func TestCleanAIPrompt(t *testing.T) {
	tests := []struct {
		name     string
		raw      string
		expected string
	}{
		{
			name:     "plain clean prompt",
			raw:      "A cinematic portrait of a cybernetic warrior.",
			expected: "A cinematic portrait of a cybernetic warrior.",
		},
		{
			name: "markdown code block with language",
			raw: "Here is your prompt for Flux:\n```markdown\nA cinematic portrait of a cybernetic warrior at dusk, 85mm lens.\n```\nHope this helps!",
			expected: "A cinematic portrait of a cybernetic warrior at dusk, 85mm lens.",
		},
		{
			name: "code block without language tag",
			raw: "```\nA lone archivist reading ancient manuscripts in an ethereal library.\n```",
			expected: "A lone archivist reading ancient manuscripts in an ethereal library.",
		},
		{
			name: "unclosed code block due to truncation",
			raw: "```text\nA lone archivist reading ancient manuscripts in an ethereal library",
			expected: "A lone archivist reading ancient manuscripts in an ethereal library",
		},
		{
			name: "escaped literal line breaks and carriage returns",
			raw: `A cinematic shot.\n\nWearing tactical coat.\r\nVolumetric lighting.`,
			expected: "A cinematic shot.\n\nWearing tactical coat.\nVolumetric lighting.",
		},
		{
			name: "escaped malformed /r/n sequences",
			raw: "Line one/r/nLine two/r/nLine three",
			expected: "Line one\nLine two\nLine three",
		},
		{
			name: "thinking tags removal",
			raw: "<think>Let me formulate the prompt.\nFirst consider lighting.</think>A futuristic street drenched in rain.",
			expected: "A futuristic street drenched in rain.",
		},
		{
			name: "unclosed thinking block",
			raw: "<think>Thinking about options...",
			expected: "",
		},
		{
			name: "conversational prefix with Prompt label",
			raw: "Sure! Here is the prompt:\n**Prompt:** A neon-lit cybernetic city under torrential rain.",
			expected: "A neon-lit cybernetic city under torrential rain.",
		},
		{
			name: "positive prompt label prefix",
			raw: "Positive Prompt: masterpiece, best quality, ultra-detailed astronaut floating in nebula.",
			expected: "masterpiece, best quality, ultra-detailed astronaut floating in nebula.",
		},
		{
			name: "conversational postambles removed",
			raw: "A majestic dragon soaring above snow-capped mountains.\n\n**Notes:** Optimized for SDXL.\nLet me know if you need any adjustments!",
			expected: "A majestic dragon soaring above snow-capped mountains.",
		},
		{
			name: "enclosing backticks or quotes",
			raw: "`A futuristic vehicle racing through neon highway.`",
			expected: "A futuristic vehicle racing through neon highway.",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CleanAIPrompt(tt.raw)
			if got != tt.expected {
				t.Errorf("\nexpected: %q\ngot:      %q", tt.expected, got)
			}
		})
	}
}
