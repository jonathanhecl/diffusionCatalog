package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
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

	if models[0].Provider != domain.ProviderOpenAI {
		t.Errorf("expected ProviderOpenAI, got %q", models[0].Provider)
	}

	out, err := client.TextToPrompt(ctx, domain.ProviderOpenAI, "http://127.0.0.1:8000", "omlx", models[0].Name, "You are a Flux prompt specialist.", "cyberpunk street", "low")
	if err != nil {
		t.Fatalf("expected OpenAI TextToPrompt to succeed, got %v", err)
	}
	if out == "" {
		t.Errorf("expected non-empty synthesized prompt")
	}

	outLegacy, err := client.TextToPrompt(ctx, domain.ProviderOMLX, "http://127.0.0.1:8000", "omlx", models[0].Name, "You are a Flux prompt specialist.", "cyberpunk street", "low")
	if err != nil {
		t.Fatalf("expected legacy oMLX TextToPrompt to succeed, got %v", err)
	}
	if outLegacy == "" {
		t.Errorf("expected non-empty synthesized prompt from legacy provider")
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
		{
			name: "json object output extraction",
			raw: `{"prompt": "A beautiful cybernetic geisha in Kyoto alley, neon reflections.", "negative_prompt": "blurry, low quality"}`,
			expected: "A beautiful cybernetic geisha in Kyoto alley, neon reflections.",
		},
		{
			name: "markdown fenced json object extraction",
			raw: "```json\n{\n  \"positive_prompt\": \"Masterpiece portrait in oil paint style, dramatic chiaroscuro.\"\n}\n```",
			expected: "Masterpiece portrait in oil paint style, dramatic chiaroscuro.",
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

func TestOllamaAndOMLXErrorParsing(t *testing.T) {
	ctx := context.Background()

	// 1. Ollama 404 JSON error
	ollamaMux := http.NewServeMux()
	ollamaMux.HandleFunc("/api/chat", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":"model 'SuperQwen3.8-27b-abliterated-MLX-4bit' not found, try pulling it first"}`))
	})
	ollamaClient := &Client{
		httpClient: &http.Client{
			Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				rec := httptest.NewRecorder()
				ollamaMux.ServeHTTP(rec, req)
				return rec.Result(), nil
			}),
		},
	}

	_, err := ollamaClient.TextToPrompt(ctx, domain.ProviderOllama, "http://127.0.0.1:11434", "", "SuperQwen3.8-27b-abliterated-MLX-4bit", "sys", "concept", "")
	if err == nil || !strings.Contains(err.Error(), "model 'SuperQwen3.8-27b-abliterated-MLX-4bit' not found") {
		t.Errorf("expected clean unmarshaled error message, got %v", err)
	}

	// 2. oMLX 500 JSON error
	omlxMux := http.NewServeMux()
	omlxMux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":{"message":"Engine out of memory while loading weights"}}`))
	})
	omlxClient := &Client{
		httpClient: &http.Client{
			Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				rec := httptest.NewRecorder()
				omlxMux.ServeHTTP(rec, req)
				return rec.Result(), nil
			}),
		},
	}

	_, err = omlxClient.TextToPrompt(ctx, domain.ProviderOpenAI, "http://127.0.0.1:8000", "omlx", "some-model", "sys", "concept", "")
	if err == nil || !strings.Contains(err.Error(), "Engine out of memory") {
		t.Errorf("expected clean unmarshaled oMLX error message, got %v", err)
	}
}

func TestImageToPromptInMemoryDataURL(t *testing.T) {
	ctx := context.Background()
	var capturedOllamaImage string
	var capturedOMLXImageURL string

	ollamaMux := http.NewServeMux()
	ollamaMux.HandleFunc("/api/chat", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Messages []struct {
				Role    string   `json:"role"`
				Content string   `json:"content"`
				Images  []string `json:"images"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		for _, m := range req.Messages {
			if len(m.Images) > 0 {
				capturedOllamaImage = m.Images[0]
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"message":{"role":"assistant","content":"Cyberpunk alleyway with neon signs"}}`))
	})

	omlxMux := http.NewServeMux()
	omlxMux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Messages []struct {
				Role    string `json:"role"`
				Content []struct {
					Type     string `json:"type"`
					Text     string `json:"text"`
					ImageURL *struct {
						URL string `json:"url"`
					} `json:"image_url"`
				} `json:"content"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		for _, m := range req.Messages {
			for _, c := range m.Content {
				if c.ImageURL != nil {
					capturedOMLXImageURL = c.ImageURL.URL
				}
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"Moody neon rain, cinematic shot"}}]}`))
	})

	client := &Client{
		httpClient: &http.Client{
			Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				rec := httptest.NewRecorder()
				if strings.Contains(req.URL.Path, "completions") {
					omlxMux.ServeHTTP(rec, req)
				} else {
					ollamaMux.ServeHTTP(rec, req)
				}
				return rec.Result(), nil
			}),
		},
	}

	rawB64 := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
	dataURL := "data:image/png;base64," + rawB64

	// Test with Ollama using in-memory data URL
	prompt, err := client.ImageToPrompt(ctx, domain.ProviderOllama, "http://127.0.0.1:11434", "", "llava", "system prompt", dataURL, "user hint", "")
	if err != nil {
		t.Fatalf("expected ImageToPrompt with data URL to succeed, got %v", err)
	}
	if prompt == "" {
		t.Errorf("expected non-empty prompt")
	}
	if capturedOllamaImage != rawB64 {
		t.Errorf("expected Ollama image to receive extracted base64 %q, got %q", rawB64, capturedOllamaImage)
	}

	// Test with oMLX using in-memory data URL
	omlxPrompt, err := client.ImageToPrompt(ctx, domain.ProviderOpenAI, "http://127.0.0.1:8000", "key", "vision-model", "system prompt", dataURL, "user hint", "")
	if err != nil {
		t.Fatalf("expected oMLX ImageToPrompt with data URL to succeed, got %v", err)
	}
	if omlxPrompt == "" {
		t.Errorf("expected non-empty omlx prompt")
	}
	if capturedOMLXImageURL != dataURL {
		t.Errorf("expected oMLX image to receive full data URL %q, got %q", dataURL, capturedOMLXImageURL)
	}
}

func TestEmbedTextAndImage(t *testing.T) {
	ctx := context.Background()

	var receivedTextPayload ollamaEmbedRequest
	var receivedImagePayload ollamaEmbedRequest

	mux := http.NewServeMux()
	mux.HandleFunc("/api/embed", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		var body map[string]interface{}
		_ = json.NewDecoder(r.Body).Decode(&body)

		if inputs, ok := body["input"].([]interface{}); ok && len(inputs) > 0 {
			if _, isMap := inputs[0].(map[string]interface{}); isMap {
				receivedImagePayload.Model = body["model"].(string)
				receivedImagePayload.Input = body["input"]
				_, _ = w.Write([]byte(`{"model": "embeddinggemma-2", "embeddings": [[0.1, 0.2, 0.3]]}`))
				return
			}
			receivedTextPayload.Model = body["model"].(string)
			receivedTextPayload.Input = body["input"]
			_, _ = w.Write([]byte(`{"model": "embeddinggemma-2", "embeddings": [[0.4, 0.5, 0.6]]}`))
			return
		}
		http.NotFound(w, r)
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

	// 1. Text embedding
	textVec, err := client.EmbedText(ctx, domain.ProviderOllama, "http://127.0.0.1:11434", "", "embeddinggemma-2", "gato durmiendo")
	if err != nil {
		t.Fatalf("EmbedText failed: %v", err)
	}
	if len(textVec) != 3 || textVec[0] != 0.4 {
		t.Errorf("unexpected text vector: %v", textVec)
	}
	if receivedTextPayload.Model != "embeddinggemma-2" {
		t.Errorf("expected model embeddinggemma-2, got %s", receivedTextPayload.Model)
	}

	// 2. Image embedding
	fakeImg := []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}
	imgVec, err := client.EmbedImage(ctx, domain.ProviderOllama, "http://127.0.0.1:11434", "", "embeddinggemma-2", fakeImg)
	if err != nil {
		t.Fatalf("EmbedImage failed: %v", err)
	}
	if len(imgVec) != 3 || imgVec[0] != 0.1 {
		t.Errorf("unexpected image vector: %v", imgVec)
	}
	if receivedImagePayload.Model != "embeddinggemma-2" {
		t.Errorf("expected model embeddinggemma-2, got %s", receivedImagePayload.Model)
	}
}
