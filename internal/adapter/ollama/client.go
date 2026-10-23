package ollama

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

// Client implements domain.OllamaProvider supporting both native Ollama REST endpoints
// (/api/version, /api/tags, /api/show, /api/chat) and oMLX OpenAI-compatible endpoints
// (/v1/models, /v1/chat/completions) on local or remote hosts (e.g. http://mac-mini.local).
type Client struct {
	httpClient *http.Client
}

// NewClient constructs a dual Ollama / oMLX REST API adapter.
func NewClient() *Client {
	return &Client{
		httpClient: &http.Client{
			Timeout: 90 * time.Second,
		},
	}
}

type versionResponse struct {
	Version string `json:"version"`
}

type tagsResponse struct {
	Models []struct {
		Name    string `json:"name"`
		Details struct {
			Family        string `json:"family"`
			ParameterSize string `json:"parameter_size"`
		} `json:"details"`
	} `json:"models"`
}

type showRequest struct {
	Model string `json:"model"`
}

type showResponse struct {
	Capabilities  []string               `json:"capabilities"`
	ModelInfo     map[string]interface{} `json:"model_info"`
	ProjectorInfo map[string]interface{} `json:"projector_info"`
}

type ollamaChatRequest struct {
	Model    string              `json:"model"`
	Messages []ollamaChatMessage `json:"messages"`
	Stream   bool                `json:"stream"`
}

type ollamaChatMessage struct {
	Role    string   `json:"role"`
	Content string   `json:"content"`
	Images  []string `json:"images,omitempty"`
}

type ollamaChatResponse struct {
	Message ollamaChatMessage `json:"message"`
	Error   string            `json:"error,omitempty"`
}

// oMLX / OpenAI-compatible payloads
type omlxModelsResponse struct {
	Data []struct {
		ID               string `json:"id"`
		OwnedBy          string `json:"owned_by"`
		MaxModelLen      int    `json:"max_model_len"`
		MaxContextLength int    `json:"max_context_length"`
		ContextLength    int    `json:"context_length"`
	} `json:"data"`
}

type openAIChatRequest struct {
	Model    string        `json:"model"`
	Messages []interface{} `json:"messages"`
	Stream   bool          `json:"stream"`
}

type openAITextMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIVisionMessage struct {
	Role    string              `json:"role"`
	Content []openAIVisionBlock `json:"content"`
}

type openAIVisionBlock struct {
	Type     string          `json:"type"`
	Text     string          `json:"text,omitempty"`
	ImageURL *openAIImageURL `json:"image_url,omitempty"`
}

type openAIImageURL struct {
	URL string `json:"url"`
}

type openAIChatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// CheckStatus queries GET /api/version to determine if Ollama is online and its version string.
func (c *Client) CheckStatus(ctx context.Context, baseURL string) (string, bool) {
	baseURL = NormalizeOllamaURL(baseURL)
	shortCtx, cancel := context.WithTimeout(ctx, 2500*time.Millisecond)
	defer cancel()

	req, err := http.NewRequestWithContext(shortCtx, http.MethodGet, baseURL+"/api/version", nil)
	if err != nil {
		return "", false
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", false
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", false
	}
	var out versionResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "online", true
	}
	return out.Version, true
}

// ListModels fetches installed models from GET /api/tags and enriches each model with
// capability metadata from POST /api/show (vision confirmed vs inferred, context length).
func (c *Client) ListModels(ctx context.Context, baseURL string) ([]domain.OllamaModelInfo, error) {
	baseURL = NormalizeOllamaURL(baseURL)
	shortCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(shortCtx, http.MethodGet, baseURL+"/api/tags", nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama /api/tags returned status %d", resp.StatusCode)
	}

	var tags tagsResponse
	if err := json.NewDecoder(resp.Body).Decode(&tags); err != nil {
		return nil, err
	}

	var out []domain.OllamaModelInfo
	for _, m := range tags.Models {
		info := domain.OllamaModelInfo{
			Provider:      domain.ProviderOllama,
			Name:          m.Name,
			ParameterSize: m.Details.ParameterSize,
			Family:        m.Details.Family,
			VisionStatus:  "none",
		}

		if show, err := c.inspectModel(shortCtx, baseURL, m.Name); err == nil {
			info.Capabilities = show.Capabilities
			for _, capName := range show.Capabilities {
				if capName == "vision" {
					info.HasVision = true
					info.VisionStatus = "confirmed"
				}
			}
			if !info.HasVision && show.ProjectorInfo != nil {
				if hasVis, ok := show.ProjectorInfo["clip.has_vision_encoder"].(bool); ok && hasVis {
					info.HasVision = true
					info.VisionStatus = "inferred"
				}
			}
			for k, v := range show.ModelInfo {
				if strings.HasSuffix(k, ".context_length") {
					if f, ok := v.(float64); ok {
						info.ContextLength = int(f)
					}
				}
			}
		}

		out = append(out, info)
	}
	return out, nil
}

// CheckOMLXStatus checks if an oMLX server (e.g. http://mac-mini.local:8000) is reachable via GET /v1/models.
func (c *Client) CheckOMLXStatus(ctx context.Context, baseURL, apiKey string) (bool, int) {
	models, err := c.ListOMLXModels(ctx, baseURL, apiKey)
	if err != nil {
		return false, 0
	}
	return true, len(models)
}

// ListOMLXModels queries GET {baseURL}/v1/models on an oMLX server, extracting model IDs and max_model_len.
func (c *Client) ListOMLXModels(ctx context.Context, baseURL, apiKey string) ([]domain.OllamaModelInfo, error) {
	baseURL = NormalizeOMLXURL(baseURL)
	if baseURL == "" {
		return nil, fmt.Errorf("empty oMLX URL")
	}
	shortCtx, cancel := context.WithTimeout(ctx, 3500*time.Millisecond)
	defer cancel()

	req, err := http.NewRequestWithContext(shortCtx, http.MethodGet, baseURL+"/v1/models", nil)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(apiKey) != "" {
		req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(apiKey))
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("omlx /v1/models returned status %d", resp.StatusCode)
	}

	var parsed omlxModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, err
	}

	var out []domain.OllamaModelInfo
	for _, m := range parsed.Data {
		ctxLen := m.MaxModelLen
		if ctxLen == 0 {
			ctxLen = m.MaxContextLength
		}
		if ctxLen == 0 {
			ctxLen = m.ContextLength
		}

		hasVis := inferOMLXVisionFromName(m.ID)
		visStatus := "none"
		caps := []string{"completion"}
		if hasVis {
			visStatus = "inferred"
			caps = append(caps, "vision")
		}

		out = append(out, domain.OllamaModelInfo{
			Provider:      domain.ProviderOMLX,
			Name:          m.ID,
			Family:        "MLX",
			Capabilities:  caps,
			HasVision:     hasVis,
			VisionStatus:  visStatus,
			ContextLength: ctxLen,
		})
	}
	return out, nil
}

func (c *Client) inspectModel(ctx context.Context, baseURL, modelName string) (*showResponse, error) {
	payload, _ := json.Marshal(showRequest{Model: modelName})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/api/show", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("status %d", resp.StatusCode)
	}
	var out showResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return &out, nil
}

// ImageToPrompt encodes a local image in base64 and sends it to either Ollama (/api/chat)
// or oMLX (/v1/chat/completions) using the target generator's system prompt.
func (c *Client) ImageToPrompt(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt, absImagePath, userHint string) (string, error) {
	if strings.TrimSpace(model) == "" {
		return "", fmt.Errorf("no vision / image2prompt model configured in Settings")
	}

	imgBytes, err := os.ReadFile(absImagePath)
	if err != nil {
		return "", fmt.Errorf("failed to read image file: %w", err)
	}
	b64 := base64.StdEncoding.EncodeToString(imgBytes)

	userContent := "Analyze this image and generate a complete, production-ready generation prompt following your system instructions. Output only the final prompt without conversational filler."
	if strings.TrimSpace(userHint) != "" {
		userContent += "\nAdditional context/hint: " + strings.TrimSpace(userHint)
	}

	if provider == domain.ProviderOMLX {
		return c.executeOMLXVisionChat(ctx, NormalizeOMLXURL(baseURL), apiKey, model, systemPrompt, userContent, b64)
	}

	var messages []ollamaChatMessage
	if strings.TrimSpace(systemPrompt) != "" {
		messages = append(messages, ollamaChatMessage{
			Role:    "system",
			Content: systemPrompt,
		})
	}
	messages = append(messages, ollamaChatMessage{
		Role:    "user",
		Content: userContent,
		Images:  []string{b64},
	})

	return c.executeOllamaChat(ctx, NormalizeOllamaURL(baseURL), model, messages)
}

// TextToPrompt expands or converts an input concept or existing prompt into a target generator prompt
// using either Ollama (/api/chat) or oMLX (/v1/chat/completions).
func (c *Client) TextToPrompt(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt, inputConcept string) (string, error) {
	if strings.TrimSpace(model) == "" {
		return "", fmt.Errorf("no text2prompt model configured in Settings")
	}

	if provider == domain.ProviderOMLX {
		return c.executeOMLXTextChat(ctx, NormalizeOMLXURL(baseURL), apiKey, model, systemPrompt, inputConcept)
	}

	var messages []ollamaChatMessage
	if strings.TrimSpace(systemPrompt) != "" {
		messages = append(messages, ollamaChatMessage{
			Role:    "system",
			Content: systemPrompt,
		})
	}
	messages = append(messages, ollamaChatMessage{
		Role:    "user",
		Content: inputConcept,
	})

	return c.executeOllamaChat(ctx, NormalizeOllamaURL(baseURL), model, messages)
}

func (c *Client) executeOllamaChat(ctx context.Context, baseURL, model string, messages []ollamaChatMessage) (string, error) {
	payload, err := json.Marshal(ollamaChatRequest{
		Model:    model,
		Messages: messages,
		Stream:   false,
	})
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/api/chat", bytes.NewReader(payload))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to connect to Ollama at %s: %w", baseURL, err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("ollama /api/chat returned status %d: %s", resp.StatusCode, string(body))
	}

	var out ollamaChatResponse
	if err := json.Unmarshal(body, &out); err != nil {
		return "", err
	}
	if out.Error != "" {
		return "", fmt.Errorf("ollama error: %s", out.Error)
	}
	return strings.TrimSpace(out.Message.Content), nil
}

func (c *Client) executeOMLXTextChat(ctx context.Context, baseURL, apiKey, model, systemPrompt, userPrompt string) (string, error) {
	var messages []interface{}
	if strings.TrimSpace(systemPrompt) != "" {
		messages = append(messages, openAITextMessage{Role: "system", Content: systemPrompt})
	}
	messages = append(messages, openAITextMessage{Role: "user", Content: userPrompt})
	return c.doOMLXChat(ctx, baseURL, apiKey, model, messages)
}

func (c *Client) executeOMLXVisionChat(ctx context.Context, baseURL, apiKey, model, systemPrompt, userPrompt, b64Image string) (string, error) {
	var messages []interface{}
	if strings.TrimSpace(systemPrompt) != "" {
		messages = append(messages, openAITextMessage{Role: "system", Content: systemPrompt})
	}
	messages = append(messages, openAIVisionMessage{
		Role: "user",
		Content: []openAIVisionBlock{
			{Type: "text", Text: userPrompt},
			{Type: "image_url", ImageURL: &openAIImageURL{URL: "data:image/png;base64," + b64Image}},
		},
	})
	return c.doOMLXChat(ctx, baseURL, apiKey, model, messages)
}

func (c *Client) doOMLXChat(ctx context.Context, baseURL, apiKey, model string, messages []interface{}) (string, error) {
	payload, err := json.Marshal(openAIChatRequest{
		Model:    model,
		Messages: messages,
		Stream:   false,
	})
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/v1/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	if strings.TrimSpace(apiKey) != "" {
		req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(apiKey))
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to connect to oMLX at %s: %w", baseURL, err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("omlx /v1/chat/completions returned status %d: %s", resp.StatusCode, string(body))
	}

	var out openAIChatResponse
	if err := json.Unmarshal(body, &out); err != nil {
		return "", err
	}
	if out.Error != nil && out.Error.Message != "" {
		return "", fmt.Errorf("omlx error: %s", out.Error.Message)
	}
	if len(out.Choices) == 0 {
		return "", fmt.Errorf("omlx returned empty choices")
	}
	return strings.TrimSpace(out.Choices[0].Message.Content), nil
}

// NormalizeOllamaURL normalizes local or remote host strings (e.g. "http://mac-mini.local" -> "http://mac-mini.local:11434").
func NormalizeOllamaURL(raw string) string {
	return normalizeEndpoint(raw, "http://localhost:11434", "11434")
}

// NormalizeOMLXURL normalizes local or remote oMLX host strings (e.g. "http://mac-mini.local" -> "http://mac-mini.local:8000").
func NormalizeOMLXURL(raw string) string {
	return normalizeEndpoint(raw, "http://mac-mini.local:8000", "8000")
}

func normalizeEndpoint(raw, fallback, defaultPort string) string {
	u := strings.TrimSpace(raw)
	if u == "" {
		return fallback
	}
	if !strings.Contains(u, "://") {
		u = "http://" + u
	}
	u = strings.TrimRight(u, "/")
	u = strings.TrimSuffix(u, "/api")
	u = strings.TrimSuffix(u, "/v1")

	parsed, err := url.Parse(u)
	if err == nil && parsed.Scheme == "http" && parsed.Port() == "" && parsed.Host != "" {
		parsed.Host = parsed.Host + ":" + defaultPort
		return strings.TrimRight(parsed.String(), "/")
	}
	return u
}

func inferOMLXVisionFromName(modelID string) bool {
	lower := strings.ToLower(modelID)
	for _, kw := range []string{"vl", "vision", "pixtral", "llava", "minicpm-v", "gemma-3", "gemma3", "gemma4", "qwen2.5-vl", "qwen3-vl"} {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	return false
}
