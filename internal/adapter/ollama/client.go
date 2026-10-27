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
	"sync"
	"time"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

// Client implements domain.OllamaProvider supporting both native Ollama REST endpoints
// (/api/version, /api/tags, /api/show, /api/chat) and OpenAI-compatible endpoints (oMLX, LMStudio, vLLM, etc.)
// (/v1/models, /v1/chat/completions) on local or remote hosts (e.g. http://127.0.0.1).
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
			Family        string   `json:"family"`
			Families      []string `json:"families"`
			ParameterSize string   `json:"parameter_size"`
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
	Model    string                 `json:"model"`
	Messages []ollamaChatMessage    `json:"messages"`
	Stream   bool                   `json:"stream"`
	Think    interface{}            `json:"think,omitempty"`
	Options  map[string]interface{} `json:"options,omitempty"`
}

type ollamaChatMessage struct {
	Role     string   `json:"role"`
	Content  string   `json:"content"`
	Thinking string   `json:"thinking,omitempty"`
	Images   []string `json:"images,omitempty"`
}

type ollamaChatResponse struct {
	Message ollamaChatMessage `json:"message"`
	Error   string            `json:"error,omitempty"`
}

// oMLX / OpenAI-compatible payloads
type omlxModelsResponse struct {
	Data []struct {
		ID               string   `json:"id"`
		OwnedBy          string   `json:"owned_by"`
		MaxModelLen      int      `json:"max_model_len"`
		MaxContextLength int      `json:"max_context_length"`
		ContextLength    int      `json:"context_length"`
		Capabilities     []string `json:"capabilities"`
		Modalities       []string `json:"modalities"`
		InputModalities  []string `json:"input_modalities"`
		Vision           bool     `json:"vision"`
		SupportsVision   bool     `json:"supports_vision"`
		Thinking         bool     `json:"thinking"`
		SupportsThinking bool     `json:"supports_thinking"`
		Tools            bool     `json:"tools"`
		SupportsTools    bool     `json:"supports_tools"`
	} `json:"data"`
}

type openAIChatRequest struct {
	Model              string                 `json:"model"`
	Messages           []interface{}          `json:"messages"`
	Stream             bool                   `json:"stream"`
	ReasoningEffort    string                 `json:"reasoning_effort,omitempty"`
	ChatTemplateKwargs map[string]interface{} `json:"chat_template_kwargs,omitempty"`
	MaxTokens          int                    `json:"max_tokens,omitempty"`
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
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content,omitempty"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// CheckStatus queries GET /api/version to determine if Ollama is online and its version string.
func (c *Client) CheckStatus(ctx context.Context, baseURL string) (string, bool) {
	baseURL = NormalizeOllamaURL(baseURL)
	shortCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
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

// ListModels fetches installed models from GET /api/tags and enriches each model concurrently with
// capability metadata from POST /api/show (vision, thinking, tools, completion, context length).
func (c *Client) ListModels(ctx context.Context, baseURL string) ([]domain.OllamaModelInfo, error) {
	baseURL = NormalizeOllamaURL(baseURL)
	tagsCtx, cancel := context.WithTimeout(ctx, 4*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(tagsCtx, http.MethodGet, baseURL+"/api/tags", nil)
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

	out := make([]domain.OllamaModelInfo, len(tags.Models))
	for i, m := range tags.Models {
		hasVis := inferVisionFromName(m.Name)
		hasThink := inferThinkingFromName(m.Name)
		hasTools := inferToolsFromName(m.Name)
		visStatus := "none"
		for _, fam := range m.Details.Families {
			fl := strings.ToLower(fam)
			if fl == "clip" || fl == "mllama" || strings.Contains(fl, "vl") || strings.Contains(fl, "vision") {
				hasVis = true
			}
		}
		if hasVis {
			visStatus = "inferred"
		}
		out[i] = domain.OllamaModelInfo{
			Provider:      domain.ProviderOllama,
			Name:          m.Name,
			ParameterSize: m.Details.ParameterSize,
			Family:        m.Details.Family,
			Capabilities:  buildCapabilitySlice(true, hasVis, hasThink, hasTools, nil),
			HasVision:     hasVis,
			VisionStatus:  visStatus,
			HasThinking:   hasThink,
			HasTools:      hasTools,
		}
	}

	// Enrich concurrently via /api/show with bounded concurrency so 50+ models complete in <1.5s
	showCtx, showCancel := context.WithTimeout(ctx, 4*time.Second)
	defer showCancel()

	sem := make(chan struct{}, 10)
	var wg sync.WaitGroup
	for i := range out {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			case <-showCtx.Done():
				return
			}
			if show, err := c.inspectModel(showCtx, baseURL, out[idx].Name); err == nil {
				for _, capName := range show.Capabilities {
					switch strings.ToLower(capName) {
					case "vision":
						out[idx].HasVision = true
						out[idx].VisionStatus = "confirmed"
					case "thinking":
						out[idx].HasThinking = true
					case "tools":
						out[idx].HasTools = true
					}
				}
				if !out[idx].HasVision && show.ProjectorInfo != nil {
					if hasVis, ok := show.ProjectorInfo["clip.has_vision_encoder"].(bool); ok && hasVis {
						out[idx].HasVision = true
						out[idx].VisionStatus = "inferred"
					}
				}
				if !out[idx].HasVision && show.ModelInfo != nil {
					if vb, ok := show.ModelInfo["vision.block_count"].(float64); ok && vb > 0 {
						out[idx].HasVision = true
						out[idx].VisionStatus = "inferred"
					}
				}
				for k, v := range show.ModelInfo {
					if strings.HasSuffix(k, ".context_length") {
						if f, ok := v.(float64); ok {
							out[idx].ContextLength = int(f)
						}
					}
				}
				out[idx].Capabilities = buildCapabilitySlice(true, out[idx].HasVision, out[idx].HasThinking, out[idx].HasTools, show.Capabilities)
			}
		}(i)
	}
	wg.Wait()

	return out, nil
}

// CheckOMLXStatus checks if an OpenAI-compatible / oMLX server (e.g. http://127.0.0.1:8000) is reachable via GET /v1/models.
func (c *Client) CheckOMLXStatus(ctx context.Context, baseURL, apiKey string) (bool, int) {
	models, err := c.ListOMLXModels(ctx, baseURL, apiKey)
	if err != nil {
		return false, 0
	}
	return true, len(models)
}

// ListOMLXModels queries GET {baseURL}/v1/models on an oMLX server, extracting model IDs, capabilities, and max_model_len.
func (c *Client) ListOMLXModels(ctx context.Context, baseURL, apiKey string) ([]domain.OllamaModelInfo, error) {
	baseURL = NormalizeOMLXURL(baseURL)
	if baseURL == "" {
		return nil, fmt.Errorf("empty oMLX URL")
	}
	shortCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
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

		hasVis := m.Vision || m.SupportsVision || inferVisionFromName(m.ID)
		visStatus := "none"
		if m.Vision || m.SupportsVision {
			visStatus = "confirmed"
		} else if hasVis {
			visStatus = "inferred"
		}

		hasThink := m.Thinking || m.SupportsThinking || inferThinkingFromName(m.ID)
		hasTools := m.Tools || m.SupportsTools || inferToolsFromName(m.ID)

		for _, mod := range append(m.Modalities, m.InputModalities...) {
			ml := strings.ToLower(mod)
			if ml == "image" || ml == "vision" || ml == "multimodal" {
				hasVis = true
				visStatus = "confirmed"
			}
		}
		for _, capName := range m.Capabilities {
			switch strings.ToLower(capName) {
			case "vision", "image":
				hasVis = true
				visStatus = "confirmed"
			case "thinking", "reasoning":
				hasThink = true
			case "tools", "function_calling":
				hasTools = true
			}
		}

		caps := buildCapabilitySlice(true, hasVis, hasThink, hasTools, m.Capabilities)

		out = append(out, domain.OllamaModelInfo{
			Provider:      domain.ProviderOpenAI,
			Name:          m.ID,
			ParameterSize: inferParameterSizeFromName(m.ID),
			Family:        "MLX",
			Capabilities:  caps,
			HasVision:     hasVis,
			VisionStatus:  visStatus,
			HasThinking:   hasThink,
			HasTools:      hasTools,
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

// ImageToPrompt encodes an image (either from a local file path or in-memory data URL/base64)
// and sends it to either Ollama (/api/chat) or oMLX (/v1/chat/completions) using the target
// generator's system prompt and optional thinking level.
func (c *Client) ImageToPrompt(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt, imageSource, userHint, thinkLevel string) (string, error) {
	if strings.TrimSpace(model) == "" {
		return "", fmt.Errorf("no vision / image2prompt model configured in Settings")
	}

	var b64 string
	trimmedSource := strings.TrimSpace(imageSource)
	if strings.HasPrefix(trimmedSource, "data:image/") {
		idx := strings.Index(trimmedSource, ";base64,")
		if idx != -1 {
			b64 = trimmedSource[idx+8:]
		} else {
			b64 = trimmedSource
		}
	} else if strings.HasPrefix(trimmedSource, "base64:") {
		b64 = strings.TrimPrefix(trimmedSource, "base64:")
	} else {
		imgBytes, err := os.ReadFile(trimmedSource)
		if err != nil {
			return "", fmt.Errorf("failed to read image file: %w", err)
		}
		b64 = base64.StdEncoding.EncodeToString(imgBytes)
	}
	b64 = strings.TrimSpace(b64)

	userContent := "Analyze this image and generate a complete, production-ready generation prompt following your system instructions. Output only the final prompt without conversational filler."
	if strings.TrimSpace(userHint) != "" {
		userContent += "\nAdditional context/hint: " + strings.TrimSpace(userHint)
	}

	if provider == domain.ProviderOMLX || provider == domain.ProviderOpenAI {
		return c.executeOMLXVisionChat(ctx, NormalizeOMLXURL(baseURL), apiKey, model, systemPrompt, userContent, b64, thinkLevel)
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

	return c.executeOllamaChat(ctx, NormalizeOllamaURL(baseURL), model, messages, thinkLevel)
}

// TextToPrompt expands or converts an input concept or existing prompt into a target generator prompt
// using either Ollama (/api/chat) or oMLX (/v1/chat/completions) and optional thinking level.
func (c *Client) TextToPrompt(ctx context.Context, provider domain.AIProviderType, baseURL, apiKey, model, systemPrompt, inputConcept, thinkLevel string) (string, error) {
	if strings.TrimSpace(model) == "" {
		return "", fmt.Errorf("no text2prompt model configured in Settings")
	}

	if provider == domain.ProviderOMLX || provider == domain.ProviderOpenAI {
		return c.executeOMLXTextChat(ctx, NormalizeOMLXURL(baseURL), apiKey, model, systemPrompt, inputConcept, thinkLevel)
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

	return c.executeOllamaChat(ctx, NormalizeOllamaURL(baseURL), model, messages, thinkLevel)
}

func mapOllamaThinkParam(model, thinkLevel string) interface{} {
	tl := strings.ToLower(strings.TrimSpace(thinkLevel))
	switch tl {
	case "", "default":
		return nil
	case "off", "false", "none":
		return false
	case "low", "medium", "high":
		// GPT-OSS supports string levels ("low", "medium", "high"); standard thinking models expect boolean true.
		if strings.Contains(strings.ToLower(model), "gpt-oss") {
			return tl
		}
		return true
	case "on", "true":
		return true
	default:
		return nil
	}
}

func (c *Client) executeOllamaChat(ctx context.Context, baseURL, model string, messages []ollamaChatMessage, thinkLevel string) (string, error) {
	reqPayload := ollamaChatRequest{
		Model:    model,
		Messages: messages,
		Stream:   false,
		Think:    mapOllamaThinkParam(model, thinkLevel),
		Options: map[string]interface{}{
			"num_predict": 2048,
		},
	}
	out, statusCode, body, err := c.doOllamaChatAttempt(ctx, baseURL, reqPayload)
	if err != nil {
		return "", err
	}
	// Graceful fallback if the selected model does not support thinking or rejects the think parameter
	if statusCode == http.StatusBadRequest && reqPayload.Think != nil {
		reqPayload.Think = nil
		out, statusCode, body, err = c.doOllamaChatAttempt(ctx, baseURL, reqPayload)
		if err != nil {
			return "", err
		}
	}
	if statusCode != http.StatusOK {
		var errResp struct {
			Error string `json:"error"`
		}
		if jsonErr := json.Unmarshal(body, &errResp); jsonErr == nil && errResp.Error != "" {
			return "", fmt.Errorf("ollama error (status %d): %s", statusCode, errResp.Error)
		}
		return "", fmt.Errorf("ollama /api/chat returned status %d: %s", statusCode, string(body))
	}
	if out.Error != "" {
		return "", fmt.Errorf("ollama error: %s", out.Error)
	}
	return CleanAIPrompt(out.Message.Content), nil
}

func (c *Client) doOllamaChatAttempt(ctx context.Context, baseURL string, reqPayload ollamaChatRequest) (ollamaChatResponse, int, []byte, error) {
	var out ollamaChatResponse
	payload, err := json.Marshal(reqPayload)
	if err != nil {
		return out, 0, nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/api/chat", bytes.NewReader(payload))
	if err != nil {
		return out, 0, nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return out, 0, nil, fmt.Errorf("failed to connect to Ollama at %s: %w", baseURL, err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode == http.StatusOK {
		_ = json.Unmarshal(body, &out)
	}
	return out, resp.StatusCode, body, nil
}

func applyOMLXThinkingDirective(userPrompt, thinkLevel string) (string, string, map[string]interface{}) {
	tl := strings.ToLower(strings.TrimSpace(thinkLevel))
	switch tl {
	case "off", "false", "none":
		// For Qwen3 / MLX models on oMLX, chat_template_kwargs.enable_thinking=false disables <think> generation
		return userPrompt, "", map[string]interface{}{"enable_thinking": false}
	case "low", "medium", "high":
		return userPrompt, tl, map[string]interface{}{"enable_thinking": true}
	case "on", "true":
		return userPrompt, "medium", map[string]interface{}{"enable_thinking": true}
	default:
		return userPrompt, "", nil
	}
}

func (c *Client) executeOMLXTextChat(ctx context.Context, baseURL, apiKey, model, systemPrompt, userPrompt, thinkLevel string) (string, error) {
	adjPrompt, effort, kwargs := applyOMLXThinkingDirective(userPrompt, thinkLevel)
	var messages []interface{}
	if strings.TrimSpace(systemPrompt) != "" {
		messages = append(messages, openAITextMessage{Role: "system", Content: systemPrompt})
	}
	messages = append(messages, openAITextMessage{Role: "user", Content: adjPrompt})
	return c.doOMLXChat(ctx, baseURL, apiKey, model, messages, effort, kwargs)
}

func (c *Client) executeOMLXVisionChat(ctx context.Context, baseURL, apiKey, model, systemPrompt, userPrompt, b64Image, thinkLevel string) (string, error) {
	adjPrompt, effort, kwargs := applyOMLXThinkingDirective(userPrompt, thinkLevel)
	var messages []interface{}
	if strings.TrimSpace(systemPrompt) != "" {
		messages = append(messages, openAITextMessage{Role: "system", Content: systemPrompt})
	}
	messages = append(messages, openAIVisionMessage{
		Role: "user",
		Content: []openAIVisionBlock{
			{Type: "text", Text: adjPrompt},
			{Type: "image_url", ImageURL: &openAIImageURL{URL: "data:image/png;base64," + b64Image}},
		},
	})
	return c.doOMLXChat(ctx, baseURL, apiKey, model, messages, effort, kwargs)
}

func (c *Client) doOMLXChat(ctx context.Context, baseURL, apiKey, model string, messages []interface{}, effort string, kwargs map[string]interface{}) (string, error) {
	payload, err := json.Marshal(openAIChatRequest{
		Model:              model,
		Messages:           messages,
		Stream:             false,
		ReasoningEffort:    effort,
		ChatTemplateKwargs: kwargs,
		MaxTokens:          2048,
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
		var errResp struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
			Message string `json:"message"`
		}
		if jsonErr := json.Unmarshal(body, &errResp); jsonErr == nil {
			if errResp.Error.Message != "" {
				return "", fmt.Errorf("omlx error (status %d): %s", resp.StatusCode, errResp.Error.Message)
			}
			if errResp.Message != "" {
				return "", fmt.Errorf("omlx error (status %d): %s", resp.StatusCode, errResp.Message)
			}
		}
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
	return CleanAIPrompt(out.Choices[0].Message.Content), nil
}

// CleanAIPrompt cleans up LLM output for diffusion synthesis:
// - Removes <think>...</think> and unclosed <think> blocks
// - Normalizes escaped characters (\r\n, \n, \r, \t, /r/n) into real line breaks
// - Extracts prompt from markdown code fences (```markdown ... ``` or ~~~)
// - Strips conversational preambles ("Here is the prompt for Flux:", "Prompt:", etc.)
// - Strips conversational postambles ("Hope this helps!", "Let me know...", etc.)
// - Trims leading/trailing quotes and backticks
func CleanAIPrompt(raw string) string {
	s := stripThinkTags(raw)
	if s == "" {
		return ""
	}

	// 1. Normalize literal string escape sequences if present (\r\n, \n, \r, \t, /r/n)
	if strings.Contains(s, `\r\n`) {
		s = strings.ReplaceAll(s, `\r\n`, "\n")
	}
	if strings.Contains(s, `\n`) {
		s = strings.ReplaceAll(s, `\n`, "\n")
	}
	if strings.Contains(s, `\r`) {
		s = strings.ReplaceAll(s, `\r`, "\n")
	}
	if strings.Contains(s, `\t`) {
		s = strings.ReplaceAll(s, `\t`, " ")
	}
	s = strings.ReplaceAll(s, "/r/n", "\n")
	s = strings.ReplaceAll(s, "/r", "\n")
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")

	// 2. Extract content from markdown code blocks if the model wrapped the prompt in ```...```
	if openIdx := strings.Index(s, "```"); openIdx != -1 {
		rest := s[openIdx+3:]
		newlineIdx := strings.Index(rest, "\n")
		var blockContent string
		if newlineIdx != -1 {
			blockContent = rest[newlineIdx+1:]
		} else {
			blockContent = rest
		}

		if closeIdx := strings.Index(blockContent, "```"); closeIdx != -1 {
			s = strings.TrimSpace(blockContent[:closeIdx])
		} else {
			s = strings.TrimSpace(blockContent)
		}
	} else if openTilde := strings.Index(s, "~~~"); openTilde != -1 {
		rest := s[openTilde+3:]
		newlineIdx := strings.Index(rest, "\n")
		var blockContent string
		if newlineIdx != -1 {
			blockContent = rest[newlineIdx+1:]
		} else {
			blockContent = rest
		}
		if closeIdx := strings.Index(blockContent, "~~~"); closeIdx != -1 {
			s = strings.TrimSpace(blockContent[:closeIdx])
		} else {
			s = strings.TrimSpace(blockContent)
		}
	}

	// 3. Strip conversational preambles line-by-line
	lines := strings.Split(s, "\n")
	startLine := 0
	for startLine < len(lines) {
		trimmed := strings.TrimSpace(lines[startLine])
		if trimmed == "" {
			startLine++
			continue
		}
		lower := strings.ToLower(trimmed)
		if isConversationalPreamble(lower) {
			startLine++
			continue
		}
		if stripped := stripPromptLabel(trimmed); stripped != trimmed {
			lines[startLine] = stripped
			break
		}
		break
	}
	lines = lines[startLine:]

	// 4. Strip conversational postambles
	endLine := len(lines)
	for endLine > 0 {
		trimmed := strings.TrimSpace(lines[endLine-1])
		if trimmed == "" {
			endLine--
			continue
		}
		lower := strings.ToLower(trimmed)
		if isConversationalPostamble(lower) {
			endLine--
			continue
		}
		break
	}
	lines = lines[:endLine]

	s = strings.Join(lines, "\n")
	s = strings.TrimSpace(s)

	// Strip enclosing backticks or quotes if wrapped
	if strings.HasPrefix(s, "`") && strings.HasSuffix(s, "`") && len(s) > 1 {
		s = strings.Trim(s, "`")
	}
	if strings.HasPrefix(s, `"`) && strings.HasSuffix(s, `"`) && len(s) > 1 {
		s = strings.Trim(s, `"`)
	}

	trimmedCandidate := strings.TrimSpace(s)
	// If the model produced a JSON object (e.g. {"prompt": "..."}), extract the prompt value
	if strings.HasPrefix(trimmedCandidate, "{") && strings.HasSuffix(trimmedCandidate, "}") {
		var obj map[string]interface{}
		if err := json.Unmarshal([]byte(trimmedCandidate), &obj); err == nil && len(obj) > 0 {
			for _, k := range []string{"prompt", "positive_prompt", "positive", "diffusion_prompt", "output", "image_prompt", "scene_prompt", "result", "caption"} {
				if v, ok := obj[k]; ok {
					if strVal, ok := v.(string); ok && strings.TrimSpace(strVal) != "" {
						return strings.TrimSpace(strVal)
					}
				}
			}
		}
	}

	return trimmedCandidate
}

func isConversationalPreamble(lower string) bool {
	cleaned := strings.TrimLeft(lower, "*#-> \t")
	prefixes := []string{
		"here is the prompt",
		"here is your prompt",
		"here's the prompt",
		"here's your prompt",
		"here is a prompt",
		"here's a prompt",
		"here is the revised prompt",
		"here's the revised prompt",
		"here is the generated prompt",
		"here's the generated prompt",
		"here is the final prompt",
		"here is a refined prompt",
		"here's a refined prompt",
		"sure, here is",
		"sure! here is",
		"sure, here's",
		"certainly! here is",
		"certainly! here's",
		"certainly, here is",
		"based on the image",
		"based on the concept",
		"prompt for flux",
		"prompt for krea",
		"prompt for sdxl",
		"prompt for qwen",
		"prompt for minimax",
		"prompt for moss",
	}
	for _, p := range prefixes {
		if strings.HasPrefix(cleaned, p) {
			return true
		}
	}
	return false
}

func stripPromptLabel(line string) string {
	cleaned := strings.TrimSpace(line)
	lower := strings.ToLower(cleaned)
	labelPrefixes := []string{
		"**prompt:**",
		"**prompt**:",
		"prompt:",
		"**positive prompt:**",
		"**positive prompt**:",
		"positive prompt:",
		"**final prompt:**",
		"final prompt:",
		"**generation prompt:**",
		"generation prompt:",
		"**result:**",
		"result:",
		"output:",
	}
	for _, lp := range labelPrefixes {
		if strings.HasPrefix(lower, lp) {
			return strings.TrimSpace(cleaned[len(lp):])
		}
	}
	return line
}

func isConversationalPostamble(lower string) bool {
	cleaned := strings.TrimLeft(lower, "*#-> \t")
	prefixes := []string{
		"hope this helps",
		"hope you like it",
		"let me know if you need",
		"let me know if you want",
		"feel free to ask",
		"enjoy generating",
		"happy generating",
		"notes:",
		"**notes:**",
		"parameters:",
		"**parameters:**",
		"settings:",
		"**settings:**",
	}
	for _, p := range prefixes {
		if strings.HasPrefix(cleaned, p) {
			return true
		}
	}
	return false
}

// stripThinkTags removes any inline <think>...</think> or <thinking>...</thinking> blocks from the final prompt output.
func stripThinkTags(raw string) string {
	s := strings.TrimSpace(raw)
	for _, tag := range []struct{ open, close string }{
		{"<think>", "</think>"},
		{"<thinking>", "</thinking>"},
		{"<thought>", "</thought>"},
	} {
		for {
			start := strings.Index(strings.ToLower(s), tag.open)
			if start == -1 {
				break
			}
			end := strings.Index(strings.ToLower(s[start:]), tag.close)
			if end == -1 {
				// Unclosed thinking block (truncated): strip to the end
				s = strings.TrimSpace(s[:start])
				break
			}
			s = strings.TrimSpace(s[:start] + s[start+end+len(tag.close):])
		}
	}
	return strings.TrimSpace(s)
}

// NormalizeOllamaURL normalizes local or remote host strings (e.g. "http://127.0.0.1" -> "http://127.0.0.1:11434").
func NormalizeOllamaURL(raw string) string {
	return normalizeEndpoint(raw, "http://127.0.0.1:11434", "11434")
}

// NormalizeOMLXURL normalizes local or remote OpenAI-compatible host strings (e.g. "http://127.0.0.1" -> "http://127.0.0.1:8000").
func NormalizeOMLXURL(raw string) string {
	return normalizeEndpoint(raw, "http://127.0.0.1:8000", "8000")
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

func inferVisionFromName(modelID string) bool {
	lower := strings.ToLower(modelID)
	// Modern multimodal & MLX VLM architectures (Bonsai, Qwen3.5/3.8/SuperQwen, Qwen-VL, Gemma-3/4, Pixtral, LLaVA, MiniCPM-V, InternVL, Molmo, GLM-4V)
	for _, kw := range []string{
		"vl", "vision", "vlm", "pixtral", "llava", "minicpm-v", "minicpm",
		"bonsai", "bonsai2", "bonsai-2",
		"gemma-3", "gemma3", "gemma-4", "gemma4", "paligemma",
		"qwen2.5-vl", "qwen3-vl", "qwen3.5", "qwen3.8", "superqwen",
		"internvl", "molmo", "glm-4v", "qvq", "mllama", "mistral-small3.1", "mistral-small3.2",
		"florence", "omni", "deepseek-vl", "idefics", "smolvlm", "phi-3.5-vision", "phi-3-vision",
		"janus", "chameleon", "aria", "cogvlm",
	} {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	return false
}

func inferThinkingFromName(modelID string) bool {
	lower := strings.ToLower(modelID)
	for _, kw := range []string{
		"qwen3", "superqwen", "deepseek-r1", "r1", "qwq", "qvq",
		"thinking", "reason", "bonsai", "gemma4", "gemma-4", "gpt-oss", "lfm2-thinking", "lfm2.5-thinking", "magistral", "cogito",
	} {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	return false
}

func inferToolsFromName(modelID string) bool {
	lower := strings.ToLower(modelID)
	for _, kw := range []string{
		"qwen", "llama3", "llama-3", "mistral", "gemma3", "gemma-3", "gemma4", "gemma-4",
		"bonsai", "command-r", "hermes", "nemotron", "gpt-oss", "lfm2", "granite",
	} {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	return false
}

func inferParameterSizeFromName(modelID string) string {
	parts := strings.FieldsFunc(modelID, func(r rune) bool {
		return r == '-' || r == '_' || r == ':'
	})
	for _, p := range parts {
		pl := strings.ToLower(p)
		if strings.HasSuffix(pl, "b") && len(pl) >= 2 && pl[0] >= '0' && pl[0] <= '9' {
			return strings.ToUpper(p)
		}
	}
	return ""
}

func buildCapabilitySlice(completion, vision, thinking, tools bool, existing []string) []string {
	seen := make(map[string]bool)
	var out []string
	add := func(c string) {
		c = strings.ToLower(strings.TrimSpace(c))
		if c != "" && !seen[c] {
			seen[c] = true
			out = append(out, c)
		}
	}
	if completion {
		add("completion")
	}
	for _, c := range existing {
		add(c)
	}
	if vision {
		add("vision")
	}
	if thinking {
		add("thinking")
	}
	if tools {
		add("tools")
	}
	return out
}
