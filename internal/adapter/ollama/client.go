package ollama

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

// Client implements domain.OllamaProvider using the native Ollama REST API (/api/version, /api/tags, /api/show, /api/chat).
type Client struct {
	httpClient *http.Client
}

// NewClient constructs a new Ollama REST API adapter.
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

type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
}

type chatMessage struct {
	Role    string   `json:"role"`
	Content string   `json:"content"`
	Images  []string `json:"images,omitempty"`
}

type chatResponse struct {
	Message chatMessage `json:"message"`
	Error   string      `json:"error,omitempty"`
}

// CheckStatus queries GET /api/version to determine if Ollama is online and its version string.
func (c *Client) CheckStatus(ctx context.Context, baseURL string) (string, bool) {
	baseURL = normalizeBaseURL(baseURL)
	shortCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
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
	baseURL = normalizeBaseURL(baseURL)
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
			Name:          m.Name,
			ParameterSize: m.Details.ParameterSize,
			Family:        m.Details.Family,
			VisionStatus:  "none",
		}

		// Inspect /api/show for capabilities, projector_info, and context_length
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

// ImageToPrompt encodes a local image in base64 and invokes POST /api/chat with a vision model
// and the target generator's system prompt.
func (c *Client) ImageToPrompt(ctx context.Context, baseURL, model, systemPrompt, absImagePath, userHint string) (string, error) {
	baseURL = normalizeBaseURL(baseURL)
	if strings.TrimSpace(model) == "" {
		return "", fmt.Errorf("no Ollama vision / image2prompt model configured in Settings")
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

	var messages []chatMessage
	if strings.TrimSpace(systemPrompt) != "" {
		messages = append(messages, chatMessage{
			Role:    "system",
			Content: systemPrompt,
		})
	}
	messages = append(messages, chatMessage{
		Role:    "user",
		Content: userContent,
		Images:  []string{b64},
	})

	return c.executeChat(ctx, baseURL, model, messages)
}

// TextToPrompt expands or converts an input concept or existing prompt into a target generator prompt
// using the configured Text2Prompt Ollama model and the selected generator's system prompt.
func (c *Client) TextToPrompt(ctx context.Context, baseURL, model, systemPrompt, inputConcept string) (string, error) {
	baseURL = normalizeBaseURL(baseURL)
	if strings.TrimSpace(model) == "" {
		return "", fmt.Errorf("no Ollama text2prompt model configured in Settings")
	}

	var messages []chatMessage
	if strings.TrimSpace(systemPrompt) != "" {
		messages = append(messages, chatMessage{
			Role:    "system",
			Content: systemPrompt,
		})
	}
	messages = append(messages, chatMessage{
		Role:    "user",
		Content: inputConcept,
	})

	return c.executeChat(ctx, baseURL, model, messages)
}

func (c *Client) executeChat(ctx context.Context, baseURL, model string, messages []chatMessage) (string, error) {
	payload, err := json.Marshal(chatRequest{
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

	var out chatResponse
	if err := json.Unmarshal(body, &out); err != nil {
		return "", err
	}
	if out.Error != "" {
		return "", fmt.Errorf("ollama error: %s", out.Error)
	}
	return strings.TrimSpace(out.Message.Content), nil
}

func normalizeBaseURL(u string) string {
	u = strings.TrimSpace(u)
	if u == "" {
		return "http://localhost:11434"
	}
	u = strings.TrimRight(u, "/")
	u = strings.TrimSuffix(u, "/api")
	u = strings.TrimSuffix(u, "/v1")
	return u
}
