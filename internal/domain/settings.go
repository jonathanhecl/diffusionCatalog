package domain

import (
	"strings"
	"time"
)

// AIProviderType identifies which inference server handles a task ("ollama" or "openai" / "omlx").
type AIProviderType string

const (
	ProviderOllama AIProviderType = "ollama"
	ProviderOpenAI AIProviderType = "openai"
	ProviderOMLX   AIProviderType = "omlx" // legacy alias for ProviderOpenAI
)

// AppSettings stores configurable application options and dual Ollama / OpenAI model routing preferences.
type AppSettings struct {
	// 1. Ollama Provider
	OllamaEnabled      bool   `json:"ollama_enabled"`
	OllamaBaseURL      string `json:"ollama_base_url"`
	OllamaDefaultModel string `json:"ollama_default_model"`
	OllamaDefaultThink string `json:"ollama_default_think"` // "default", "off", "low", "medium", "high"

	// 2. OpenAI-Compatible Provider (oMLX, vLLM, OpenAI)
	OpenAIEnabled      bool   `json:"openai_enabled"`
	OpenAIBaseURL      string `json:"openai_base_url"`
	OpenAIAPIKey       string `json:"openai_api_key"`
	OpenAIDefaultModel string `json:"openai_default_model"`
	OpenAIDefaultThink string `json:"openai_default_think"` // "default", "off", "low", "medium", "high"

	// 3 & 4. Context Defaults
	RemixDefaultProvider  AIProviderType `json:"remix_default_provider"`  // "ollama" or "openai"
	StudioDefaultProvider AIProviderType `json:"studio_default_provider"` // "ollama" or "openai"

	// General
	DefaultTargetGenerator string `json:"default_target_generator"`
	AutoCaptionOnScan      bool   `json:"auto_caption_on_scan"`

	// Multimodal Embedding Settings
	EmbeddingEnabled  bool           `json:"embedding_enabled"`
	EmbeddingProvider AIProviderType `json:"embedding_provider"`
	EmbeddingModel    string         `json:"embedding_model"`

	// Legacy / Compatibility fields (kept in sync for backwards compatibility)
	OMLXBaseURL          string         `json:"omlx_base_url,omitempty"`
	OMLXAPIKey           string         `json:"omlx_api_key,omitempty"`
	VisionProvider       AIProviderType `json:"vision_provider,omitempty"`
	VisionModel          string         `json:"vision_model,omitempty"`
	VisionThink          string         `json:"vision_think,omitempty"`
	Image2PromptProvider AIProviderType `json:"image2prompt_provider,omitempty"`
	Image2PromptModel    string         `json:"image2prompt_model,omitempty"`
	Image2PromptThink    string         `json:"image2prompt_think,omitempty"`
	Text2PromptProvider  AIProviderType `json:"text2prompt_provider,omitempty"`
	Text2PromptModel     string         `json:"text2prompt_model,omitempty"`
	Text2PromptThink     string         `json:"text2prompt_think,omitempty"`
}

// GetProviderDefaults returns the default model and thinking level for a given provider.
func (s AppSettings) GetProviderDefaults(provider AIProviderType) (string, string) {
	if provider == ProviderOpenAI || provider == ProviderOMLX {
		model := s.OpenAIDefaultModel
		if model == "" {
			model = s.Text2PromptModel
		}
		think := s.OpenAIDefaultThink
		if think == "" {
			think = s.Text2PromptThink
		}
		if think == "" {
			think = "low"
		}
		return model, think
	}
	model := s.OllamaDefaultModel
	if model == "" {
		model = s.VisionModel
	}
	if model == "" {
		model = s.Image2PromptModel
	}
	think := s.OllamaDefaultThink
	if think == "" {
		think = s.VisionThink
	}
	if think == "" {
		think = "off"
	}
	return model, think
}

// GetRemixDefaults returns the default provider, model, and thinking level for Remix tasks.
func (s AppSettings) GetRemixDefaults() (AIProviderType, string, string) {
	prov := s.RemixDefaultProvider
	if prov == "" {
		if s.Image2PromptProvider != "" {
			prov = s.Image2PromptProvider
		} else {
			prov = ProviderOllama
		}
	}
	mod, think := s.GetProviderDefaults(prov)
	return prov, mod, think
}

// GetStudioDefaults returns the default provider, model, and thinking level for Studio 2D / General tasks.
func (s AppSettings) GetStudioDefaults() (AIProviderType, string, string) {
	prov := s.StudioDefaultProvider
	if prov == "" {
		if s.Text2PromptProvider != "" {
			prov = s.Text2PromptProvider
		} else {
			prov = ProviderOpenAI
		}
	}
	mod, think := s.GetProviderDefaults(prov)
	return prov, mod, think
}

// PromptProfile defines a target diffusion or video generator (e.g., flux, krea2, moss, minimaxh3)
// and the dedicated system prompt used by Ollama/oMLX for image2prompt and text2prompt generation.
type PromptProfile struct {
	ID             int64     `json:"id"`
	Slug           string    `json:"slug"`
	Name           string    `json:"name"`
	TargetModality MediaType `json:"target_modality"` // "image" or "video"
	Description    string    `json:"description"`
	SystemPrompt   string    `json:"system_prompt"`
	UpdatedAt      time.Time `json:"updated_at"`
}

// TokenCount returns the estimated LLM token count of the system prompt.
func (p *PromptProfile) TokenCount() int {
	return EstimateTokens(p.SystemPrompt)
}

// ModelLoRA stores documentation, trigger words, weight syntax, and few-shot usage examples
// for a LoRA scoped to a specific target generator model (e.g. flux, krea2, qwen-image, minimaxh3).
type ModelLoRA struct {
	ID                int64     `json:"id"`
	ProfileSlug       string    `json:"profile_slug"`       // e.g. "flux", "krea2", "qwen-image", "minimaxh3"
	Name              string    `json:"name"`               // e.g. "Anamorphic Halation v2"
	SyntaxTag         string    `json:"syntax_tag"`         // e.g. "<lora:krea2_halation:0.75>" or empty if trigger-only
	TriggerWords      string    `json:"trigger_words"`      // e.g. "anamorphic_halation, tungsten bloom" (empty if no trigger)
	RecommendedWeight string    `json:"recommended_weight"` // e.g. "0.75"
	Description       string    `json:"description"`        // What the LoRA is for and how it behaves on this engine
	UsageExamples     string    `json:"usage_examples"`     // Concrete example prompts showing proper usage
	CreatedAt         time.Time `json:"created_at"`
}

// TokenCount returns the estimated LLM token count of the LoRA documentation and usage examples.
func (l *ModelLoRA) TokenCount() int {
	combined := strings.TrimSpace(l.SyntaxTag + " " + l.TriggerWords + " " + l.Description + " " + l.UsageExamples)
	return EstimateTokens(combined)
}

// OllamaModelInfo represents a model discovered from either an Ollama server (/api/tags + /api/show)
// or an oMLX OpenAI-compatible server (/v1/models).
type OllamaModelInfo struct {
	Provider      AIProviderType `json:"provider"` // "ollama" or "omlx"
	Name          string         `json:"name"`
	ParameterSize string         `json:"parameter_size"`
	Family        string         `json:"family"`
	Capabilities  []string       `json:"capabilities"`
	HasCompletion bool           `json:"has_completion"`
	HasVision     bool           `json:"has_vision"`
	VisionStatus  string         `json:"vision_status"` // "confirmed", "inferred", or "none"
	HasThinking   bool           `json:"has_thinking"`
	HasTools      bool           `json:"has_tools"`
	HasEmbedding  bool           `json:"has_embedding"`
	ContextLength int            `json:"context_length"`
}

// HasCapability checks whether a specific capability string is declared for this model.
func (m OllamaModelInfo) HasCapability(capName string) bool {
	target := strings.ToLower(strings.TrimSpace(capName))
	for _, c := range m.Capabilities {
		if strings.ToLower(strings.TrimSpace(c)) == target {
			return true
		}
	}
	return false
}

// IsCompletionModel returns true if the model is capable of generating prompt completions.
// Models strictly intended for embeddings, decision/classification, or speech/audio are excluded.
func (m OllamaModelInfo) IsCompletionModel() bool {
	if m.HasCompletion {
		return true
	}
	if len(m.Capabilities) > 0 {
		return m.HasCapability("completion")
	}
	nameLower := strings.ToLower(m.Name)
	famLower := strings.ToLower(m.Family)
	for _, banned := range []string{"embed", "decider", "decision", "rerank", "phonon", "whisper", "tts", "stt"} {
		if strings.Contains(nameLower, banned) || strings.Contains(famLower, banned) {
			return false
		}
	}
	return true
}

// IsEmbeddingVision returns true if the model supports both embedding and vision.
func (m OllamaModelInfo) IsEmbeddingVision() bool {
	hasVis := m.HasVision
	hasEmb := m.HasEmbedding
	for _, c := range m.Capabilities {
		cl := strings.ToLower(c)
		if cl == "vision" {
			hasVis = true
		}
		if cl == "embedding" {
			hasEmb = true
		}
	}
	nameLower := strings.ToLower(m.Name)
	famLower := strings.ToLower(m.Family)
	if strings.Contains(nameLower, "embed") || strings.Contains(famLower, "embed") ||
		strings.Contains(nameLower, "clip") || strings.Contains(famLower, "clip") ||
		strings.Contains(nameLower, "siglip") || strings.Contains(famLower, "siglip") {
		hasEmb = true
	}
	if strings.Contains(nameLower, "vision") || strings.Contains(nameLower, "-vl") || strings.Contains(nameLower, "_vl") {
		hasVis = true
	}
	return hasVis && hasEmb
}

// PromptExecutionOverride allows overriding the default Settings AI backend/model and thinking level
// at prompt-generation time without modifying persistent application settings.
type PromptExecutionOverride struct {
	Provider   AIProviderType `json:"provider,omitempty"`
	Model      string         `json:"model,omitempty"`
	ThinkLevel string         `json:"think_level,omitempty"` // "", "default", "off", "low", "medium", "high"
}

// GeneratedPrompt records a prompt synthesized by Ollama/oMLX from either a cataloged Asset or a Character folder.
type GeneratedPrompt struct {
	ID                int64          `json:"id"`
	AssetID           *int64         `json:"asset_id,omitempty"`
	CharacterID       *int64         `json:"character_id,omitempty"`
	CharacterName     string         `json:"character_name,omitempty"`
	TargetProfileSlug string         `json:"target_profile_slug"`
	SceneDirection    string         `json:"scene_direction"`
	OutputPrompt      string         `json:"output_prompt"`
	Provider          AIProviderType `json:"provider"`
	ModelName         string         `json:"model_name"`
	DurationSec       float64        `json:"duration_sec"`
	CreatedAt         time.Time      `json:"created_at"`
}
