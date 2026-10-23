package domain

import "time"

// AIProviderType identifies which inference server handles a task ("ollama" or "omlx").
type AIProviderType string

const (
	ProviderOllama AIProviderType = "ollama"
	ProviderOMLX   AIProviderType = "omlx"
)

// AppSettings stores configurable application options and dual Ollama / oMLX model routing preferences.
type AppSettings struct {
	OllamaBaseURL          string         `json:"ollama_base_url"`
	OMLXBaseURL            string         `json:"omlx_base_url"`
	OMLXAPIKey             string         `json:"omlx_api_key"`
	VisionProvider         AIProviderType `json:"vision_provider"`
	VisionModel            string         `json:"vision_model"`
	Image2PromptProvider   AIProviderType `json:"image2prompt_provider"`
	Image2PromptModel      string         `json:"image2prompt_model"`
	Text2PromptProvider    AIProviderType `json:"text2prompt_provider"`
	Text2PromptModel       string         `json:"text2prompt_model"`
	DefaultTargetGenerator string         `json:"default_target_generator"`
	AutoCaptionOnScan      bool           `json:"auto_caption_on_scan"`
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

// OllamaModelInfo represents a model discovered from either an Ollama server (/api/tags + /api/show)
// or an oMLX OpenAI-compatible server (/v1/models).
type OllamaModelInfo struct {
	Provider      AIProviderType `json:"provider"` // "ollama" or "omlx"
	Name          string         `json:"name"`
	ParameterSize string         `json:"parameter_size"`
	Family        string         `json:"family"`
	Capabilities  []string       `json:"capabilities"`
	HasVision     bool           `json:"has_vision"`
	VisionStatus  string         `json:"vision_status"` // "confirmed", "inferred", or "none"
	ContextLength int            `json:"context_length"`
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
	CreatedAt         time.Time      `json:"created_at"`
}
