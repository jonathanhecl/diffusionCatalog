package domain

import "time"

// AppSettings stores configurable application options and Ollama model routing preferences.
type AppSettings struct {
	OllamaBaseURL          string `json:"ollama_base_url"`
	VisionModel            string `json:"vision_model"`
	Image2PromptModel      string `json:"image2prompt_model"`
	Text2PromptModel       string `json:"text2prompt_model"`
	DefaultTargetGenerator string `json:"default_target_generator"`
	AutoCaptionOnScan      bool   `json:"auto_caption_on_scan"`
}

// PromptProfile defines a target diffusion or video generator (e.g., flux, krea2, moss, minimaxh3)
// and the dedicated system prompt used by Ollama for image2prompt and text2prompt generation.
type PromptProfile struct {
	ID             int64     `json:"id"`
	Slug           string    `json:"slug"`
	Name           string    `json:"name"`
	TargetModality MediaType `json:"target_modality"` // "image" or "video"
	Description    string    `json:"description"`
	SystemPrompt   string    `json:"system_prompt"`
	UpdatedAt      time.Time `json:"updated_at"`
}

// OllamaModelInfo represents an installed Ollama model and its detected capabilities.
type OllamaModelInfo struct {
	Name          string   `json:"name"`
	ParameterSize string   `json:"parameter_size"`
	Family        string   `json:"family"`
	Capabilities  []string `json:"capabilities"`
	HasVision     bool     `json:"has_vision"`
	VisionStatus  string   `json:"vision_status"` // "confirmed", "inferred", or "none"
	ContextLength int      `json:"context_length"`
}
