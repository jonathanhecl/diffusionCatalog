package domain

import (
	"math"
	"strings"
	"time"
)

// KnowledgeSnippet represents a modular, engine-agnostic context/knowledge document
// (e.g., lighting physics, lens choices, anatomy guidance, composition rules, style directives)
// that can be injected alongside system prompts to guide LLMs during prompt synthesis.
type KnowledgeSnippet struct {
	ID          int64     `json:"id"`
	Slug        string    `json:"slug"`
	Title       string    `json:"title"`
	Description string    `json:"description"`
	Content     string    `json:"content"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

// TokenCount returns an estimated token count for this snippet's content.
func (k *KnowledgeSnippet) TokenCount() int {
	return EstimateTokens(k.Content)
}

// EstimateTokens provides a fast, lightweight approximation of LLM token count (~3.8 to 4 chars/token).
func EstimateTokens(text string) int {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return 0
	}
	tokens := int(math.Ceil(float64(len(trimmed)) / 3.8))
	if tokens < 1 {
		return 1
	}
	return tokens
}
