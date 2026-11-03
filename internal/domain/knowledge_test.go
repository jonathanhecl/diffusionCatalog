package domain

import (
	"testing"
)

func TestEstimateTokens(t *testing.T) {
	if got := EstimateTokens(""); got != 0 {
		t.Errorf("expected 0 for empty string, got %d", got)
	}
	if got := EstimateTokens("   "); got != 0 {
		t.Errorf("expected 0 for whitespace string, got %d", got)
	}

	shortText := "Hello world"
	if got := EstimateTokens(shortText); got < 1 || got > 5 {
		t.Errorf("expected ~3 tokens for %q, got %d", shortText, got)
	}

	longText := "You are an expert prompt engineer for Flux.1 diffusion models. Write rich, natural-language descriptive prose."
	tokens := EstimateTokens(longText)
	if tokens < 20 || tokens > 40 {
		t.Errorf("expected between 20 and 40 tokens for long text, got %d", tokens)
	}

	snip := KnowledgeSnippet{
		Title:   "Cinematography",
		Content: longText,
	}
	if snip.TokenCount() != tokens {
		t.Errorf("expected snippet.TokenCount() %d, got %d", tokens, snip.TokenCount())
	}
}
