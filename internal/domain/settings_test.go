package domain_test

import (
	"testing"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

func TestOllamaModelInfo_IsEmbeddingVision(t *testing.T) {
	cases := []struct {
		name     string
		model    domain.OllamaModelInfo
		expected bool
	}{
		{
			name: "embeddinggemma-2 with vision and embedding capabilities",
			model: domain.OllamaModelInfo{
				Name:         "embeddinggemma-2:latest",
				Capabilities: []string{"embedding", "vision", "audio"},
				HasVision:    true,
				HasEmbedding: true,
			},
			expected: true,
		},
		{
			name: "clip model inferred from name",
			model: domain.OllamaModelInfo{
				Name:      "clip-vit-base-patch32:latest",
				HasVision: true,
			},
			expected: true,
		},
		{
			name: "text only embedding model",
			model: domain.OllamaModelInfo{
				Name:         "nomic-embed-text:latest",
				Capabilities: []string{"embedding"},
				HasEmbedding: true,
				HasVision:    false,
			},
			expected: false,
		},
		{
			name: "vision only completion LLM",
			model: domain.OllamaModelInfo{
				Name:         "LiquidAI/LFM3-Omni-2B-GGUF:Q4_K_M",
				Capabilities: []string{"tools", "thinking", "completion", "vision", "audio"},
				HasVision:    true,
				HasEmbedding: false,
			},
			expected: false,
		},
		{
			name: "text only LLM",
			model: domain.OllamaModelInfo{
				Name:         "AtomicChat/LFM3-1.7B-Uncensored-GGUF:Q4_K_M",
				Capabilities: []string{"tools", "thinking", "completion"},
				HasVision:    false,
				HasEmbedding: false,
			},
			expected: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.model.IsEmbeddingVision()
			if got != tc.expected {
				t.Fatalf("expected IsEmbeddingVision() = %v, got %v for model %+v", tc.expected, got, tc.model)
			}
		})
	}
}

func TestOllamaModelInfo_IsCompletionModel(t *testing.T) {
	cases := []struct {
		name     string
		model    domain.OllamaModelInfo
		expected bool
	}{
		{
			name: "standard LLM completion model",
			model: domain.OllamaModelInfo{
				Name:          "huggingface.co/grumd0/Nemotron-3.5-Bolt-E4B-Heretic-GGUF:Q4_K_M",
				Capabilities:  []string{"tools", "completion", "vision", "audio"},
				HasCompletion: true,
				HasVision:     true,
			},
			expected: true,
		},
		{
			name: "pure decision model",
			model: domain.OllamaModelInfo{
				Name:          "strands-decider:latest",
				Capabilities:  []string{"decision"},
				HasCompletion: false,
			},
			expected: false,
		},
		{
			name: "pure embedding model",
			model: domain.OllamaModelInfo{
				Name:          "embeddinggemma:latest",
				Capabilities:  []string{"embedding"},
				HasCompletion: false,
				HasEmbedding:  true,
			},
			expected: false,
		},
		{
			name: "multimodal vision embedding model without completion",
			model: domain.OllamaModelInfo{
				Name:          "embeddinggemma-2:latest",
				Capabilities:  []string{"embedding", "vision", "audio"},
				HasCompletion: false,
				HasVision:     true,
				HasEmbedding:  true,
			},
			expected: false,
		},
		{
			name: "speech model",
			model: domain.OllamaModelInfo{
				Name:          "phonon:40m-es",
				Capabilities:  []string{"speech"},
				HasCompletion: false,
			},
			expected: false,
		},
		{
			name: "model without capabilities inferred by name",
			model: domain.OllamaModelInfo{
				Name: "bge-m3-embedding",
			},
			expected: false,
		},
		{
			name: "standard text model without capabilities inferred as completion",
			model: domain.OllamaModelInfo{
				Name: "mistral:7b-instruct",
			},
			expected: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.model.IsCompletionModel()
			if got != tc.expected {
				t.Fatalf("expected IsCompletionModel() = %v, got %v for model %+v", tc.expected, got, tc.model)
			}
		})
	}
}
