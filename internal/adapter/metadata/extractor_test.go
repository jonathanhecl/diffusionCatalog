package metadata

import (
	"testing"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

func TestParseA1111Parameters(t *testing.T) {
	raw := `masterpiece, best quality, cinematic portrait of a cyberpunk archivist, neon rain, <lora:cyber_visor_v2:0.8>
Negative prompt: lowres, bad anatomy, blurry, watermark
Steps: 28, Sampler: DPM++ 2M Karras, Schedule type: Karras, CFG scale: 6.5, Seed: 4294967291, Size: 832x1216, Model hash: a1b2c3d4, Model: juggernautXL_v9`

	params := ParseA1111Parameters(raw)
	if params.Engine != domain.EngineA1111 {
		t.Fatalf("expected engine %s, got %s", domain.EngineA1111, params.Engine)
	}
	if params.Steps != 28 {
		t.Errorf("expected 28 steps, got %d", params.Steps)
	}
	if params.CFGScale != 6.5 {
		t.Errorf("expected CFG 6.5, got %v", params.CFGScale)
	}
	if params.Seed != 4294967291 {
		t.Errorf("expected seed 4294967291, got %d", params.Seed)
	}
	if params.ModelName != "juggernautXL_v9" {
		t.Errorf("expected model juggernautXL_v9, got %s", params.ModelName)
	}
	if len(params.LoRAs) != 1 || params.LoRAs[0] != "cyber_visor_v2" {
		t.Errorf("expected LoRA cyber_visor_v2, got %v", params.LoRAs)
	}
}

func TestParseComfyUIPromptGraph(t *testing.T) {
	graph := `{
		"4": {
			"class_type": "CheckpointLoaderSimple",
			"inputs": {"ckpt_name": "flux1-dev-fp8.safetensors"}
		},
		"6": {
			"class_type": "CLIPTextEncode",
			"inputs": {"text": "editorial studio photography, dramatic rim lighting, 85mm lens"}
		},
		"7": {
			"class_type": "CLIPTextEncode",
			"inputs": {"text": "worst quality, low quality, blurry"}
		},
		"3": {
			"class_type": "KSampler",
			"inputs": {
				"seed": 99887766,
				"steps": 24,
				"cfg": 3.5,
				"sampler_name": "euler",
				"scheduler": "simple"
			}
		}
	}`

	params := ParseComfyUIPromptGraph(graph, "")
	if params.Engine != domain.EngineComfyUI {
		t.Fatalf("expected engine %s, got %s", domain.EngineComfyUI, params.Engine)
	}
	if params.ModelName != "flux1-dev-fp8.safetensors" {
		t.Errorf("expected model flux1-dev-fp8.safetensors, got %s", params.ModelName)
	}
	if params.Steps != 24 {
		t.Errorf("expected 24 steps, got %d", params.Steps)
	}
	if params.Seed != 99887766 {
		t.Errorf("expected seed 99887766, got %d", params.Seed)
	}
	if params.PositivePrompt == "" || params.NegativePrompt == "" {
		t.Errorf("expected both positive and negative prompts to be populated, got pos=%q neg=%q", params.PositivePrompt, params.NegativePrompt)
	}
}
