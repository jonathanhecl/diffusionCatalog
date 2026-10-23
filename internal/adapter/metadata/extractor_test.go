package metadata

import (
	"bytes"
	"context"
	"encoding/binary"
	"hash/crc32"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"
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

func TestExtractor_TXTFirstAndAutoCreateSidecar(t *testing.T) {
	tmpDir := t.TempDir()
	imgPath := filepath.Join(tmpDir, "cyber_scene.png")
	txtPath := filepath.Join(tmpDir, "cyber_scene.txt")

	// 1. Write PNG with embedded parameters chunk and NO .txt file yet
	embedded := `cinematic netrunner on rainy balcony, <lora:flux_anamorphic:0.8>
Negative prompt: blurry, lowres
Steps: 25, Sampler: Euler, CFG scale: 3.5, Seed: 777111, Model: flux1-dev`
	writeTestPNGWithChunk(t, imgPath, 320, 480, "parameters", embedded)

	ext := NewExtractor()
	params, w, h, err := ext.Extract(context.Background(), imgPath, domain.MediaTypeImage)
	if err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if w != 320 || h != 480 {
		t.Errorf("expected 320x480, got %dx%d", w, h)
	}
	if !strings.Contains(params.PositivePrompt, "cinematic netrunner") {
		t.Errorf("unexpected positive prompt: %q", params.PositivePrompt)
	}

	// Verify that cyber_scene.txt was automatically created
	sidecarBytes, err := os.ReadFile(txtPath)
	if err != nil {
		t.Fatalf("expected auto-created sidecar .txt at %s: %v", txtPath, err)
	}
	if !strings.Contains(string(sidecarBytes), "cinematic netrunner") {
		t.Errorf("expected sidecar .txt to contain prompt, got %q", string(sidecarBytes))
	}

	// 2. Modify cyber_scene.txt and verify .txt takes priority over embedded PNG chunk
	overrideTXT := `overridden prompt from companion txt sidecar, <lora:krea2_skin:0.7>
Negative prompt: plastic skin
Steps: 30, Sampler: DPM++ 2M, CFG scale: 5.0, Seed: 999888, Model: krea2-pro, Engine: comfyui`
	if err := os.WriteFile(txtPath, []byte(overrideTXT), 0644); err != nil {
		t.Fatalf("failed to write override txt: %v", err)
	}

	params2, w2, h2, err := ext.Extract(context.Background(), imgPath, domain.MediaTypeImage)
	if err != nil {
		t.Fatalf("Extract from .txt failed: %v", err)
	}
	if w2 != 320 || h2 != 480 {
		t.Errorf("expected 320x480 on .txt fast path, got %dx%d", w2, h2)
	}
	if !strings.Contains(params2.PositivePrompt, "overridden prompt from companion txt sidecar") {
		t.Errorf("expected .txt prompt priority, got %q", params2.PositivePrompt)
	}
	if params2.Engine != domain.EngineComfyUI {
		t.Errorf("expected engine comfyui from .txt, got %s", params2.Engine)
	}
}

func writeTestPNGWithChunk(t *testing.T, path string, w, h int, key, val string) {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	img.Set(0, 0, color.RGBA{10, 20, 30, 255})

	var raw bytes.Buffer
	if err := png.Encode(&raw, img); err != nil {
		t.Fatalf("png encode: %v", err)
	}
	data := raw.Bytes()
	iend := len(data) - 12

	var payload bytes.Buffer
	payload.WriteString(key)
	payload.WriteByte(0)
	payload.WriteString(val)
	pBytes := payload.Bytes()

	var chunk bytes.Buffer
	_ = binary.Write(&chunk, binary.BigEndian, uint32(len(pBytes)))
	chunk.WriteString("tEXt")
	chunk.Write(pBytes)
	crc := crc32.NewIEEE()
	crc.Write([]byte("tEXt"))
	crc.Write(pBytes)
	_ = binary.Write(&chunk, binary.BigEndian, crc.Sum32())

	var out bytes.Buffer
	out.Write(data[:iend])
	out.Write(chunk.Bytes())
	out.Write(data[iend:])

	if err := os.WriteFile(path, out.Bytes(), 0644); err != nil {
		t.Fatalf("write png: %v", err)
	}
}
