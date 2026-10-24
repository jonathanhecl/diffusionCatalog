package metadata

import (
	"bytes"
	"compress/zlib"
	"context"
	"encoding/binary"
	"hash/crc32"
	"image"
	"image/color"
	"image/jpeg"
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
				"scheduler": "simple",
				"positive": ["6", 0],
				"negative": ["7", 0]
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

func TestComfyUIRecursiveTraversal(t *testing.T) {
	// KSampler connects to positive prompt via Primitive / Reroute intermediate node
	graph := `{
		"1": {
			"class_type": "KSampler",
			"inputs": {
				"seed": 123456,
				"positive": ["2", 0],
				"negative": ["3", 0]
			}
		},
		"2": {
			"class_type": "Reroute",
			"inputs": {
				"pipe": ["10", 0]
			}
		},
		"10": {
			"class_type": "CLIPTextEncode",
			"inputs": {
				"text": "hyperrealistic portrait of a robotic astronaut on mars"
			}
		},
		"3": {
			"class_type": "CLIPTextEncode",
			"inputs": {
				"text": "blurry, low quality, watermark, bad anatomy"
			}
		}
	}`

	params := ExtractComfyGraphParams(graph, "")
	if !strings.Contains(params.PositivePrompt, "robotic astronaut on mars") {
		t.Errorf("expected recursive prompt extraction to find astronaut prompt, got %q", params.PositivePrompt)
	}
	if !strings.Contains(params.NegativePrompt, "watermark") {
		t.Errorf("expected negative prompt extraction, got %q", params.NegativePrompt)
	}
}

func TestPNG_ZTXTChunk(t *testing.T) {
	tmpDir := t.TempDir()
	imgPath := filepath.Join(tmpDir, "test_ztxt.png")

	promptText := "cyberpunk city street at night in rain\nSteps: 20, Sampler: Euler, Seed: 100"
	writeTestPNGWithZTXT(t, imgPath, 64, 64, "parameters", promptText)

	ext := NewExtractor()
	params, w, h, err := ext.Extract(context.Background(), imgPath, domain.MediaTypeImage)
	if err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if w != 64 || h != 64 {
		t.Errorf("expected 64x64, got %dx%d", w, h)
	}
	if !strings.Contains(params.PositivePrompt, "cyberpunk city street") {
		t.Errorf("expected prompt from zTXt chunk, got: %q", params.PositivePrompt)
	}
}

func TestPNG_ExtraMetadata(t *testing.T) {
	tmpDir := t.TempDir()
	imgPath := filepath.Join(tmpDir, "test_extra_meta.png")

	extraJSON := `{"prompt": "portrait of ancient warrior in ornate golden armor", "steps": 30}`
	writeTestPNGWithChunk(t, imgPath, 64, 64, "extraMetadata", extraJSON)

	ext := NewExtractor()
	params, _, _, err := ext.Extract(context.Background(), imgPath, domain.MediaTypeImage)
	if err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if !strings.Contains(params.PositivePrompt, "portrait of ancient warrior") {
		t.Errorf("expected prompt from extraMetadata, got: %q", params.PositivePrompt)
	}
}

func TestJPEG_EXIFUserComment(t *testing.T) {
	tmpDir := t.TempDir()
	imgPath := filepath.Join(tmpDir, "test_exif.jpg")

	promptText := "majestic eagle flying over snow-covered pine mountains\nNegative prompt: blurry, deformed\nSteps: 30, Sampler: DPM++ 2M, CFG scale: 7.0, Seed: 555444"
	writeTestJPEGWithEXIFUserComment(t, imgPath, 128, 128, promptText)

	ext := NewExtractor()
	params, w, h, err := ext.Extract(context.Background(), imgPath, domain.MediaTypeImage)
	if err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if w != 128 || h != 128 {
		t.Errorf("expected 128x128, got %dx%d", w, h)
	}
	if !strings.Contains(params.PositivePrompt, "majestic eagle flying") {
		t.Errorf("expected prompt from JPEG EXIF UserComment, got: %q", params.PositivePrompt)
	}
}

func TestWebP_EXIFChunk(t *testing.T) {
	tmpDir := t.TempDir()
	imgPath := filepath.Join(tmpDir, "test_riff.webp")

	promptText := "ancient library filled with glowing magical manuscripts\nSteps: 25, Seed: 987654"
	writeTestWebPWithEXIF(t, imgPath, promptText)

	ext := NewExtractor()
	params, _, _, err := ext.Extract(context.Background(), imgPath, domain.MediaTypeImage)
	if err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if !strings.Contains(params.PositivePrompt, "ancient library filled with glowing magical") {
		t.Errorf("expected prompt from WebP EXIF chunk, got: %q", params.PositivePrompt)
	}
}

func TestExtractor_TXTFirstAndAutoCreateSidecar(t *testing.T) {
	tmpDir := t.TempDir()
	imgPath := filepath.Join(tmpDir, "cyber_scene.png")
	txtPath := filepath.Join(tmpDir, "cyber_scene.txt")

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

	sidecarBytes, err := os.ReadFile(txtPath)
	if err != nil {
		t.Fatalf("expected auto-created sidecar .txt at %s: %v", txtPath, err)
	}
	if !strings.Contains(string(sidecarBytes), "cinematic netrunner") {
		t.Errorf("expected sidecar .txt to contain prompt, got %q", string(sidecarBytes))
	}

	overrideTXT := `overridden prompt from companion txt sidecar, <lora:krea2_skin:0.7>
Negative prompt: plastic skin
Steps: 30, Sampler: DPM++ 2M, CFG scale: 5.0, Seed: 999888, Model: krea2-pro, Engine: comfyui`
	if err := os.WriteFile(txtPath, []byte(overrideTXT), 0644); err != nil {
		t.Fatalf("failed to write override txt: %v", err)
	}

	_, _, _, err = ext.Extract(context.Background(), imgPath, domain.MediaTypeImage)
	if err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	txtContentAfter, err := os.ReadFile(txtPath)
	if err != nil {
		t.Fatalf("failed to read txt: %v", err)
	}
	if string(txtContentAfter) != overrideTXT {
		t.Errorf("expected existing (media).txt to NOT be overwritten, got: %q", string(txtContentAfter))
	}

	noMetaImgPath := filepath.Join(tmpDir, "plain_scene.png")
	noMetaTxtPath := filepath.Join(tmpDir, "plain_scene.txt")
	writeTestPNGWithChunk(t, noMetaImgPath, 320, 480, "Software", "Adobe Photoshop")
	plainTXT := "a sunset over mountains, 8k photo\nNegative prompt: blurry\nSteps: 20, Model: sd_xl_base"
	if err := os.WriteFile(noMetaTxtPath, []byte(plainTXT), 0644); err != nil {
		t.Fatalf("failed to write plain txt: %v", err)
	}

	plainParams, _, _, err := ext.Extract(context.Background(), noMetaImgPath, domain.MediaTypeImage)
	if err != nil {
		t.Fatalf("Extract from no-meta image failed: %v", err)
	}
	if !strings.Contains(plainParams.PositivePrompt, "a sunset over mountains") {
		t.Errorf("expected prompt from (media).txt when not in metadata, got: %q", plainParams.PositivePrompt)
	}
}

// Helpers for test generation

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

func writeTestPNGWithZTXT(t *testing.T, path string, w, h int, key, val string) {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	img.Set(0, 0, color.RGBA{10, 20, 30, 255})

	var raw bytes.Buffer
	if err := png.Encode(&raw, img); err != nil {
		t.Fatalf("png encode: %v", err)
	}
	data := raw.Bytes()
	iend := len(data) - 12

	var zbuf bytes.Buffer
	zw := zlib.NewWriter(&zbuf)
	_, _ = zw.Write([]byte(val))
	_ = zw.Close()

	var payload bytes.Buffer
	payload.WriteString(key)
	payload.WriteByte(0)
	payload.WriteByte(0) // compression method
	payload.Write(zbuf.Bytes())
	pBytes := payload.Bytes()

	var chunk bytes.Buffer
	_ = binary.Write(&chunk, binary.BigEndian, uint32(len(pBytes)))
	chunk.WriteString("zTXt")
	chunk.Write(pBytes)
	crc := crc32.NewIEEE()
	crc.Write([]byte("zTXt"))
	crc.Write(pBytes)
	_ = binary.Write(&chunk, binary.BigEndian, crc.Sum32())

	var out bytes.Buffer
	out.Write(data[:iend])
	out.Write(chunk.Bytes())
	out.Write(data[iend:])

	if err := os.WriteFile(path, out.Bytes(), 0644); err != nil {
		t.Fatalf("write png ztxt: %v", err)
	}
}

func buildTIFFWithUserComment(comment string) []byte {
	var buf bytes.Buffer
	bo := binary.LittleEndian

	// Header: 'II', magic 42, offset to IFD0 = 8
	buf.WriteString("II")
	_ = binary.Write(&buf, bo, uint16(42))
	_ = binary.Write(&buf, bo, uint32(8))

	// IFD0 has 1 entry: Tag 0x9286 (UserComment), type 7 (UNDEFINED)
	numEntries := uint16(1)
	_ = binary.Write(&buf, bo, numEntries)

	// UserComment body: "UNICODE\x00" + comment in utf-8 or ascii
	commBytes := append([]byte("ASCII\x00\x00\x00"), []byte(comment)...)
	valOffset := uint32(8 + 2 + 12 + 4) // after header (8) + numEntries (2) + 1 entry (12) + nextIFD (4) = 26

	// Entry 1: tag=0x9286, type=7, count=len(commBytes), offset=valOffset
	_ = binary.Write(&buf, bo, uint16(0x9286))
	_ = binary.Write(&buf, bo, uint16(7))
	_ = binary.Write(&buf, bo, uint32(len(commBytes)))
	_ = binary.Write(&buf, bo, valOffset)

	// Next IFD = 0
	_ = binary.Write(&buf, bo, uint32(0))

	// Data
	buf.Write(commBytes)

	return buf.Bytes()
}

func writeTestJPEGWithEXIFUserComment(t *testing.T, path string, w, h int, comment string) {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	img.Set(0, 0, color.RGBA{50, 100, 150, 255})

	var jpgBuf bytes.Buffer
	if err := jpeg.Encode(&jpgBuf, img, nil); err != nil {
		t.Fatalf("jpeg encode: %v", err)
	}
	raw := jpgBuf.Bytes()

	// Build APP1 segment: 0xFF 0xE1 [len: 2 bytes] "Exif\x00\x00" [TIFF payload]
	tiffData := buildTIFFWithUserComment(comment)
	app1Payload := append([]byte("Exif\x00\x00"), tiffData...)
	segLen := uint16(len(app1Payload) + 2)

	var app1 bytes.Buffer
	app1.WriteByte(0xFF)
	app1.WriteByte(0xE1)
	_ = binary.Write(&app1, binary.BigEndian, segLen)
	app1.Write(app1Payload)

	// Insert APP1 after SOI (0xFF 0xD8)
	var finalJPG bytes.Buffer
	finalJPG.Write(raw[:2])
	finalJPG.Write(app1.Bytes())
	finalJPG.Write(raw[2:])

	if err := os.WriteFile(path, finalJPG.Bytes(), 0644); err != nil {
		t.Fatalf("write jpeg: %v", err)
	}
}

func writeTestWebPWithEXIF(t *testing.T, path string, comment string) {
	t.Helper()
	tiffData := buildTIFFWithUserComment(comment)
	exifPayload := append([]byte("Exif\x00\x00"), tiffData...)

	var chunk bytes.Buffer
	chunk.WriteString("EXIF")
	_ = binary.Write(&chunk, binary.LittleEndian, uint32(len(exifPayload)))
	chunk.Write(exifPayload)
	if len(exifPayload)%2 != 0 {
		chunk.WriteByte(0)
	}

	riffPayloadLen := uint32(4 + chunk.Len())
	var out bytes.Buffer
	out.WriteString("RIFF")
	_ = binary.Write(&out, binary.LittleEndian, riffPayloadLen)
	out.WriteString("WEBP")
	out.Write(chunk.Bytes())

	if err := os.WriteFile(path, out.Bytes(), 0644); err != nil {
		t.Fatalf("write webp: %v", err)
	}
}
