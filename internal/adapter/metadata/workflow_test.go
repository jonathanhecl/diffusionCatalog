package metadata

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
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

func buildSyntheticMP4(w, h int, tags [][2]string) []byte {
	tkhdData := make([]byte, 84)
	binary.BigEndian.PutUint32(tkhdData[76:80], uint32(w)<<16)
	binary.BigEndian.PutUint32(tkhdData[80:84], uint32(h)<<16)
	tkhdData = append(tkhdData, 0, 0, 0, 0, 0, 0, 0, 0)
	// Put width/height in last 8 bytes of tkhdData
	tkhd := make([]byte, 92)
	binary.BigEndian.PutUint32(tkhd[84:88], uint32(w)<<16)
	binary.BigEndian.PutUint32(tkhd[88:92], uint32(h)<<16)
	trakBox := encodeBox("trak", encodeBox("tkhd", tkhd))

	var keysPayload bytes.Buffer
	_ = binary.Write(&keysPayload, binary.BigEndian, uint32(0)) // version + flags
	_ = binary.Write(&keysPayload, binary.BigEndian, uint32(len(tags)))
	for _, kv := range tags {
		entrySize := uint32(8 + len(kv[0]))
		_ = binary.Write(&keysPayload, binary.BigEndian, entrySize)
		keysPayload.WriteString("mdta")
		keysPayload.WriteString(kv[0])
	}
	keysBox := encodeBox("keys", keysPayload.Bytes())

	var ilstPayload bytes.Buffer
	for i, kv := range tags {
		var dataPayload bytes.Buffer
		_ = binary.Write(&dataPayload, binary.BigEndian, uint32(1)) // UTF-8
		_ = binary.Write(&dataPayload, binary.BigEndian, uint32(0)) // locale
		dataPayload.WriteString(kv[1])
		dataBox := encodeBox("data", dataPayload.Bytes())

		itemBox := encodeBox("____", dataBox)
		binary.BigEndian.PutUint32(itemBox[4:8], uint32(i+1))
		ilstPayload.Write(itemBox)
	}
	ilstBox := encodeBox("ilst", ilstPayload.Bytes())

	var metaPayload bytes.Buffer
	_ = binary.Write(&metaPayload, binary.BigEndian, uint32(0))
	metaPayload.Write(keysBox)
	metaPayload.Write(ilstBox)
	metaBox := encodeBox("meta", metaPayload.Bytes())

	udtaBox := encodeBox("udta", metaBox)
	return encodeBox("moov", append(trakBox, udtaBox...))
}

func writeTestPNGWithChunks(t *testing.T, path string, w, h int, chunks [][2]string) {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	img.Set(0, 0, color.RGBA{10, 20, 30, 255})

	var raw bytes.Buffer
	if err := png.Encode(&raw, img); err != nil {
		t.Fatalf("png encode: %v", err)
	}
	data := raw.Bytes()
	iend := len(data) - 12

	var out bytes.Buffer
	out.Write(data[:iend])
	for _, kv := range chunks {
		var payload bytes.Buffer
		payload.WriteString(kv[0])
		payload.WriteByte(0)
		payload.WriteString(kv[1])
		pBytes := payload.Bytes()

		_ = binary.Write(&out, binary.BigEndian, uint32(len(pBytes)))
		out.WriteString("tEXt")
		out.Write(pBytes)
		crc := crc32.NewIEEE()
		crc.Write([]byte("tEXt"))
		crc.Write(pBytes)
		_ = binary.Write(&out, binary.BigEndian, crc.Sum32())
	}
	out.Write(data[iend:])

	if err := os.WriteFile(path, out.Bytes(), 0644); err != nil {
		t.Fatalf("write png: %v", err)
	}
}

func buildTIFFTwoStrings(tag1 uint16, val1 string, tag2 uint16, val2 string) []byte {
	var buf bytes.Buffer
	bo := binary.LittleEndian

	buf.WriteString("II")
	_ = binary.Write(&buf, bo, uint16(42))
	_ = binary.Write(&buf, bo, uint32(8))
	_ = binary.Write(&buf, bo, uint16(2)) // 2 entries

	b1 := append([]byte(val1), 0)
	b2 := append([]byte(val2), 0)
	off1 := uint32(8 + 2 + 2*12 + 4)
	off2 := off1 + uint32(len(b1))

	_ = binary.Write(&buf, bo, tag1)
	_ = binary.Write(&buf, bo, uint16(2)) // ASCII
	_ = binary.Write(&buf, bo, uint32(len(b1)))
	_ = binary.Write(&buf, bo, off1)

	_ = binary.Write(&buf, bo, tag2)
	_ = binary.Write(&buf, bo, uint16(2)) // ASCII
	_ = binary.Write(&buf, bo, uint32(len(b2)))
	_ = binary.Write(&buf, bo, off2)

	_ = binary.Write(&buf, bo, uint32(0)) // next IFD
	buf.Write(b1)
	buf.Write(b2)
	return buf.Bytes()
}

func TestMP4ComfyUIWorkflowAndPromptExtraction(t *testing.T) {
	tmpDir := t.TempDir()
	mp4Path := filepath.Join(tmpDir, "comfy_video.mp4")

	workflowJSON := `{"id":"wf-1","nodes":[{"id":1,"type":"UNETLoader"},{"id":2,"type":"SaveVideo"}],"links":[[1,1,0,2,0,"VIDEO"]]}`
	promptJSON := `{"1":{"class_type":"UNETLoader","inputs":{"unet_name":"wan22_fp8.safetensors"}},"2":{"class_type":"CLIPTextEncode","inputs":{"text":"A neon train speeding through misty mountains"}},"3":{"class_type":"KSampler","inputs":{"seed":424242,"steps":25,"cfg":4.5,"sampler_name":"euler","scheduler":"simple","positive":["2",0],"negative":["4",0]}},"4":{"class_type":"CLIPTextEncode","inputs":{"text":"blurry, low quality, worst quality, deformed"}}}`

	mp4Data := buildSyntheticMP4(1280, 720, [][2]string{
		{"workflow", workflowJSON},
		{"prompt", promptJSON},
	})
	if err := os.WriteFile(mp4Path, mp4Data, 0644); err != nil {
		t.Fatalf("write mp4: %v", err)
	}

	ext := NewExtractor()
	params, w, h, err := ext.Extract(context.Background(), mp4Path, domain.MediaTypeVideo)
	if err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if w != 1280 || h != 720 {
		t.Errorf("expected 1280x720, got %dx%d", w, h)
	}
	if params.Engine != domain.EngineComfyUI {
		t.Errorf("expected EngineComfyUI, got %s", params.Engine)
	}
	if params.PositivePrompt != "A neon train speeding through misty mountains" {
		t.Errorf("unexpected positive prompt: %q", params.PositivePrompt)
	}
	if !strings.Contains(params.WorkflowJSON, `"nodes"`) {
		t.Errorf("expected WorkflowJSON to preserve full UI workflow, got %q", params.WorkflowJSON)
	}

	wf, ok := ExtractWorkflowFromFile(mp4Path, domain.MediaTypeVideo)
	if !ok {
		t.Fatalf("expected ExtractWorkflowFromFile to succeed")
	}
	if wf.Kind != "comfyui_workflow" {
		t.Errorf("expected Kind comfyui_workflow, got %q", wf.Kind)
	}
	if wf.NodeCount != 2 {
		t.Errorf("expected NodeCount 2, got %d", wf.NodeCount)
	}
	if !json.Valid([]byte(wf.JSON)) {
		t.Errorf("expected valid formatted JSON")
	}
	if wf.FileName != "comfy_video_workflow.json" {
		t.Errorf("expected FileName comfy_video_workflow.json, got %q", wf.FileName)
	}
}

func TestMP4VHSVideoCombineCommentExtraction(t *testing.T) {
	tmpDir := t.TempDir()
	mp4Path := filepath.Join(tmpDir, "vhs_video.mp4")

	// Mirrors BZW6P8EH46RQAX51ME7W2EGPT0.mp4: VHS_VideoCombine puts stringified JSON inside comment{"prompt": "..."}
	innerGraph := `{
		"2": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["63", 0], "guider": ["6", 0], "sampler": ["30", 0], "sigmas": ["75", 0]}},
		"6": {"class_type": "BasicGuider", "inputs": {"model": ["81", 0], "conditioning": ["7", 0]}},
		"7": {"class_type": "CLIPTextEncode", "inputs": {"text": ["9", 0]}},
		"9": {"class_type": "OpenSeaArt_Chat", "inputs": {"prompt1": ["79", 0]}},
		"30": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
		"63": {"class_type": "RandomNoise", "inputs": {"noise_seed": 435061356817856}},
		"75": {"class_type": "BasicScheduler", "inputs": {"scheduler": "simple", "steps": 12}},
		"79": {"class_type": "Primitive string multiline [Crystools]", "inputs": {"string": "Direction: dynamic cyberpunk funk dance under neon lights"}},
		"81": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_unet.safetensors"}},
		"83": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "dance_style_lora.safetensors"}}
	}`
	commentEnvelope, _ := json.Marshal(map[string]string{"prompt": innerGraph})

	mp4Data := buildSyntheticMP4(768, 1024, [][2]string{
		{"comment", string(commentEnvelope)},
	})
	if err := os.WriteFile(mp4Path, mp4Data, 0644); err != nil {
		t.Fatalf("write mp4: %v", err)
	}

	ext := NewExtractor()
	params, _, _, err := ext.Extract(context.Background(), mp4Path, domain.MediaTypeVideo)
	if err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if params.Engine != domain.EngineComfyUI {
		t.Errorf("expected EngineComfyUI, got %s", params.Engine)
	}
	if strings.HasPrefix(strings.TrimSpace(params.PositivePrompt), "{") {
		t.Fatalf("expected human prompt text, not raw JSON dump: %q", params.PositivePrompt)
	}
	if !strings.Contains(params.PositivePrompt, "cyberpunk funk dance") {
		t.Errorf("expected prompt to contain extracted direction text, got %q", params.PositivePrompt)
	}
	if params.Seed != 435061356817856 {
		t.Errorf("expected Seed 435061356817856 from RandomNoise, got %d", params.Seed)
	}
	if params.ModelName != "minimax_h3_unet.safetensors" {
		t.Errorf("expected UNET model minimax_h3_unet.safetensors, got %q", params.ModelName)
	}
	if len(params.LoRAs) != 1 || params.LoRAs[0] != "dance_style_lora.safetensors" {
		t.Errorf("expected LoRA dance_style_lora.safetensors, got %v", params.LoRAs)
	}

	wf, ok := ExtractWorkflowFromFile(mp4Path, domain.MediaTypeVideo)
	if !ok {
		t.Fatalf("expected ExtractWorkflowFromFile to unwrap VHS comment prompt")
	}
	if wf.Kind != "comfyui_prompt" {
		t.Errorf("expected Kind comfyui_prompt, got %q", wf.Kind)
	}
	if wf.NodeCount != 10 {
		t.Errorf("expected 10 unwrapped nodes, got %d", wf.NodeCount)
	}
}

func TestMP4SwarmUICommentExtraction(t *testing.T) {
	tmpDir := t.TempDir()
	mp4Path := filepath.Join(tmpDir, "swarm_video.mp4")

	commentJSON := `{
		"sui_image_params": {
			"prompt": "subject_definitions: <Subject 1> stands on a rainy rooftop at dusk",
			"negativeprompt": "blurry, watermark",
			"model": "PRISM_DMAD_PDMD_w6a8_g32.safetensors",
			"seed": 77112233,
			"steps": 28,
			"cfgscale": 5.5,
			"sampler": "res_multistep",
			"scheduler": "simple",
			"loras": ["cinematic_rain_v1.safetensors", "film_grain"]
		},
		"sui_models": [
			{"name": "PRISM_DMAD_PDMD_w6a8_g32.safetensors", "param": "model"}
		]
	}`

	mp4Data := buildSyntheticMP4(704, 1280, [][2]string{
		{"comment", commentJSON},
	})
	if err := os.WriteFile(mp4Path, mp4Data, 0644); err != nil {
		t.Fatalf("write mp4: %v", err)
	}

	ext := NewExtractor()
	params, _, _, err := ext.Extract(context.Background(), mp4Path, domain.MediaTypeVideo)
	if err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if params.Engine != domain.EngineSwarmUI {
		t.Errorf("expected EngineSwarmUI, got %s", params.Engine)
	}
	if !strings.Contains(params.PositivePrompt, "rainy rooftop at dusk") {
		t.Errorf("unexpected positive prompt: %q", params.PositivePrompt)
	}
	if params.NegativePrompt != "blurry, watermark" {
		t.Errorf("unexpected negative prompt: %q", params.NegativePrompt)
	}
	if params.ModelName != "PRISM_DMAD_PDMD_w6a8_g32.safetensors" {
		t.Errorf("unexpected model name: %q", params.ModelName)
	}
	if params.Steps != 28 || params.CFGScale != 5.5 || params.Seed != 77112233 {
		t.Errorf("unexpected steps/cfg/seed: %d / %v / %d", params.Steps, params.CFGScale, params.Seed)
	}
	if params.Sampler != "res_multistep" || params.Scheduler != "simple" {
		t.Errorf("unexpected sampler/scheduler: %q / %q", params.Sampler, params.Scheduler)
	}
	if len(params.LoRAs) != 2 {
		t.Errorf("expected 2 LoRAs, got %v", params.LoRAs)
	}

	wf, ok := ExtractWorkflowFromFile(mp4Path, domain.MediaTypeVideo)
	if !ok {
		t.Fatalf("expected ExtractWorkflowFromFile to detect SwarmUI metadata")
	}
	if wf.Kind != "swarmui" {
		t.Errorf("expected Kind swarmui, got %q", wf.Kind)
	}
}

func TestMP4MiniMaxBasicGuiderRandomNoiseExtraction(t *testing.T) {
	tmpDir := t.TempDir()
	mp4Path := filepath.Join(tmpDir, "minimax_h3.mp4")

	promptJSON := `{
		"UNETLoader": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3.safetensors"}},
		"MiniMaxH3ImageToVideo": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"prompt": "Young backpackers scramble up desert banks away from a flash flood.", "clip": ["CLIPLoader", 0]}},
		"RandomNoise": {"class_type": "RandomNoise", "inputs": {"noise_seed": 1458661632}},
		"BasicGuider": {"class_type": "BasicGuider", "inputs": {"conditioning": ["MiniMaxH3ImageToVideo", 0]}},
		"KSamplerSelect": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
		"BasicScheduler": {"class_type": "BasicScheduler", "inputs": {"scheduler": "simple", "steps": 30}},
		"SamplerCustomAdvanced": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["RandomNoise", 0], "guider": ["BasicGuider", 0]}}
	}`

	mp4Data := buildSyntheticMP4(704, 704, [][2]string{
		{"prompt", promptJSON},
	})
	if err := os.WriteFile(mp4Path, mp4Data, 0644); err != nil {
		t.Fatalf("write mp4: %v", err)
	}

	ext := NewExtractor()
	params, _, _, err := ext.Extract(context.Background(), mp4Path, domain.MediaTypeVideo)
	if err != nil {
		t.Fatalf("Extract failed: %v", err)
	}
	if params.PositivePrompt != "Young backpackers scramble up desert banks away from a flash flood." {
		t.Errorf("unexpected positive prompt: %q", params.PositivePrompt)
	}
	if params.Seed != 1458661632 {
		t.Errorf("expected noise_seed 1458661632, got %d", params.Seed)
	}
	if params.Steps != 30 || params.Sampler != "res_multistep" || params.Scheduler != "simple" {
		t.Errorf("unexpected steps/sampler/scheduler: %d / %q / %q", params.Steps, params.Sampler, params.Scheduler)
	}
	if !strings.Contains(params.WorkflowJSON, `"MiniMaxH3ImageToVideo"`) {
		t.Errorf("expected WorkflowJSON to preserve MiniMaxH3 graph")
	}
}

func TestMP4WithoutMetadataHasNoWorkflow(t *testing.T) {
	tmpDir := t.TempDir()
	mp4Path := filepath.Join(tmpDir, "stripped.mp4")

	mp4Data := buildSyntheticMP4(768, 1024, [][2]string{
		{"encoder", "Lavf62.12.102"},
	})
	if err := os.WriteFile(mp4Path, mp4Data, 0644); err != nil {
		t.Fatalf("write mp4: %v", err)
	}

	if wf, ok := ExtractWorkflowFromFile(mp4Path, domain.MediaTypeVideo); ok {
		t.Fatalf("expected ok=false for stripped MP4, got %+v", wf)
	}
}

func TestPNGWorkflowPreservesLiteralInfinityString(t *testing.T) {
	tmpDir := t.TempDir()
	pngPath := filepath.Join(tmpDir, "infinity_prompt.png")

	promptJSON := `{"1":{"class_type":"CLIPTextEncode","inputs":{"text":"A portal opening to Infinity and beyond"}},"2":{"class_type":"KSampler","inputs":{"positive":["1",0],"negative":["3",0]}},"3":{"class_type":"CLIPTextEncode","inputs":{"text":"blurry, worst quality, low quality, deformed"}}}`
	workflowJSON := `{"nodes":[{"id":1,"type":"CLIPTextEncode","widgets_values":["To Infinity and beyond"]},{"id":2,"type":"KSampler"}]}`

	writeTestPNGWithChunks(t, pngPath, 32, 32, [][2]string{
		{"prompt", promptJSON},
		{"workflow", workflowJSON},
	})

	wf, ok := ExtractWorkflowFromFile(pngPath, domain.MediaTypeImage)
	if !ok {
		t.Fatalf("expected PNG workflow extraction to succeed")
	}
	if wf.Kind != "comfyui_workflow" || wf.NodeCount != 2 {
		t.Errorf("expected comfyui_workflow with 2 nodes, got %+v", wf)
	}
	if !strings.Contains(wf.JSON, "To Infinity and beyond") {
		t.Errorf("expected literal word 'Infinity' inside string to remain uncorrupted, got %s", wf.JSON)
	}
}

func TestJPEGAndWebPEXIFWorkflowLocalization(t *testing.T) {
	tmpDir := t.TempDir()
	promptTag := `Prompt:{"1":{"class_type":"CLIPTextEncode","inputs":{"text":"Sunset over glass dunes"}},"2":{"class_type":"KSampler","inputs":{"positive":["1",0],"negative":["3",0]}},"3":{"class_type":"CLIPTextEncode","inputs":{"text":"blurry, low quality, worst quality"}}}`
	workflowTag := `Workflow:{"nodes":[{"id":1,"type":"CLIPTextEncode"},{"id":2,"type":"KSampler"},{"id":3,"type":"SaveImage"}],"links":[]}`

	tiff := buildTIFFTwoStrings(0x010E, promptTag, 0x010F, workflowTag)

	// 1. JPEG
	jpgPath := filepath.Join(tmpDir, "comfy.jpg")
	img := image.NewRGBA(image.Rect(0, 0, 16, 16))
	var jpgBuf bytes.Buffer
	if err := jpeg.Encode(&jpgBuf, img, nil); err != nil {
		t.Fatalf("jpeg encode: %v", err)
	}
	rawJPG := jpgBuf.Bytes()
	app1Payload := append([]byte("Exif\x00\x00"), tiff...)
	var app1 bytes.Buffer
	app1.WriteByte(0xFF)
	app1.WriteByte(0xE1)
	_ = binary.Write(&app1, binary.BigEndian, uint16(len(app1Payload)+2))
	app1.Write(app1Payload)

	var finalJPG bytes.Buffer
	finalJPG.Write(rawJPG[:2])
	finalJPG.Write(app1.Bytes())
	finalJPG.Write(rawJPG[2:])
	if err := os.WriteFile(jpgPath, finalJPG.Bytes(), 0644); err != nil {
		t.Fatalf("write jpg: %v", err)
	}

	wfJPG, ok := ExtractWorkflowFromFile(jpgPath, domain.MediaTypeImage)
	if !ok || wfJPG.Kind != "comfyui_workflow" || wfJPG.NodeCount != 3 {
		t.Fatalf("expected JPEG workflow with 3 nodes, got ok=%v wf=%+v", ok, wfJPG)
	}

	// 2. WebP (RIFF WEBP + EXIF chunk)
	webpPath := filepath.Join(tmpDir, "comfy.webp")
	exifPayload := append([]byte("Exif\x00\x00"), tiff...)
	var riff bytes.Buffer
	riff.WriteString("RIFF")
	_ = binary.Write(&riff, binary.LittleEndian, uint32(4+8+len(exifPayload)+(len(exifPayload)&1)))
	riff.WriteString("WEBP")
	riff.WriteString("EXIF")
	_ = binary.Write(&riff, binary.LittleEndian, uint32(len(exifPayload)))
	riff.Write(exifPayload)
	if len(exifPayload)&1 == 1 {
		riff.WriteByte(0)
	}
	if err := os.WriteFile(webpPath, riff.Bytes(), 0644); err != nil {
		t.Fatalf("write webp: %v", err)
	}

	wfWebP, ok := ExtractWorkflowFromFile(webpPath, domain.MediaTypeImage)
	if !ok || wfWebP.Kind != "comfyui_workflow" || wfWebP.NodeCount != 3 {
		t.Fatalf("expected WebP workflow with 3 nodes, got ok=%v wf=%+v", ok, wfWebP)
	}
}
