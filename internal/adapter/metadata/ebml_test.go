package metadata

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

func TestParseEBMLFile_RealSample(t *testing.T) {
	samplePath := "/mnt/backup/DATASET/h3/GZ1CD0V8V9BTHXX69KC0FJ3NM0.webm"
	if _, err := os.Stat(samplePath); os.IsNotExist(err) {
		t.Skip("sample file not available on this machine")
	}

	tags, width, height, err := ParseEBMLFile(samplePath)
	if err != nil {
		t.Fatalf("ParseEBMLFile failed: %v", err)
	}

	if width != 800 || height != 1088 {
		t.Errorf("expected dimensions 800x1088, got %dx%d", width, height)
	}

	promptTag, hasPrompt := tags["PROMPT"]
	if !hasPrompt || len(promptTag) == 0 {
		t.Fatalf("expected PROMPT tag to be found, got tags: %v", tags)
	}

	workflowTag, hasWorkflow := tags["WORKFLOW"]
	if !hasWorkflow || len(workflowTag) == 0 {
		t.Fatalf("expected WORKFLOW tag to be found")
	}

	params := ExtractComfyGraphParams(promptTag, workflowTag)
	if params.PositivePrompt == "" {
		t.Fatalf("failed to extract PositivePrompt from PROMPT tag")
	}

	if !strings.HasPrefix(params.PositivePrompt, "For the target video") {
		t.Errorf("unexpected prompt content: %s", params.PositivePrompt[:min(len(params.PositivePrompt), 60)])
	}

	if params.Steps != 6 {
		t.Errorf("expected 6 steps, got %d", params.Steps)
	}

	if params.Sampler != "euler" {
		t.Errorf("expected sampler 'euler', got %s", params.Sampler)
	}

	if len(params.LoRAs) != 2 {
		t.Errorf("expected 2 LoRAs, got %v", params.LoRAs)
	}
}

func TestEBMLParser_Synthetic(t *testing.T) {
	// 1. EBML Header
	header := encodeElement(idEBMLHeader, []byte{0x42, 0x86, 0x81, 0x01})

	// 2. Tracks with 1920x1080 video
	pw := encodeElement(idPixelWidth, []byte{0x07, 0x80})  // 1920
	ph := encodeElement(idPixelHeight, []byte{0x04, 0x38}) // 1080
	vid := encodeElement(idVideo, append(pw, ph...))
	trackType := encodeElement(idTrackType, []byte{0x01}) // 1 = video
	trackEntry := encodeElement(idTrackEntry, append(trackType, vid...))
	tracks := encodeElement(idTracks, trackEntry)

	// 3. Tags with PROMPT
	promptJSON := `{"6":{"inputs":{"text":"cyberpunk futuristic street in neon rain"},"class_type":"CLIPTextEncode"}}`
	tagName := encodeElement(idTagName, []byte("PROMPT"))
	tagStr := encodeElement(idTagString, []byte(promptJSON))
	simpleTag := encodeElement(idSimpleTag, append(tagName, tagStr...))
	tag := encodeElement(idTag, simpleTag)
	tagsElem := encodeElement(idTags, tag)

	// 4. Segment
	segment := encodeElement(idSegment, append(tracks, tagsElem...))

	ebmlBytes := append(header, segment...)

	tags, w, h, err := ParseEBML(bytes.NewReader(ebmlBytes))
	if err != nil {
		t.Fatalf("ParseEBML synthetic failed: %v", err)
	}

	if w != 1920 || h != 1080 {
		t.Errorf("expected 1920x1080, got %dx%d", w, h)
	}

	if tags["PROMPT"] != promptJSON {
		t.Errorf("expected PROMPT %s, got %s", promptJSON, tags["PROMPT"])
	}
}

func TestExtractor_WebM_Integration(t *testing.T) {
	samplePath := "/mnt/backup/DATASET/h3/GZ1CD0V8V9BTHXX69KC0FJ3NM0.webm"
	if _, err := os.Stat(samplePath); os.IsNotExist(err) {
		t.Skip("sample file not available on this machine")
	}

	// Copy to a temp directory so we can test companion .txt sidecar generation
	tmpDir := t.TempDir()
	tempWebM := filepath.Join(tmpDir, "test_video.webm")
	data, err := os.ReadFile(samplePath)
	if err != nil {
		t.Fatalf("failed to read sample: %v", err)
	}
	if err := os.WriteFile(tempWebM, data, 0644); err != nil {
		t.Fatalf("failed to write temp file: %v", err)
	}

	extractor := NewExtractor()
	params, w, h, err := extractor.Extract(context.Background(), tempWebM, domain.MediaTypeVideo)
	if err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	if w != 800 || h != 1088 {
		t.Errorf("expected 800x1088, got %dx%d", w, h)
	}

	if !strings.HasPrefix(params.PositivePrompt, "For the target video") {
		t.Errorf("extracted prompt incorrect: %s", params.PositivePrompt[:min(len(params.PositivePrompt), 60)])
	}

	if params.Engine != domain.EngineComfyUI {
		t.Errorf("expected engine %s, got %s", domain.EngineComfyUI, params.Engine)
	}

	if params.Steps != 6 {
		t.Errorf("expected 6 steps, got %d", params.Steps)
	}

	// Verify companion .txt was written
	sidecarTXT := filepath.Join(tmpDir, "test_video.txt")
	txtData, err := os.ReadFile(sidecarTXT)
	if err != nil {
		t.Fatalf("expected companion .txt to be generated: %v", err)
	}
	txtContent := string(txtData)
	if !strings.Contains(txtContent, "For the target video") {
		t.Errorf("sidecar .txt does not contain prompt: %s", txtContent[:min(len(txtContent), 100)])
	}
	if !strings.Contains(txtContent, "Steps: 6") {
		t.Errorf("sidecar .txt does not contain Steps: 6: %s", txtContent)
	}
}

func encodeVINT(val uint64) []byte {
	if val < 0x7F {
		return []byte{byte(0x80 | val)}
	}
	if val < 0x3FFF {
		return []byte{byte(0x40 | (val >> 8)), byte(val & 0xFF)}
	}
	if val < 0x1FFFFF {
		return []byte{byte(0x20 | (val >> 16)), byte((val >> 8) & 0xFF), byte(val & 0xFF)}
	}
	return []byte{byte(0x10 | (val >> 24)), byte((val >> 16) & 0xFF), byte((val >> 8) & 0xFF), byte(val & 0xFF)}
}

func encodeElement(id uint64, data []byte) []byte {
	var idBytes []byte
	if id > 0xFFFFFF {
		idBytes = []byte{byte(id >> 24), byte(id >> 16), byte(id >> 8), byte(id)}
	} else if id > 0xFFFF {
		idBytes = []byte{byte(id >> 16), byte(id >> 8), byte(id)}
	} else if id > 0xFF {
		idBytes = []byte{byte(id >> 8), byte(id)}
	} else {
		idBytes = []byte{byte(id)}
	}
	sizeBytes := encodeVINT(uint64(len(data)))
	res := append(idBytes, sizeBytes...)
	return append(res, data...)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

