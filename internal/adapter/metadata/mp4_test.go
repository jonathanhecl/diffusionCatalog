package metadata

import (
	"bytes"
	"context"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

func TestParseMP4File_RealSample(t *testing.T) {
	samplePath := "/mnt/backup/DATASET/h3/5KEHWT64MXCV0X3RARCT2KN9D0.mp4"
	if _, err := os.Stat(samplePath); os.IsNotExist(err) {
		t.Skip("sample file not available on this machine")
	}

	tags, width, height, err := ParseMP4File(samplePath)
	if err != nil {
		t.Fatalf("ParseMP4File failed: %v", err)
	}

	if width != 768 || height != 1024 {
		t.Errorf("expected dimensions 768x1024, got %dx%d", width, height)
	}

	paramsStr, hasParams := tags["parameters"]
	if !hasParams || len(paramsStr) == 0 {
		t.Fatalf("expected 'parameters' tag to be extracted, got tags: %v", tags)
	}

	// Verify the prompt starts from the beginning (NOT cut off at 'g hand waving quickly')
	if !strings.HasPrefix(paramsStr, "------ Score to Screen ------") {
		t.Errorf("expected prompt to start with '------ Score to Screen ------', got: %s", paramsStr[:min(len(paramsStr), 80)])
	}

	// Parse parameters with ParseA1111Parameters
	p := ParseA1111Parameters(paramsStr)
	if !strings.HasPrefix(p.PositivePrompt, "------ Score to Screen ------") {
		t.Errorf("expected positive prompt to start with '------ Score to Screen ------', got: %s", p.PositivePrompt[:min(len(p.PositivePrompt), 80)])
	}

	if p.Steps != 4 {
		t.Errorf("expected 4 steps, got %d", p.Steps)
	}

	if p.Sampler != "res_multistep_simple" {
		t.Errorf("expected sampler 'res_multistep_simple', got %s", p.Sampler)
	}

	if p.Seed != 601763488305293 {
		t.Errorf("expected seed 601763488305293, got %d", p.Seed)
	}

	if !strings.Contains(p.ModelName, "minimax_h3_ref2va_pruned_int8_convrot") {
		t.Errorf("expected model name to contain 'minimax_h3_ref2va_pruned_int8_convrot', got %s", p.ModelName)
	}

	// Also verify Extractor.Extract heals the real dataset file's sidecar .txt
	ext := NewExtractor()
	extMeta, _, _, err := ext.Extract(context.Background(), samplePath, domain.MediaTypeVideo)
	if err != nil {
		t.Fatalf("Extractor.Extract failed on real sample: %v", err)
	}
	if !strings.HasPrefix(extMeta.PositivePrompt, "------ Score to Screen ------") {
		t.Errorf("expected extMeta positive prompt to start with '------ Score to Screen ------', got: %s", extMeta.PositivePrompt[:min(len(extMeta.PositivePrompt), 80)])
	}
}

func TestExtractor_MP4_Integration(t *testing.T) {
	samplePath := "/mnt/backup/DATASET/h3/5KEHWT64MXCV0X3RARCT2KN9D0.mp4"
	if _, err := os.Stat(samplePath); os.IsNotExist(err) {
		t.Skip("sample file not available on this machine")
	}

	tmpDir := t.TempDir()
	tempMP4 := filepath.Join(tmpDir, "test_video.mp4")
	tempTXT := filepath.Join(tmpDir, "test_video.txt")

	data, err := os.ReadFile(samplePath)
	if err != nil {
		t.Fatalf("failed to read sample: %v", err)
	}
	if err := os.WriteFile(tempMP4, data, 0644); err != nil {
		t.Fatalf("failed to write temp file: %v", err)
	}

	// Write truncated .txt to simulate previous bug
	truncatedContent := "g hand moving quickly, mouth closed.\n\nAudio: <Audio 1> exactly as it is.\n\n[Scene 2]\n<Picture 1> shows the opening scene."
	if err := os.WriteFile(tempTXT, []byte(truncatedContent), 0644); err != nil {
		t.Fatalf("failed to write truncated txt: %v", err)
	}

	extractor := NewExtractor()
	params, w, h, err := extractor.Extract(context.Background(), tempMP4, domain.MediaTypeVideo)
	if err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	if w != 768 || h != 1024 {
		t.Errorf("expected 768x1024, got %dx%d", w, h)
	}

	if !strings.HasPrefix(params.PositivePrompt, "------ Score to Screen ------") {
		t.Errorf("expected prompt to start with '------ Score to Screen ------', got: %s", params.PositivePrompt[:min(len(params.PositivePrompt), 60)])
	}

	if params.Steps != 4 {
		t.Errorf("expected 4 steps, got %d", params.Steps)
	}

	if params.Engine != domain.EngineComfyUI {
		t.Errorf("expected engine %s, got %s", domain.EngineComfyUI, params.Engine)
	}

	// Verify that the truncated .txt was healed and now has the full prompt!
	healedTXT, err := os.ReadFile(tempTXT)
	if err != nil {
		t.Fatalf("failed to read healed .txt: %v", err)
	}
	if !strings.HasPrefix(string(healedTXT), "------ Score to Screen ------") {
		t.Errorf("expected healed .txt to start with '------ Score to Screen ------', got: %s", string(healedTXT)[:min(len(string(healedTXT)), 60)])
	}
}

func TestParseMP4_Synthetic(t *testing.T) {
	// 1. Synthesize tkhd with 1920x1080
	tkhdData := make([]byte, 84)
	tkhdData = append(tkhdData, 0x07, 0x80, 0x00, 0x00) // 1920 << 16
	tkhdData = append(tkhdData, 0x04, 0x38, 0x00, 0x00) // 1080 << 16
	tkhdBox := encodeBox("tkhd", tkhdData)
	trakBox := encodeBox("trak", tkhdBox)

	// 2. Synthesize keys box with key "parameters"
	keysHeader := []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01} // version(1)+flags(3), count(4) = 1
	keyEntry := []byte{0x00, 0x00, 0x00, 0x12, 'm', 'd', 't', 'a'}      // size 18, namespace "mdta"
	keyEntry = append(keyEntry, []byte("parameters")...)
	keysBox := encodeBox("keys", append(keysHeader, keyEntry...))

	// 3. Synthesize ilst box with item 1 containing "data"
	promptText := "masterpiece portrait of a robotic guitarist\nSteps: 30, Sampler: Euler"
	dataHdr := []byte{0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00} // type 1 (utf-8), locale 0
	dataBox := encodeBox("data", append(dataHdr, []byte(promptText)...))
	itemBox := encodeBox("____", dataBox)
	// Replace fourcc with index 1 (uint32 big-endian 1)
	binary.BigEndian.PutUint32(itemBox[4:8], 1)

	ilstBox := encodeBox("ilst", itemBox)

	// 4. Synthesize meta box (with 4 bytes version+flags)
	metaData := []byte{0x00, 0x00, 0x00, 0x00} // version + flags
	metaData = append(metaData, keysBox...)
	metaData = append(metaData, ilstBox...)
	metaBox := encodeBox("meta", metaData)

	// 5. Synthesize udta -> moov
	udtaBox := encodeBox("udta", metaBox)
	moovBox := encodeBox("moov", append(trakBox, udtaBox...))

	tags, w, h, err := ParseMP4(bytes.NewReader(moovBox))
	if err != nil {
		t.Fatalf("ParseMP4 synthetic failed: %v", err)
	}

	if w != 1920 || h != 1080 {
		t.Errorf("expected 1920x1080, got %dx%d", w, h)
	}

	if tags["parameters"] != promptText {
		t.Errorf("expected parameters %q, got %q", promptText, tags["parameters"])
	}
}

func encodeBox(fourcc string, data []byte) []byte {
	size := uint32(len(data) + 8)
	buf := make([]byte, 8+len(data))
	binary.BigEndian.PutUint32(buf[0:4], size)
	copy(buf[4:8], []byte(fourcc))
	copy(buf[8:], data)
	return buf
}
