package metadata

import (
	"bytes"
	"context"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

// Extractor implements domain.MetadataExtractor using pure Go parsers for PNG, JPEG/WebP, and sidecar files.
type Extractor struct{}

// NewExtractor returns a new pure-Go metadata extractor.
func NewExtractor() *Extractor {
	return &Extractor{}
}

// Extract inspects a companion .txt / .json sidecar first for fast prompt loading, and if missing,
// extracts embedded diffusion prompts/parameters from the media binary and auto-creates a .txt sidecar.
func (e *Extractor) Extract(ctx context.Context, absPath string, mediaType domain.MediaType) (domain.GenerationParams, int, int, error) {
	ext := strings.ToLower(filepath.Ext(absPath))

	var params domain.GenerationParams
	var width, height int

	// 1. Check companion .txt / .caption / .json sidecar first (fast path)
	sidecarParams, fromTXT, hasSidecar := readSidecarMetadata(absPath)
	if hasSidecar && sidecarParams.PositivePrompt != "" {
		params = sidecarParams
		if mediaType == domain.MediaTypeImage {
			width, height = decodeStandardImageConfig(absPath)
		} else if mediaType == domain.MediaTypeVideo {
			width, height = 1024, 576
		}
		// If loaded from .json or .caption and .txt does not exist yet, persist .txt for future instant reads
		if !fromTXT {
			_ = WritePromptSidecarTXT(absPath, params)
		}
		if params.Engine == "" {
			params.Engine = domain.EngineUnknown
		}
		return params, width, height, nil
	}

	// 2. Fallback: extract embedded metadata from PNG chunks / EXIF / raw parameters
	if ext == ".png" {
		p, w, h, err := ParsePNGFile(absPath)
		if err == nil {
			params = p
			width = w
			height = h
		}
	} else if mediaType == domain.MediaTypeImage {
		width, height = decodeStandardImageConfig(absPath)
		params = scanRawFileForParameters(absPath)
	} else if mediaType == domain.MediaTypeVideo {
		width, height = 1024, 576
		params = scanRawFileForParameters(absPath)
	}

	// 3. Auto-create companion <basename>.txt when embedded prompt metadata was extracted and no .txt exists yet
	if !fromTXT && (strings.TrimSpace(params.PositivePrompt) != "" || strings.TrimSpace(params.RawMetadata) != "") {
		_ = WritePromptSidecarTXT(absPath, params)
	}

	if params.Engine == "" {
		params.Engine = domain.EngineUnknown
	}
	return params, width, height, nil
}

func decodeStandardImageConfig(absPath string) (int, int) {
	f, err := os.Open(absPath)
	if err != nil {
		return 0, 0
	}
	defer f.Close()
	cfg, _, err := image.DecodeConfig(f)
	if err != nil {
		return 0, 0
	}
	return cfg.Width, cfg.Height
}

func scanRawFileForParameters(absPath string) domain.GenerationParams {
	data, err := os.ReadFile(absPath)
	if err != nil {
		return domain.GenerationParams{Engine: domain.EngineUnknown}
	}
	// Restrict scan to first 256KB and last 256KB for large video files
	sample := data
	if len(data) > 512*1024 {
		sample = append(append([]byte{}, data[:256*1024]...), data[len(data)-256*1024:]...)
	}

	// Look for "Steps:" marker typical of Automatic1111 / Forge EXIF UserComment
	idx := bytes.Index(sample, []byte("Steps:"))
	if idx > 0 {
		start := idx - 2048
		if start < 0 {
			start = 0
		}
		end := idx + 1024
		if end > len(sample) {
			end = len(sample)
		}
		snippet := cleanPrintableASCII(sample[start:end])
		return ParseA1111Parameters(snippet)
	}
	return domain.GenerationParams{Engine: domain.EngineUnknown}
}

func readSidecarMetadata(absPath string) (domain.GenerationParams, bool, bool) {
	base := strings.TrimSuffix(absPath, filepath.Ext(absPath))
	for _, sideExt := range []string{".txt", ".caption", ".json"} {
		sidePath := base + sideExt
		rawBytes, err := os.ReadFile(sidePath)
		if err != nil {
			continue
		}
		raw := strings.TrimSpace(string(rawBytes))
		if raw == "" {
			continue
		}
		isTXT := sideExt == ".txt"
		if sideExt == ".json" {
			if p, ok := parseJSONMetadata(raw); ok {
				return p, isTXT, true
			}
		}
		if strings.Contains(raw, "Steps:") || strings.Contains(raw, "Seed:") || strings.Contains(raw, "Negative prompt:") {
			return ParseA1111Parameters(raw), isTXT, true
		}
		return domain.GenerationParams{
			Engine:         domain.EngineUnknown,
			PositivePrompt: raw,
			LoRAs:          extractLoRAs(raw),
			RawMetadata:    raw,
		}, isTXT, true
	}
	return domain.GenerationParams{}, false, false
}

// WritePromptSidecarTXT writes the prompt and generation parameters to a companion <basename>.txt file
// next to the media resource so future scans can read the prompt directly from .txt.
func WritePromptSidecarTXT(absMediaPath string, params domain.GenerationParams) error {
	base := strings.TrimSuffix(absMediaPath, filepath.Ext(absMediaPath))
	txtPath := base + ".txt"

	pos := strings.TrimSpace(params.PositivePrompt)
	neg := strings.TrimSpace(params.NegativePrompt)
	if pos == "" && neg == "" {
		raw := strings.TrimSpace(params.RawMetadata)
		if raw == "" {
			return nil
		}
		return os.WriteFile(txtPath, []byte(raw+"\n"), 0644)
	}

	var b strings.Builder
	b.WriteString(pos)
	if neg != "" {
		b.WriteString("\nNegative prompt: ")
		b.WriteString(neg)
	}

	var kv []string
	if params.Steps > 0 {
		kv = append(kv, fmt.Sprintf("Steps: %d", params.Steps))
	}
	if params.Sampler != "" {
		kv = append(kv, fmt.Sprintf("Sampler: %s", params.Sampler))
	}
	if params.Scheduler != "" {
		kv = append(kv, fmt.Sprintf("Schedule type: %s", params.Scheduler))
	}
	if params.CFGScale > 0 {
		kv = append(kv, fmt.Sprintf("CFG scale: %.1f", params.CFGScale))
	}
	if params.Seed != 0 {
		kv = append(kv, fmt.Sprintf("Seed: %d", params.Seed))
	}
	if params.ModelHash != "" {
		kv = append(kv, fmt.Sprintf("Model hash: %s", params.ModelHash))
	}
	if params.ModelName != "" {
		kv = append(kv, fmt.Sprintf("Model: %s", params.ModelName))
	}
	if params.Engine != "" && params.Engine != domain.EngineUnknown {
		kv = append(kv, fmt.Sprintf("Engine: %s", params.Engine))
	}
	if len(kv) > 0 {
		b.WriteString("\n")
		b.WriteString(strings.Join(kv, ", "))
	}
	b.WriteString("\n")

	return os.WriteFile(txtPath, []byte(b.String()), 0644)
}

func cleanPrintableASCII(b []byte) string {
	var out strings.Builder
	for _, c := range b {
		if (c >= 32 && c <= 126) || c == '\n' || c == '\r' {
			out.WriteByte(c)
		}
	}
	return strings.TrimSpace(out.String())
}
