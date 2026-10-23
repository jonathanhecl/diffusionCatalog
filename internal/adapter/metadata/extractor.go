package metadata

import (
	"bytes"
	"context"
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

// Extract inspects a media file and optional sidecar (.txt / .json) to extract diffusion prompts and dimensions.
func (e *Extractor) Extract(ctx context.Context, absPath string, mediaType domain.MediaType) (domain.GenerationParams, int, int, error) {
	ext := strings.ToLower(filepath.Ext(absPath))

	var params domain.GenerationParams
	var width, height int

	if ext == ".png" {
		p, w, h, err := ParsePNGFile(absPath)
		if err == nil {
			params = p
			width = w
			height = h
		}
	} else if mediaType == domain.MediaTypeImage {
		// Decode image dimensions and scan raw bytes for embedded A1111/EXIF UserComment text
		width, height = decodeStandardImageConfig(absPath)
		params = scanRawFileForParameters(absPath)
	} else if mediaType == domain.MediaTypeVideo {
		width, height = 1024, 576
		params = scanRawFileForParameters(absPath)
	}

	// Check for companion .txt or .json sidecar file (common in Kohya / ComfyUI video exports)
	if params.PositivePrompt == "" {
		if sidecarParams, ok := readSidecarMetadata(absPath); ok {
			params = sidecarParams
		}
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

func readSidecarMetadata(absPath string) (domain.GenerationParams, bool) {
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
		if sideExt == ".json" {
			if p, ok := parseJSONMetadata(raw); ok {
				return p, true
			}
		}
		if strings.Contains(raw, "Steps:") {
			return ParseA1111Parameters(raw), true
		}
		return domain.GenerationParams{
			Engine:         domain.EngineUnknown,
			PositivePrompt: raw,
			RawMetadata:    raw,
		}, true
	}
	return domain.GenerationParams{}, false
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
