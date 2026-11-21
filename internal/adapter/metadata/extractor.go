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

// Extractor implements domain.MetadataExtractor using pure Go parsers for PNG, JPEG, WebP, and sidecar files.
type Extractor struct{}

// NewExtractor returns a new pure-Go metadata extractor.
func NewExtractor() *Extractor {
	return &Extractor{}
}

// Extract checks embedded diffusion metadata first: if present, it is used and saved to (media).txt
// (only if (media).txt does not already exist). If the prompt is not in metadata, it uses what (media).txt says.
func (e *Extractor) Extract(ctx context.Context, absPath string, mediaType domain.MediaType) (domain.GenerationParams, int, int, error) {
	ext := strings.ToLower(filepath.Ext(absPath))

	var params domain.GenerationParams
	var width, height int

	switch ext {
	case ".png":
		p, w, h, err := ParsePNGFile(absPath)
		if err == nil {
			params = p
			width = w
			height = h
		}
	case ".jpg", ".jpeg":
		width, height = decodeStandardImageConfig(absPath)
		payloads, err := ExtractJPEGEXIFFromFile(absPath)
		if err == nil && len(payloads) > 0 {
			params = extractEXIFParams(payloads)
		}
		if params.PositivePrompt == "" {
			rawParams := scanRawFileForParameters(absPath)
			if rawParams.WorkflowJSON == "" {
				rawParams.WorkflowJSON = params.WorkflowJSON
			}
			params = rawParams
		}
	case ".webp":
		width, height = decodeStandardImageConfig(absPath)
		payloads, err := ExtractWebPEXIFFromFile(absPath)
		if err == nil && len(payloads) > 0 {
			params = extractEXIFParams(payloads)
		}
		if params.PositivePrompt == "" {
			rawParams := scanRawFileForParameters(absPath)
			if rawParams.WorkflowJSON == "" {
				rawParams.WorkflowJSON = params.WorkflowJSON
			}
			params = rawParams
		}
	case ".mp4", ".mov", ".m4v":
		tags, w, h, err := ParseMP4File(absPath)
		if err == nil {
			if w > 0 && h > 0 {
				width = w
				height = h
			}
			params = extractMP4Params(tags)
		}
		if params.PositivePrompt == "" {
			rawParams := scanRawFileForParameters(absPath)
			if rawParams.PositivePrompt != "" {
				if rawParams.WorkflowJSON == "" {
					rawParams.WorkflowJSON = params.WorkflowJSON
				}
				params = rawParams
			}
		}
	case ".webm", ".mkv":
		tags, w, h, err := ParseEBMLFile(absPath)
		if err == nil {
			if w > 0 && h > 0 {
				width = w
				height = h
			}
			params = extractEBMLParams(tags)
		}
		if params.PositivePrompt == "" {
			rawParams := scanRawFileForParameters(absPath)
			if rawParams.PositivePrompt != "" {
				if rawParams.WorkflowJSON == "" {
					rawParams.WorkflowJSON = params.WorkflowJSON
				}
				params = rawParams
			}
		}
	default:
		if mediaType == domain.MediaTypeImage {
			width, height = decodeStandardImageConfig(absPath)
			params = scanRawFileForParameters(absPath)
		} else if mediaType == domain.MediaTypeVideo {
			mp4Tags, mw, mh, err := ParseMP4File(absPath)
			if err == nil && (len(mp4Tags) > 0 || (mw > 0 && mh > 0)) {
				width, height = mw, mh
				params = extractMP4Params(mp4Tags)
			} else {
				tags, w, h, err := ParseEBMLFile(absPath)
				if err == nil {
					width, height = w, h
					params = extractEBMLParams(tags)
				}
			}
			if width == 0 || height == 0 {
				width, height = 1024, 576
			}
			if params.PositivePrompt == "" {
				rawParams := scanRawFileForParameters(absPath)
				if rawParams.PositivePrompt != "" {
					if rawParams.WorkflowJSON == "" {
						rawParams.WorkflowJSON = params.WorkflowJSON
					}
					params = rawParams
				}
			}
		}
	}

	if mediaType == domain.MediaTypeImage && (width == 0 || height == 0) {
		width, height = decodeStandardImageConfig(absPath)
	}

	hasMetadataPrompt := strings.TrimSpace(params.PositivePrompt) != "" || strings.TrimSpace(params.RawMetadata) != ""

	if hasMetadataPrompt {
		// Contains the prompt that generated it in metadata.
		// Save it to companion <media>.txt only if (media).txt does not already exist.
		_ = WritePromptSidecarTXT(absPath, params)
	} else {
		// If not in metadata, use what (media).txt (or companion sidecar) says
		sidecarParams, _, hasSidecar := readSidecarMetadata(absPath)
		if hasSidecar && sidecarParams.PositivePrompt != "" {
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
		start := idx
		// Walk backwards while characters are printable text up to 64KB
		for start > 0 && (start > idx-65536) {
			b := sample[start-1]
			if b == 0 || (b < 32 && b != '\n' && b != '\r' && b != '\t') {
				break
			}
			start--
		}
		end := idx
		for end < len(sample) && (end < idx+4096) {
			b := sample[end]
			if b == 0 {
				break
			}
			end++
		}
		snippet := cleanPrintableASCII(sample[start:end])
		return ParseA1111Parameters(snippet)
	}

	// Look for JSON prompt marker: "prompt":
	pIdx := bytes.Index(sample, []byte(`"prompt":`))
	if pIdx > 0 {
		start := pIdx
		for start > 0 && (start > pIdx-4096) {
			if sample[start] == '{' {
				break
			}
			start--
		}
		if sample[start] == '{' {
			end := pIdx
			depth := 0
			for end < len(sample) && (end < pIdx+65536) {
				if sample[end] == '{' {
					depth++
				} else if sample[end] == '}' {
					depth--
					if depth == 0 {
						end++
						break
					}
				}
				end++
			}
			if depth == 0 && end > start {
				if parsed, ok := ParseJSONMetadata(string(sample[start:end])); ok && parsed.PositivePrompt != "" {
					return parsed
				}
			}
		}
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

	// If (media).txt already exists, only keep it if it has equal or greater prompt detail.
	// If the newly extracted embedded prompt is longer, update .txt to heal truncated sidecars.
	if _, err := os.Stat(txtPath); err == nil {
		existingParams, _, hasExisting := readSidecarMetadata(absMediaPath)
		if hasExisting && len(existingParams.PositivePrompt) >= len(pos) {
			return nil
		}
	}

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

func extractEXIFParams(payloads [][]byte) domain.GenerationParams {
	if len(payloads) == 0 {
		return domain.GenerationParams{}
	}
	texts := extractEXIFStrings(payloads)
	var params domain.GenerationParams
	for _, s := range texts {
		if parsed, found := ClassifyComment(s); found && parsed.PositivePrompt != "" {
			params = parsed
			break
		}
	}
	if wfInfo, ok := extractWorkflowFromEXIFStrings(texts); ok {
		if params.WorkflowJSON == "" || wfInfo.Kind == "comfyui_workflow" {
			if norm, okNorm := NormalizeValidJSON(wfInfo.JSON); okNorm {
				params.WorkflowJSON = norm
			}
		}
	}
	return params
}

func extractMP4Params(tags map[string]string) domain.GenerationParams {
	var params domain.GenerationParams
	if paramsStr, ok := tags["parameters"]; ok && strings.TrimSpace(paramsStr) != "" {
		params = ParseA1111Parameters(paramsStr)
	}
	wfTag := tags["workflow"]
	if strings.TrimSpace(wfTag) == "" {
		wfTag = tags["Workflow"]
	}
	promptTag := tags["prompt"]
	if strings.TrimSpace(promptTag) == "" {
		promptTag = tags["Prompt"]
	}
	if strings.TrimSpace(promptTag) != "" {
		comfyParams := ExtractComfyGraphParams(promptTag, wfTag)
		if params.PositivePrompt == "" {
			params = comfyParams
		} else {
			params.Engine = domain.EngineComfyUI
			if params.ModelName == "" {
				params.ModelName = comfyParams.ModelName
			}
			if len(params.LoRAs) == 0 {
				params.LoRAs = comfyParams.LoRAs
			}
			if params.WorkflowJSON == "" {
				params.WorkflowJSON = comfyParams.WorkflowJSON
			}
		}
	}
	if params.PositivePrompt == "" {
		for _, k := range []string{"description", "comment", "cmt", "title", "civitaiResources"} {
			if val, ok := tags[k]; ok && strings.TrimSpace(val) != "" {
				if parsed, found := ClassifyComment(val); found && parsed.PositivePrompt != "" {
					params = parsed
					break
				}
			}
		}
	}
	if wfNorm, ok := NormalizeValidJSON(wfTag); ok {
		if params.WorkflowJSON == "" {
			params.WorkflowJSON = wfNorm
		} else if existingInfo, okEx := ClassifyWorkflowJSON(params.WorkflowJSON); !okEx || existingInfo.Kind != "comfyui_workflow" {
			if wfInfo, okWf := ClassifyWorkflowJSON(wfNorm); okWf && wfInfo.Kind == "comfyui_workflow" {
				params.WorkflowJSON = wfNorm
			}
		}
		if params.Engine == "" || params.Engine == domain.EngineUnknown {
			params.Engine = domain.EngineComfyUI
		}
		if params.RawMetadata == "" {
			params.RawMetadata = wfTag
		}
	}
	return params
}

func extractEBMLParams(tags map[string]string) domain.GenerationParams {
	wfTag := tags["WORKFLOW"]
	if strings.TrimSpace(wfTag) == "" {
		wfTag = tags["workflow"]
	}
	promptTag := tags["PROMPT"]
	if strings.TrimSpace(promptTag) == "" {
		promptTag = tags["prompt"]
	}
	var params domain.GenerationParams
	if strings.TrimSpace(promptTag) != "" {
		params = ExtractComfyGraphParams(promptTag, wfTag)
	} else {
		for _, k := range []string{"DESCRIPTION", "COMMENT", "TITLE", "Prompt", "prompt"} {
			if val, ok := tags[k]; ok && strings.TrimSpace(val) != "" {
				if parsed, found := ClassifyComment(val); found && parsed.PositivePrompt != "" {
					params = parsed
					break
				}
			}
		}
	}
	if wfNorm, ok := NormalizeValidJSON(wfTag); ok {
		params.WorkflowJSON = wfNorm
		if params.Engine == "" || params.Engine == domain.EngineUnknown {
			params.Engine = domain.EngineComfyUI
		}
		if params.RawMetadata == "" {
			params.RawMetadata = wfTag
		}
	}
	return params
}
