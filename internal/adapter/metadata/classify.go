package metadata

import (
	"encoding/json"
	"strings"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

// A1111ParsePrompt extracts only the prompt lines up to "Negative prompt:" or "Steps:".
func A1111ParsePrompt(text string) string {
	normalized := strings.ReplaceAll(text, "\r\n", "\n")
	lines := strings.Split(normalized, "\n")
	var out []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "Negative prompt:") || strings.HasPrefix(trimmed, "Steps:") {
			break
		}
		out = append(out, line)
	}
	return strings.TrimSpace(strings.Join(out, "\n"))
}

// ClassifyComment inspects an EXIF UserComment, ImageDescription or parameters string
// and returns extracted GenerationParams and whether a valid prompt was found.
func ClassifyComment(s string) (domain.GenerationParams, bool) {
	st := strings.TrimSpace(s)
	if st == "" {
		return domain.GenerationParams{}, false
	}

	// 1. A1111 / Forge multi-line format
	if strings.HasPrefix(st, "Steps:") || strings.Contains(st, "\nSteps:") || strings.Contains(st, "\nNegative prompt:") {
		params := ParseA1111Parameters(st)
		prompt := SanitizePrompt(A1111ParsePrompt(st))
		if prompt != "" {
			params.PositivePrompt = prompt
			return params, true
		}
	}

	// 2. Embedded "Prompt:" marker (frequently found in ComfyUI EXIF UserComment or Fooocus)
	idx := strings.Index(st, "Prompt:")
	for idx != -1 {
		rest := strings.TrimLeft(st[idx+len("Prompt:"):], " \t\r\n")
		if strings.HasPrefix(rest, "{") {
			// ComfyUI JSON embedded in comment
			var graph map[string]interface{}
			// Find balancing JSON or decode directly
			if err := json.Unmarshal([]byte(rest), &graph); err == nil {
				prompt := SanitizePrompt(ComfyExtract(graph))
				if prompt != "" {
					params := ExtractComfyGraphParams(rest, "")
					params.PositivePrompt = prompt
					return params, true
				}
			}
		} else if idx == 0 {
			cand := SanitizePrompt(rest)
			if cand != "" {
				return domain.GenerationParams{
					Engine:         domain.EngineUnknown,
					PositivePrompt: cand,
					RawMetadata:    st,
					LoRAs:          extractLoRAs(cand),
				}, true
			}
		}
		next := strings.Index(st[idx+1:], "Prompt:")
		if next == -1 {
			break
		}
		idx = idx + 1 + next
	}

	// 3. Workflow or raw graph dumps without usable prompt
	stLower := strings.ToLower(st)
	if strings.HasPrefix(stLower, "workflow:") || strings.Contains(st, `"class_type"`) {
		// Attempt to parse as ComfyUI JSON if it starts with {
		if strings.HasPrefix(st, "{") {
			var graph map[string]interface{}
			if err := json.Unmarshal([]byte(st), &graph); err == nil {
				prompt := SanitizePrompt(ComfyExtract(graph))
				if prompt != "" {
					params := ExtractComfyGraphParams(st, "")
					params.PositivePrompt = prompt
					return params, true
				}
			}
		}
		return domain.GenerationParams{}, false
	}

	// 4. Standalone prompt text
	cand := SanitizePrompt(st)
	if cand != "" {
		return domain.GenerationParams{
			Engine:         domain.EngineUnknown,
			PositivePrompt: cand,
			RawMetadata:    st,
			LoRAs:          extractLoRAs(cand),
		}, true
	}

	return domain.GenerationParams{}, false
}
