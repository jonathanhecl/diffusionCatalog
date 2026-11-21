package metadata

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

// NormalizeValidJSON trims whitespace and verifies that raw is valid JSON (checking json.Valid first
// so literal words like "Infinity" or "NaN" inside quoted prompt strings are never altered).
// Only if raw is not valid JSON directly does it apply sanitizeComfyJSON for non-standard bare NaN/Infinity tokens.
func NormalizeValidJSON(raw string) (string, bool) {
	s := strings.TrimSpace(raw)
	if len(s) < 2 || (s[0] != '{' && s[0] != '[') {
		return "", false
	}
	if json.Valid([]byte(s)) {
		return s, true
	}
	sanitized := sanitizeComfyJSON(s)
	if json.Valid([]byte(sanitized)) {
		return sanitized, true
	}
	return "", false
}

// PrettyJSON formats a valid JSON string with two-space indentation and a trailing newline.
func PrettyJSON(raw string) (string, bool) {
	valid, ok := NormalizeValidJSON(raw)
	if !ok {
		return "", false
	}
	var buf bytes.Buffer
	if err := json.Indent(&buf, []byte(valid), "", "  "); err != nil {
		return "", false
	}
	buf.WriteByte('\n')
	return buf.String(), true
}

// ClassifyWorkflowJSON inspects a JSON payload, unwraps nested ComfyUI workflows or prompt graphs
// when present (e.g. VHS_VideoCombine or SwarmUI envelopes), and returns pretty-printed WorkflowInfo.
func ClassifyWorkflowJSON(raw string) (domain.WorkflowInfo, bool) {
	valid, ok := NormalizeValidJSON(raw)
	if !ok {
		return domain.WorkflowInfo{}, false
	}

	if valid[0] == '[' {
		var arr []interface{}
		if err := json.Unmarshal([]byte(valid), &arr); err != nil || len(arr) == 0 {
			return domain.WorkflowInfo{}, false
		}
		pretty, ok := PrettyJSON(valid)
		if !ok {
			return domain.WorkflowInfo{}, false
		}
		return domain.WorkflowInfo{
			JSON:  pretty,
			Kind:  "metadata_json",
			Label: "Generation JSON",
		}, true
	}

	var obj map[string]interface{}
	if err := json.Unmarshal([]byte(valid), &obj); err != nil || len(obj) == 0 {
		return domain.WorkflowInfo{}, false
	}

	// 1. Direct ComfyUI UI workflow (contains top-level "nodes" array)
	if nodes, ok := obj["nodes"].([]interface{}); ok {
		pretty, ok := PrettyJSON(valid)
		if !ok {
			return domain.WorkflowInfo{}, false
		}
		return domain.WorkflowInfo{
			JSON:      pretty,
			Kind:      "comfyui_workflow",
			Label:     "ComfyUI Workflow",
			NodeCount: len(nodes),
		}, true
	}

	// 2. Embedded "workflow" (either nested object or JSON-encoded string) containing "nodes"
	if wfStr, ok := extractEmbeddedJSONField(obj, "workflow"); ok {
		var wfObj map[string]interface{}
		if err := json.Unmarshal([]byte(wfStr), &wfObj); err == nil {
			if nodes, ok := wfObj["nodes"].([]interface{}); ok {
				pretty, ok := PrettyJSON(wfStr)
				if ok {
					return domain.WorkflowInfo{
						JSON:      pretty,
						Kind:      "comfyui_workflow",
						Label:     "ComfyUI Workflow",
						NodeCount: len(nodes),
					}, true
				}
			}
		}
	}

	// 3. SwarmUI sui_extra_data.comfyui_workflow
	if extra, ok := obj["sui_extra_data"].(map[string]interface{}); ok {
		if comfyStr, ok := extractEmbeddedJSONField(extra, "comfyui_workflow"); ok {
			if wfInfo, ok := ClassifyWorkflowJSON(comfyStr); ok {
				return wfInfo, true
			}
		}
	}

	// 4. Direct ComfyUI API prompt graph (node map with "class_type")
	if isComfyPromptGraph(obj) {
		pretty, ok := PrettyJSON(valid)
		if !ok {
			return domain.WorkflowInfo{}, false
		}
		return domain.WorkflowInfo{
			JSON:      pretty,
			Kind:      "comfyui_prompt",
			Label:     "ComfyUI API Graph",
			NodeCount: len(obj),
		}, true
	}

	// 5. Embedded "prompt" (object or JSON-encoded string, e.g. VHS_VideoCombine) that is a ComfyUI API graph
	if promptStr, ok := extractEmbeddedJSONField(obj, "prompt"); ok {
		var promptObj map[string]interface{}
		if err := json.Unmarshal([]byte(promptStr), &promptObj); err == nil && isComfyPromptGraph(promptObj) {
			pretty, ok := PrettyJSON(promptStr)
			if ok {
				return domain.WorkflowInfo{
					JSON:      pretty,
					Kind:      "comfyui_prompt",
					Label:     "ComfyUI API Graph",
					NodeCount: len(promptObj),
				}, true
			}
		}
	}

	// 6. SwarmUI envelope
	if _, ok := obj["sui_image_params"].(map[string]interface{}); ok {
		pretty, ok := PrettyJSON(valid)
		if !ok {
			return domain.WorkflowInfo{}, false
		}
		return domain.WorkflowInfo{
			JSON:  pretty,
			Kind:  "swarmui",
			Label: "SwarmUI Metadata JSON",
		}, true
	}

	// 7. Generic non-empty JSON metadata (WanGP, InvokeAI, etc.)
	pretty, ok := PrettyJSON(valid)
	if !ok {
		return domain.WorkflowInfo{}, false
	}
	return domain.WorkflowInfo{
		JSON:  pretty,
		Kind:  "metadata_json",
		Label: "Generation JSON",
	}, true
}

// ExtractWorkflowFromFile inspects a media file on disk in pure Go without loading video pixel streams (mdat)
// and localizes the embedded workflow or generation JSON (preferring full ComfyUI UI workflows over API graphs).
func ExtractWorkflowFromFile(absPath string, mediaType domain.MediaType) (domain.WorkflowInfo, bool) {
	if strings.TrimSpace(absPath) == "" {
		return domain.WorkflowInfo{}, false
	}

	ext := strings.ToLower(filepath.Ext(absPath))
	switch ext {
	case ".png":
		chunks, exifPayloads, _, _, err := readPNGChunksAndEXIFFile(absPath)
		if err == nil {
			for _, k := range []string{"workflow", "Workflow", "prompt", "Prompt", "sui_image_params", "invokeai_metadata", "sd-metadata", "generation_data", "extraMetadata"} {
				if raw, ok := chunks[k]; ok {
					if wf, ok := ClassifyWorkflowJSON(raw); ok {
						if k == "sui_image_params" && wf.Kind == "metadata_json" {
							wf.Kind = "swarmui"
							wf.Label = "SwarmUI Metadata JSON"
						}
						wf.FileName = SuggestedWorkflowFileName(absPath)
						return wf, true
					}
				}
			}
			if len(exifPayloads) > 0 {
				if wf, ok := extractWorkflowFromEXIFStrings(extractEXIFStrings(exifPayloads)); ok {
					wf.FileName = SuggestedWorkflowFileName(absPath)
					return wf, true
				}
			}
			for _, k := range []string{"Comment", "Description", "UserComment"} {
				if raw, ok := chunks[k]; ok {
					if wf, ok := ClassifyWorkflowJSON(raw); ok {
						wf.FileName = SuggestedWorkflowFileName(absPath)
						return wf, true
					}
				}
			}
		}

	case ".mp4", ".mov", ".m4v":
		tags, _, _, err := ParseMP4File(absPath)
		if err == nil {
			if wf, ok := extractWorkflowFromMP4Tags(tags, absPath); ok {
				return wf, true
			}
		}

	case ".webm", ".mkv":
		tags, _, _, err := ParseEBMLFile(absPath)
		if err == nil {
			if wf, ok := extractWorkflowFromEBMLTags(tags, absPath); ok {
				return wf, true
			}
		}

	case ".jpg", ".jpeg":
		if payloads, err := ExtractJPEGEXIFFromFile(absPath); err == nil && len(payloads) > 0 {
			if wf, ok := extractWorkflowFromEXIFStrings(extractEXIFStrings(payloads)); ok {
				wf.FileName = SuggestedWorkflowFileName(absPath)
				return wf, true
			}
		}

	case ".webp":
		if payloads, err := ExtractWebPEXIFFromFile(absPath); err == nil && len(payloads) > 0 {
			if wf, ok := extractWorkflowFromEXIFStrings(extractEXIFStrings(payloads)); ok {
				wf.FileName = SuggestedWorkflowFileName(absPath)
				return wf, true
			}
		}

	default:
		if mediaType == domain.MediaTypeVideo {
			if tags, _, _, err := ParseMP4File(absPath); err == nil && len(tags) > 0 {
				if wf, ok := extractWorkflowFromMP4Tags(tags, absPath); ok {
					return wf, true
				}
			} else if tags, _, _, err := ParseEBMLFile(absPath); err == nil && len(tags) > 0 {
				if wf, ok := extractWorkflowFromEBMLTags(tags, absPath); ok {
					return wf, true
				}
			}
		}
	}

	// Companion sidecar JSON fallback (<base>.workflow.json, <base>.json)
	base := strings.TrimSuffix(absPath, filepath.Ext(absPath))
	for _, sideExt := range []string{".workflow.json", ".json"} {
		if rawBytes, err := os.ReadFile(base + sideExt); err == nil {
			if wf, ok := ClassifyWorkflowJSON(string(rawBytes)); ok {
				wf.FileName = SuggestedWorkflowFileName(absPath)
				return wf, true
			}
		}
	}

	return domain.WorkflowInfo{}, false
}

func extractWorkflowFromMP4Tags(tags map[string]string, absPath string) (domain.WorkflowInfo, bool) {
	for _, k := range []string{"workflow", "Workflow"} {
		if raw, ok := tags[k]; ok {
			if wf, ok := ClassifyWorkflowJSON(raw); ok {
				wf.FileName = SuggestedWorkflowFileName(absPath)
				return wf, true
			}
		}
	}

	var fallback *domain.WorkflowInfo
	for _, k := range []string{"comment", "cmt", "description", "des", "desc"} {
		raw, ok := tags[k]
		if !ok || strings.TrimSpace(raw) == "" {
			continue
		}
		if wf, ok := ClassifyWorkflowJSON(raw); ok {
			if wf.Kind == "comfyui_workflow" || wf.Kind == "comfyui_prompt" || wf.Kind == "swarmui" {
				wf.FileName = SuggestedWorkflowFileName(absPath)
				return wf, true
			}
			if fallback == nil {
				cp := wf
				fallback = &cp
			}
		} else if wf, ok := extractWorkflowFromEXIFStrings([]string{raw}); ok {
			wf.FileName = SuggestedWorkflowFileName(absPath)
			return wf, true
		}
	}

	for _, k := range []string{"prompt", "Prompt", "civitaiResources"} {
		if raw, ok := tags[k]; ok {
			if wf, ok := ClassifyWorkflowJSON(raw); ok {
				wf.FileName = SuggestedWorkflowFileName(absPath)
				return wf, true
			}
		}
	}

	if fallback != nil {
		fallback.FileName = SuggestedWorkflowFileName(absPath)
		return *fallback, true
	}
	return domain.WorkflowInfo{}, false
}

func extractWorkflowFromEBMLTags(tags map[string]string, absPath string) (domain.WorkflowInfo, bool) {
	for _, k := range []string{"WORKFLOW", "workflow", "PROMPT", "prompt", "DESCRIPTION", "COMMENT", "TITLE"} {
		if raw, ok := tags[k]; ok {
			if wf, ok := ClassifyWorkflowJSON(raw); ok {
				wf.FileName = SuggestedWorkflowFileName(absPath)
				return wf, true
			}
		}
	}
	return domain.WorkflowInfo{}, false
}

func extractWorkflowFromEXIFStrings(texts []string) (domain.WorkflowInfo, bool) {
	// Pass 1: Explicit "Workflow:" prefix or direct comfyui_workflow JSON
	for _, s := range texts {
		lower := strings.ToLower(s)
		if idx := strings.Index(lower, "workflow:"); idx != -1 {
			rest := strings.TrimSpace(s[idx+len("workflow:"):])
			if obj := extractLeadingJSONObject(rest); obj != "" {
				if wf, ok := ClassifyWorkflowJSON(obj); ok {
					return wf, true
				}
			}
		}
		if wf, ok := ClassifyWorkflowJSON(s); ok && wf.Kind == "comfyui_workflow" {
			return wf, true
		}
	}

	// Pass 2: Explicit "Prompt:" prefix JSON
	for _, s := range texts {
		lower := strings.ToLower(s)
		if idx := strings.Index(lower, "prompt:"); idx != -1 {
			rest := strings.TrimSpace(s[idx+len("prompt:"):])
			if obj := extractLeadingJSONObject(rest); obj != "" {
				if wf, ok := ClassifyWorkflowJSON(obj); ok {
					return wf, true
				}
			}
		}
	}

	// Pass 3: Any standalone JSON payload in EXIF strings
	for _, s := range texts {
		if wf, ok := ClassifyWorkflowJSON(s); ok {
			return wf, true
		}
	}
	return domain.WorkflowInfo{}, false
}

func extractEmbeddedJSONField(m map[string]interface{}, key string) (string, bool) {
	val, ok := m[key]
	if !ok || val == nil {
		return "", false
	}
	switch v := val.(type) {
	case string:
		return NormalizeValidJSON(v)
	case map[string]interface{}, []interface{}:
		b, err := json.Marshal(v)
		if err != nil {
			return "", false
		}
		return NormalizeValidJSON(string(b))
	default:
		return "", false
	}
}

func extractLeadingJSONObject(s string) string {
	s = strings.TrimSpace(s)
	if len(s) < 2 || s[0] != '{' {
		return ""
	}
	depth := 0
	inString := false
	escaped := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if escaped {
			escaped = false
			continue
		}
		if c == '\\' && inString {
			escaped = true
			continue
		}
		if c == '"' {
			inString = !inString
			continue
		}
		if inString {
			continue
		}
		if c == '{' {
			depth++
		} else if c == '}' {
			depth--
			if depth == 0 {
				return s[:i+1]
			}
		}
	}
	return ""
}

// SuggestedWorkflowFileName builds a sanitized "<stem>_workflow.json" filename for HTTP downloads.
func SuggestedWorkflowFileName(assetFileName string) string {
	base := filepath.Base(strings.TrimSpace(assetFileName))
	stem := strings.TrimSuffix(base, filepath.Ext(base))
	var b strings.Builder
	for _, r := range stem {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '.' || r == '_' || r == '-' {
			b.WriteRune(r)
		} else if r == ' ' {
			b.WriteByte('_')
		}
	}
	cleanStem := strings.Trim(b.String(), "._-")
	if cleanStem == "" {
		cleanStem = "asset"
	}
	return cleanStem + "_workflow.json"
}
