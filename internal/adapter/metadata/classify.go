package metadata

import (
	"encoding/json"
	"strconv"
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

// ParseJSONMetadata attempts to extract generation parameters from a JSON string.
// Supports WanGP / Maestro / Pinokio / NovelAI / SwarmUI / ComfyUI / VHS_VideoCombine and general JSON diffusion metadata.
func ParseJSONMetadata(raw string) (domain.GenerationParams, bool) {
	validRaw, ok := NormalizeValidJSON(raw)
	if !ok || validRaw[0] != '{' {
		return domain.GenerationParams{}, false
	}

	var root map[string]interface{}
	if err := json.Unmarshal([]byte(validRaw), &root); err != nil {
		return domain.GenerationParams{}, false
	}

	// 1a. Check if root itself is a ComfyUI node graph (entries whose value has "class_type")
	if isComfyPromptGraph(root) {
		params := ExtractComfyGraphParams(validRaw, "")
		if p := SanitizePrompt(params.PositivePrompt); p != "" {
			params.PositivePrompt = p
		}
		if params.PositivePrompt != "" {
			return params, true
		}
	}

	// 1b. ComfyUI graph inside "prompt" key (either nested object OR stringified JSON from VHS_VideoCombine)
	if promptJSON, hasPrompt := extractEmbeddedJSONField(root, "prompt"); hasPrompt {
		var promptObj map[string]interface{}
		if err := json.Unmarshal([]byte(promptJSON), &promptObj); err == nil && isComfyPromptGraph(promptObj) {
			wfJSON, _ := extractEmbeddedJSONField(root, "workflow")
			params := ExtractComfyGraphParams(promptJSON, wfJSON)
			if p := SanitizePrompt(params.PositivePrompt); p != "" {
				params.PositivePrompt = p
			}
			if params.PositivePrompt != "" {
				return params, true
			}
		}
	}

	// 1c. SwarmUI envelope ({"sui_image_params": {...}, "sui_extra_data": {...}, "sui_models": [...]})
	if sui, ok := root["sui_image_params"].(map[string]interface{}); ok {
		params := extractSwarmUIParams(sui, root, raw, validRaw)
		if params.PositivePrompt != "" || params.ModelName != "" {
			return params, true
		}
	}

	// 2. Direct string prompt (WanGP, Maestro, Fooocus, NovelAI, Kohya, etc.)
	var positivePrompt string
	for _, k := range []string{"prompt", "positive_prompt", "positive", "caption", "text"} {
		if val, ok := root[k].(string); ok && strings.TrimSpace(val) != "" {
			positivePrompt = SanitizePrompt(val)
			if positivePrompt != "" {
				break
			}
		}
	}

	if positivePrompt == "" {
		// Try deeper search with FindPromptKey
		candidate := FindPromptKey(root)
		if candidate != "" {
			positivePrompt = SanitizePrompt(candidate)
		}
	}

	if positivePrompt == "" {
		return domain.GenerationParams{}, false
	}

	params := domain.GenerationParams{
		Engine:         domain.EngineUnknown,
		PositivePrompt: positivePrompt,
		RawMetadata:    raw,
		WorkflowJSON:   validRaw,
		LoRAs:          extractLoRAs(positivePrompt),
	}

	// Negative prompt
	for _, k := range []string{"alt_prompt", "negative_prompt", "negative", "uc"} {
		if val, ok := root[k].(string); ok && strings.TrimSpace(val) != "" {
			params.NegativePrompt = strings.TrimSpace(val)
			break
		}
	}

	// Seed
	if seedVal, ok := root["seed"]; ok {
		switch s := seedVal.(type) {
		case float64:
			params.Seed = int64(s)
		case string:
			if n, err := strconv.ParseInt(strings.TrimSpace(s), 10, 64); err == nil {
				params.Seed = n
			}
		}
	}

	// Steps
	for _, k := range []string{"num_inference_steps", "steps", "step", "sampling_steps"} {
		if stepVal, ok := root[k]; ok {
			switch s := stepVal.(type) {
			case float64:
				params.Steps = int(s)
			case string:
				if n, err := strconv.Atoi(strings.TrimSpace(s)); err == nil {
					params.Steps = n
				}
			}
			if params.Steps > 0 {
				break
			}
		}
	}

	// CFG Scale
	for _, k := range []string{"cfg_scale", "guidance_scale", "identity_guidance_scale", "cfg"} {
		if cfgVal, ok := root[k]; ok {
			switch c := cfgVal.(type) {
			case float64:
				params.CFGScale = c
			case string:
				if f, err := strconv.ParseFloat(strings.TrimSpace(c), 64); err == nil {
					params.CFGScale = f
				}
			}
			if params.CFGScale > 0 {
				break
			}
		}
	}

	// Sampler
	for _, k := range []string{"sampler", "sampler_name", "scheduler"} {
		if sampVal, ok := root[k].(string); ok && strings.TrimSpace(sampVal) != "" {
			params.Sampler = strings.TrimSpace(sampVal)
			break
		}
	}

	// Model Name
	if mf, ok := root["model_filename"].(string); ok && strings.TrimSpace(mf) != "" {
		cleaned := strings.TrimSpace(mf)
		if strings.Contains(cleaned, "/") {
			parts := strings.Split(cleaned, "/")
			cleaned = parts[len(parts)-1]
		}
		if strings.Contains(cleaned, "\\") {
			parts := strings.Split(cleaned, "\\")
			cleaned = parts[len(parts)-1]
		}
		params.ModelName = cleaned
	}
	if params.ModelName == "" {
		for _, k := range []string{"model_type", "base_model_type", "model", "checkpoint"} {
			if mVal, ok := root[k].(string); ok && strings.TrimSpace(mVal) != "" {
				params.ModelName = strings.TrimSpace(mVal)
				break
			}
		}
	}

	// LoRAs
	if lorasList, ok := root["activated_loras"].([]interface{}); ok && len(lorasList) > 0 {
		for _, item := range lorasList {
			if loraName, ok := item.(string); ok && strings.TrimSpace(loraName) != "" {
				cleanName := strings.TrimSpace(loraName)
				cleanName = strings.TrimSuffix(cleanName, ".safetensors")
				found := false
				for _, existing := range params.LoRAs {
					if strings.EqualFold(existing, cleanName) {
						found = true
						break
					}
				}
				if !found {
					params.LoRAs = append(params.LoRAs, cleanName)
				}
			}
		}
	}

	// Engine
	for _, k := range []string{"type", "engine", "software", "generator"} {
		if tVal, ok := root[k].(string); ok && strings.TrimSpace(tVal) != "" {
			tLower := strings.ToLower(tVal)
			if strings.Contains(tLower, "wangp") || strings.Contains(tLower, "deepbeepmeep") {
				params.Engine = domain.EngineWanGP
			} else if strings.Contains(tLower, "comfyui") {
				params.Engine = domain.EngineComfyUI
			} else if strings.Contains(tLower, "fooocus") {
				params.Engine = domain.EngineFooocus
			} else if strings.Contains(tLower, "invokeai") {
				params.Engine = domain.EngineInvokeAI
			} else if strings.Contains(tLower, "swarmui") {
				params.Engine = domain.EngineSwarmUI
			} else if strings.Contains(tLower, "a1111") || strings.Contains(tLower, "automatic1111") {
				params.Engine = domain.EngineA1111
			} else {
				params.Engine = domain.EngineType(strings.TrimSpace(tVal))
			}
			break
		}
	}

	return params, true
}

// ClassifyComment inspects an EXIF UserComment, ImageDescription or parameters string
// and returns extracted GenerationParams and whether a valid prompt was found.
func ClassifyComment(s string) (domain.GenerationParams, bool) {
	st := strings.TrimSpace(s)
	if st == "" {
		return domain.GenerationParams{}, false
	}

	// 0. JSON metadata format (WanGP / Maestro / NovelAI / ComfyUI JSON / Fooocus)
	if strings.HasPrefix(st, "{") && strings.HasSuffix(st, "}") {
		if parsed, ok := ParseJSONMetadata(st); ok {
			return parsed, true
		}
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

func extractSwarmUIParams(sui, root map[string]interface{}, raw, validRaw string) domain.GenerationParams {
	params := domain.GenerationParams{
		Engine:       domain.EngineSwarmUI,
		RawMetadata:  raw,
		WorkflowJSON: validRaw,
	}

	if extra, ok := root["sui_extra_data"].(map[string]interface{}); ok {
		if comfyStr, ok := extractEmbeddedJSONField(extra, "comfyui_workflow"); ok {
			params.WorkflowJSON = comfyStr
		}
	}
	if wfStr, ok := extractEmbeddedJSONField(root, "workflow"); ok {
		params.WorkflowJSON = wfStr
	}

	for _, k := range []string{"prompt", "positive_prompt", "Prompt"} {
		if val, ok := sui[k].(string); ok && strings.TrimSpace(val) != "" {
			if p := SanitizePrompt(val); p != "" {
				params.PositivePrompt = p
				break
			}
		}
	}
	for _, k := range []string{"negativeprompt", "negative_prompt", "Negative Prompt", "negative"} {
		if val, ok := sui[k].(string); ok && strings.TrimSpace(val) != "" {
			params.NegativePrompt = strings.TrimSpace(val)
			break
		}
	}
	for _, k := range []string{"model", "base_model", "model_name", "Model"} {
		if val, ok := sui[k].(string); ok && strings.TrimSpace(val) != "" {
			params.ModelName = strings.TrimSpace(val)
			break
		}
	}
	if params.ModelName == "" {
		if models, ok := root["sui_models"].([]interface{}); ok && len(models) > 0 {
			if first, ok := models[0].(map[string]interface{}); ok {
				if name, ok := first["name"].(string); ok && strings.TrimSpace(name) != "" {
					params.ModelName = strings.TrimSpace(name)
				}
			}
		}
	}
	if v, ok := toFloat(sui["steps"]); ok {
		params.Steps = int(v)
	}
	for _, k := range []string{"cfgscale", "cfg_scale", "guidance_scale"} {
		if v, ok := toFloat(sui[k]); ok && v > 0 {
			params.CFGScale = v
			break
		}
	}
	if v, ok := toFloat(sui["seed"]); ok {
		params.Seed = int64(v)
	}
	if v, ok := sui["sampler"].(string); ok && strings.TrimSpace(v) != "" {
		params.Sampler = strings.TrimSpace(v)
	}
	if v, ok := sui["scheduler"].(string); ok && strings.TrimSpace(v) != "" {
		params.Scheduler = strings.TrimSpace(v)
	}

	seenLoRA := make(map[string]bool)
	addLoRA := func(name string) {
		n := strings.TrimSuffix(strings.TrimSpace(name), ".safetensors")
		if n != "" && !seenLoRA[strings.ToLower(n)] {
			seenLoRA[strings.ToLower(n)] = true
			params.LoRAs = append(params.LoRAs, n)
		}
	}
	if loras, ok := sui["loras"].([]interface{}); ok {
		for _, item := range loras {
			if s, ok := item.(string); ok {
				addLoRA(s)
			}
		}
	} else if loraStr, ok := sui["loras"].(string); ok && strings.TrimSpace(loraStr) != "" {
		for _, part := range strings.Split(loraStr, ",") {
			addLoRA(part)
		}
	}
	for _, l := range extractLoRAs(params.PositivePrompt) {
		addLoRA(l)
	}

	return params
}
