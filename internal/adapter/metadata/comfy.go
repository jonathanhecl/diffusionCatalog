package metadata

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"unicode"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

var stringKeys = []string{"text", "value", "positive_prompt", "prompt", "string"}

var negHints = []string{
	"blurry", "low quality", "worst quality", "lowres", "low res", "deformed",
	"watermark", "bad anatomy", "bad hands", "jpeg artifacts", "extra fingers",
	"mutated hands", "long neck", "text overlay",
}

var filePattern = regexp.MustCompile(`(?i)\.(safetensors|sft|pth|pt|ckpt|gguf|onnx|engine|trt|bin|txt|json|yaml|yml)\b`)

// UsableString checks whether a string is a viable candidate for a diffusion prompt.
func UsableString(s string) bool {
	t := strings.TrimSpace(s)
	if len(t) < 3 {
		return false
	}
	hasAlpha := false
	for _, r := range t {
		if unicode.IsLetter(r) {
			hasAlpha = true
			break
		}
	}
	if !hasAlpha {
		return false
	}
	if strings.Contains(t, ".safetensors") && strings.Contains(t, "|") {
		return false
	}
	if filePattern.MatchString(t) {
		return false
	}
	return true
}

// LooksNegative determines if text resembles a negative prompt based on threshold of negative keywords.
func LooksNegative(s string) bool {
	low := strings.ToLower(s)
	count := 0
	for _, h := range negHints {
		if strings.Contains(low, h) {
			count++
			if count >= 3 {
				return true
			}
		}
	}
	return false
}

// StripNegativeTail removes any trailing negative prompt snippet.
func StripNegativeTail(s string) string {
	low := strings.ToLower(s)
	idx := strings.Index(low, "negative prompt:")
	if idx != -1 {
		s = s[:idx]
	}
	return strings.TrimSpace(s)
}

// SanitizePrompt cleans, strips negative tail, and filters out negative / non-usable strings.
func SanitizePrompt(p string) string {
	p = StripNegativeTail(p)
	p = strings.TrimSpace(p)
	if len(p) < 3 || LooksNegative(p) {
		return ""
	}
	return p
}

// FindPromptKey recursively traverses arbitrary JSON map/slice looking for a "prompt" or "positive_prompt" key.
func FindPromptKey(obj interface{}) string {
	switch v := obj.(type) {
	case map[string]interface{}:
		for _, k := range []string{"prompt", "positive_prompt"} {
			if val, ok := v[k].(string); ok && strings.TrimSpace(val) != "" {
				return strings.TrimSpace(val)
			}
		}
		for _, child := range v {
			if r := FindPromptKey(child); r != "" {
				return r
			}
		}
	case []interface{}:
		for _, child := range v {
			if r := FindPromptKey(child); r != "" {
				return r
			}
		}
	}
	return ""
}

// ComfyExtract walks the ComfyUI workflow graph to locate the primary prompt string.
func ComfyExtract(wf map[string]interface{}) string {
	if len(wf) == 0 {
		return ""
	}

	var results []string

	// 1. Follow nodes with both positive and negative inputs (e.g. KSampler)
	for _, rawNode := range wf {
		node, ok := rawNode.(map[string]interface{})
		if !ok {
			continue
		}
		inputs, ok := node["inputs"].(map[string]interface{})
		if !ok {
			continue
		}
		_, hasPos := inputs["positive"]
		_, hasNeg := inputs["negative"]
		if hasPos && hasNeg {
			pos := inputs["positive"]
			if arr, ok := pos.([]interface{}); ok && len(arr) >= 2 {
				cands := collectStrings(wf, arr[0], make(map[string]bool), 0)
				if len(cands) > 0 {
					longest := maxByLen(cands)
					if longest != "" {
						results = append(results, longest)
					}
				}
			}
		}
	}

	// 2. Fallback: inspect all nodes for string keys
	if len(results) == 0 {
		for _, rawNode := range wf {
			node, ok := rawNode.(map[string]interface{})
			if !ok {
				continue
			}
			inputs, ok := node["inputs"].(map[string]interface{})
			if !ok {
				continue
			}
			for _, k := range stringKeys {
				if v, ok := inputs[k].(string); ok {
					if UsableString(v) && !LooksNegative(v) {
						results = append(results, strings.TrimSpace(v))
					}
				}
			}
		}
	}

	if len(results) == 0 {
		return ""
	}

	var clean []string
	for _, r := range results {
		if !LooksNegative(r) {
			clean = append(clean, r)
		}
	}
	if len(clean) == 0 {
		return ""
	}
	return maxByLen(clean)
}

func collectStrings(wf map[string]interface{}, nid interface{}, seen map[string]bool, depth int) []string {
	var out []string
	if depth > 15 {
		return out
	}
	key := fmt.Sprintf("%v", nid)
	if seen[key] {
		return out
	}
	seen[key] = true

	rawNode, ok := wf[key]
	if !ok {
		return out
	}
	node, ok := rawNode.(map[string]interface{})
	if !ok {
		return out
	}
	inputs, ok := node["inputs"].(map[string]interface{})
	if !ok {
		return out
	}

	for _, v := range inputs {
		switch val := v.(type) {
		case string:
			if UsableString(val) {
				out = append(out, strings.TrimSpace(val))
			}
		case []interface{}:
			if len(val) >= 2 {
				out = append(out, collectStrings(wf, val[0], seen, depth+1)...)
			}
		}
	}
	return out
}

func maxByLen(list []string) string {
	if len(list) == 0 {
		return ""
	}
	best := list[0]
	for _, s := range list[1:] {
		if len(s) > len(best) {
			best = s
		}
	}
	return best
}

// ExtractComfyGraphParams parses both the prompt text and diffusion params from a ComfyUI JSON graph.
func ExtractComfyGraphParams(rawPromptJSON, rawWorkflowJSON string) domain.GenerationParams {
	params := domain.GenerationParams{
		Engine:      domain.EngineComfyUI,
		RawMetadata: rawPromptJSON,
	}

	var graph map[string]interface{}
	if err := json.Unmarshal([]byte(rawPromptJSON), &graph); err != nil {
		return params
	}

	// 1. Extract primary prompt using recursive graph traversal
	extractedPrompt := ComfyExtract(graph)
	params.PositivePrompt = extractedPrompt

	// 2. Extract technical parameters from known nodes
	for _, rawNode := range graph {
		node, ok := rawNode.(map[string]interface{})
		if !ok {
			continue
		}
		ct, _ := node["class_type"].(string)
		ctLower := strings.ToLower(ct)
		inputs, _ := node["inputs"].(map[string]interface{})
		if inputs == nil {
			continue
		}

		switch {
		case strings.Contains(ctLower, "checkpointloader"):
			if ckpt, ok := inputs["ckpt_name"].(string); ok {
				params.ModelName = ckpt
			}
		case strings.Contains(ctLower, "unetloader"):
			if unet, ok := inputs["unet_name"].(string); ok && params.ModelName == "" {
				params.ModelName = unet
			}
		case strings.Contains(ctLower, "loraloader"):
			if lora, ok := inputs["lora_name"].(string); ok {
				params.LoRAs = append(params.LoRAs, lora)
			}
		case strings.Contains(ctLower, "ksampler"):
			if steps, ok := toFloat(inputs["steps"]); ok {
				params.Steps = int(steps)
			}
			if cfg, ok := toFloat(inputs["cfg"]); ok {
				params.CFGScale = cfg
			}
			if seed, ok := toFloat(inputs["seed"]); ok {
				params.Seed = int64(seed)
			} else if nseed, ok := toFloat(inputs["noise_seed"]); ok {
				params.Seed = int64(nseed)
			}
			if sampler, ok := inputs["sampler_name"].(string); ok {
				params.Sampler = sampler
			}
			if scheduler, ok := inputs["scheduler"].(string); ok {
				params.Scheduler = scheduler
			}

			// Extract negative prompt linked to this KSampler if available
			if neg, ok := inputs["negative"].([]interface{}); ok && len(neg) >= 2 {
				negCands := collectStrings(graph, neg[0], make(map[string]bool), 0)
				if len(negCands) > 0 {
					params.NegativePrompt = maxByLen(negCands)
				}
			}
		}
	}

	// If negative prompt not yet populated, check all nodes for negative prompt candidates
	if params.NegativePrompt == "" {
		for _, rawNode := range graph {
			node, ok := rawNode.(map[string]interface{})
			if !ok {
				continue
			}
			inputs, _ := node["inputs"].(map[string]interface{})
			if inputs == nil {
				continue
			}
			for _, k := range stringKeys {
				if text, ok := inputs[k].(string); ok {
					t := strings.TrimSpace(text)
					if LooksNegative(t) || strings.Contains(strings.ToLower(t), "worst quality") || strings.Contains(strings.ToLower(t), "low quality") {
						params.NegativePrompt = t
						break
					}
				}
			}
			if params.NegativePrompt != "" {
				break
			}
		}
	}

	// If no positive prompt extracted via graph, try fallback cliptextencode / fallback keys
	if params.PositivePrompt == "" {
		for _, rawNode := range graph {
			node, ok := rawNode.(map[string]interface{})
			if !ok {
				continue
			}
			inputs, _ := node["inputs"].(map[string]interface{})
			if inputs == nil {
				continue
			}
			if text, ok := inputs["text"].(string); ok {
				t := strings.TrimSpace(text)
				if UsableString(t) && !LooksNegative(t) {
					params.PositivePrompt = t
					break
				}
			}
		}
	}

	if params.LoRAs == nil {
		params.LoRAs = extractLoRAs(params.PositivePrompt)
	}

	return params
}
