package metadata

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

var pngSignature = []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}

var loraRegex = regexp.MustCompile(`<lora:([^:>]+):[^>]+>`)

// ParsePNGFile inspects a PNG file's IHDR (dimensions) and text chunks (tEXt, iTXt, zTXt)
// to extract diffusion generation parameters without external dependencies.
func ParsePNGFile(absPath string) (domain.GenerationParams, int, int, error) {
	f, err := os.Open(absPath)
	if err != nil {
		return domain.GenerationParams{Engine: domain.EngineUnknown}, 0, 0, err
	}
	defer f.Close()

	return ParsePNGReader(f)
}

// ParsePNGReader reads PNG chunks from an io.Reader.
func ParsePNGReader(r io.Reader) (domain.GenerationParams, int, int, error) {
	sig := make([]byte, 8)
	if _, err := io.ReadFull(r, sig); err != nil {
		return domain.GenerationParams{Engine: domain.EngineUnknown}, 0, 0, err
	}
	if !bytes.Equal(sig, pngSignature) {
		return domain.GenerationParams{Engine: domain.EngineUnknown}, 0, 0, fmt.Errorf("not a valid PNG signature")
	}

	var width, height int
	chunks := make(map[string]string)

	for {
		var length uint32
		if err := binary.Read(r, binary.BigEndian, &length); err != nil {
			if err == io.EOF {
				break
			}
			return domain.GenerationParams{Engine: domain.EngineUnknown}, width, height, nil
		}

		chunkTypeBytes := make([]byte, 4)
		if _, err := io.ReadFull(r, chunkTypeBytes); err != nil {
			break
		}
		chunkType := string(chunkTypeBytes)

		if length > 32*1024*1024 {
			// Guard against malformed huge chunks
			break
		}

		data := make([]byte, length)
		if _, err := io.ReadFull(r, data); err != nil {
			break
		}

		// Skip 4-byte CRC
		crcBuf := make([]byte, 4)
		_, _ = io.ReadFull(r, crcBuf)

		switch chunkType {
		case "IHDR":
			if len(data) >= 8 {
				width = int(binary.BigEndian.Uint32(data[0:4]))
				height = int(binary.BigEndian.Uint32(data[4:8]))
			}
		case "tEXt":
			key, val := parseTEXTChunk(data)
			if key != "" {
				chunks[key] = val
			}
		case "iTXt":
			key, val := parseITXTChunk(data)
			if key != "" {
				chunks[key] = val
			}
		case "IEND":
			goto done
		}
	}

done:
	params := InterpretTextChunks(chunks)
	return params, width, height, nil
}

func parseTEXTChunk(data []byte) (string, string) {
	sep := bytes.IndexByte(data, 0)
	if sep < 0 {
		return "", ""
	}
	return string(data[:sep]), string(data[sep+1:])
}

func parseITXTChunk(data []byte) (string, string) {
	sep := bytes.IndexByte(data, 0)
	if sep < 0 || sep+3 >= len(data) {
		return "", ""
	}
	key := string(data[:sep])
	compFlag := data[sep+1]
	// sep+2 is compression method
	rest := data[sep+3:]

	// Skip language tag (null-terminated)
	langSep := bytes.IndexByte(rest, 0)
	if langSep < 0 {
		return key, ""
	}
	rest = rest[langSep+1:]

	// Skip translated keyword (null-terminated)
	transSep := bytes.IndexByte(rest, 0)
	if transSep < 0 {
		return key, ""
	}
	textData := rest[transSep+1:]

	if compFlag == 1 {
		zr, err := zlib.NewReader(bytes.NewReader(textData))
		if err != nil {
			return key, ""
		}
		defer zr.Close()
		out, err := io.ReadAll(zr)
		if err != nil {
			return key, ""
		}
		return key, string(out)
	}
	return key, string(textData)
}

// InterpretTextChunks normalizes PNG text chunk key-value pairs into GenerationParams.
func InterpretTextChunks(chunks map[string]string) domain.GenerationParams {
	// 1. Automatic1111 / Forge / Fooocus "parameters" chunk
	if raw, ok := chunks["parameters"]; ok && strings.TrimSpace(raw) != "" {
		// Check if it's JSON (some Fooocus/Swarm exports)
		if strings.HasPrefix(strings.TrimSpace(raw), "{") {
			if p, ok := parseJSONMetadata(raw); ok {
				return p
			}
		}
		return ParseA1111Parameters(raw)
	}

	// 2. SwarmUI "sui_image_params"
	if raw, ok := chunks["sui_image_params"]; ok && strings.TrimSpace(raw) != "" {
		if p, ok := parseJSONMetadata(raw); ok {
			p.Engine = domain.EngineSwarmUI
			return p
		}
	}

	// 3. InvokeAI "invokeai_metadata" or "sd-metadata"
	for _, k := range []string{"invokeai_metadata", "sd-metadata"} {
		if raw, ok := chunks[k]; ok && strings.TrimSpace(raw) != "" {
			if p, ok := parseJSONMetadata(raw); ok {
				p.Engine = domain.EngineInvokeAI
				return p
			}
		}
	}

	// 4. ComfyUI "prompt" or "workflow"
	if rawPrompt, ok := chunks["prompt"]; ok && strings.TrimSpace(rawPrompt) != "" {
		return ParseComfyUIPromptGraph(rawPrompt, chunks["workflow"])
	}

	// 5. Fallback: Comment or Description chunk
	for _, k := range []string{"Comment", "Description", "UserComment"} {
		if raw, ok := chunks[k]; ok && strings.TrimSpace(raw) != "" {
			if strings.Contains(raw, "Steps:") {
				return ParseA1111Parameters(raw)
			}
			if p, ok := parseJSONMetadata(raw); ok {
				return p
			}
			return domain.GenerationParams{
				Engine:         domain.EngineUnknown,
				PositivePrompt: strings.TrimSpace(raw),
				RawMetadata:    raw,
			}
		}
	}

	return domain.GenerationParams{Engine: domain.EngineUnknown}
}

// ParseA1111Parameters parses Automatic1111 / Forge multi-line parameter text.
func ParseA1111Parameters(raw string) domain.GenerationParams {
	params := domain.GenerationParams{
		Engine:      domain.EngineA1111,
		RawMetadata: raw,
	}

	lines := strings.Split(strings.ReplaceAll(raw, "\r\n", "\n"), "\n")
	var posLines []string
	var negLines []string
	var kvLine string
	inNeg := false

	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if i == len(lines)-1 && (strings.Contains(trimmed, "Steps:") || strings.Contains(trimmed, "Seed:")) {
			kvLine = trimmed
			continue
		}
		if strings.HasPrefix(trimmed, "Negative prompt:") {
			inNeg = true
			negLines = append(negLines, strings.TrimSpace(strings.TrimPrefix(trimmed, "Negative prompt:")))
			continue
		}
		if strings.HasPrefix(trimmed, "Steps:") && kvLine == "" {
			kvLine = trimmed
			continue
		}
		if inNeg {
			negLines = append(negLines, line)
		} else {
			posLines = append(posLines, line)
		}
	}

	params.PositivePrompt = strings.TrimSpace(strings.Join(posLines, "\n"))
	params.NegativePrompt = strings.TrimSpace(strings.Join(negLines, "\n"))
	params.LoRAs = extractLoRAs(params.PositivePrompt)

	if kvLine != "" {
		pairs := splitKeyValueCSV(kvLine)
		for k, v := range pairs {
			switch strings.ToLower(strings.TrimSpace(k)) {
			case "steps":
				params.Steps, _ = strconv.Atoi(v)
			case "sampler":
				params.Sampler = v
			case "schedule type", "scheduler":
				params.Scheduler = v
			case "cfg scale":
				params.CFGScale, _ = strconv.ParseFloat(v, 64)
			case "seed":
				params.Seed, _ = strconv.ParseInt(v, 10, 64)
			case "model":
				params.ModelName = v
			case "model hash":
				params.ModelHash = v
			case "engine":
				if v != "" {
					params.Engine = domain.EngineType(v)
				}
			case "version":
				if strings.Contains(strings.ToLower(v), "fooocus") {
					params.Engine = domain.EngineFooocus
				}
			}
		}
	}

	return params
}

// ParseComfyUIPromptGraph extracts positive/negative CLIPTextEncode nodes, KSampler settings, and CheckpointLoader from a ComfyUI prompt JSON graph.
func ParseComfyUIPromptGraph(rawPromptJSON, rawWorkflowJSON string) domain.GenerationParams {
	params := domain.GenerationParams{
		Engine:      domain.EngineComfyUI,
		RawMetadata: rawPromptJSON,
	}

	var nodes map[string]struct {
		ClassType string                 `json:"class_type"`
		Inputs    map[string]interface{} `json:"inputs"`
	}
	if err := json.Unmarshal([]byte(rawPromptJSON), &nodes); err != nil {
		return params
	}

	var prompts []string
	for _, node := range nodes {
		ct := strings.ToLower(node.ClassType)
		switch {
		case strings.Contains(ct, "checkpointloader"):
			if ckpt, ok := node.Inputs["ckpt_name"].(string); ok {
				params.ModelName = ckpt
			}
		case strings.Contains(ct, "unetloader"):
			if unet, ok := node.Inputs["unet_name"].(string); ok && params.ModelName == "" {
				params.ModelName = unet
			}
		case strings.Contains(ct, "loraloader"):
			if lora, ok := node.Inputs["lora_name"].(string); ok {
				params.LoRAs = append(params.LoRAs, lora)
			}
		case strings.Contains(ct, "ksampler"):
			if steps, ok := toFloat(node.Inputs["steps"]); ok {
				params.Steps = int(steps)
			}
			if cfg, ok := toFloat(node.Inputs["cfg"]); ok {
				params.CFGScale = cfg
			}
			if seed, ok := toFloat(node.Inputs["seed"]); ok {
				params.Seed = int64(seed)
			} else if nseed, ok := toFloat(node.Inputs["noise_seed"]); ok {
				params.Seed = int64(nseed)
			}
			if sampler, ok := node.Inputs["sampler_name"].(string); ok {
				params.Sampler = sampler
			}
			if scheduler, ok := node.Inputs["scheduler"].(string); ok {
				params.Scheduler = scheduler
			}
		case strings.Contains(ct, "cliptextencode"):
			if text, ok := node.Inputs["text"].(string); ok && strings.TrimSpace(text) != "" {
				prompts = append(prompts, strings.TrimSpace(text))
			}
		}
	}

	// Heuristic: longer prompt is usually positive, or prompt containing negative keywords is negative.
	for _, p := range prompts {
		lower := strings.ToLower(p)
		if strings.Contains(lower, "worst quality") || strings.Contains(lower, "low quality") || strings.Contains(lower, "bad anatomy") || strings.Contains(lower, "ugly") {
			if params.NegativePrompt == "" {
				params.NegativePrompt = p
				continue
			}
		}
		if params.PositivePrompt == "" {
			params.PositivePrompt = p
		} else if params.NegativePrompt == "" {
			params.NegativePrompt = p
		}
	}

	return params
}

func parseJSONMetadata(raw string) (domain.GenerationParams, bool) {
	var data map[string]interface{}
	if err := json.Unmarshal([]byte(raw), &data); err != nil {
		return domain.GenerationParams{}, false
	}

	// Check if nested under "sui_image_params"
	if inner, ok := data["sui_image_params"].(map[string]interface{}); ok {
		data = inner
	}

	params := domain.GenerationParams{
		Engine:      domain.EngineFooocus,
		RawMetadata: raw,
	}

	for _, k := range []string{"prompt", "positive_prompt", "Prompt"} {
		if v, ok := data[k].(string); ok && v != "" {
			params.PositivePrompt = v
			break
		}
	}
	for _, k := range []string{"negative_prompt", "Negative Prompt", "negative"} {
		if v, ok := data[k].(string); ok && v != "" {
			params.NegativePrompt = v
			break
		}
	}
	for _, k := range []string{"model", "base_model", "model_name", "Model"} {
		if v, ok := data[k].(string); ok && v != "" {
			params.ModelName = v
			break
		}
	}
	if v, ok := toFloat(data["steps"]); ok {
		params.Steps = int(v)
	}
	if v, ok := toFloat(data["cfg_scale"]); ok {
		params.CFGScale = v
	} else if v, ok := toFloat(data["guidance_scale"]); ok {
		params.CFGScale = v
	}
	if v, ok := toFloat(data["seed"]); ok {
		params.Seed = int64(v)
	}
	if v, ok := data["sampler"].(string); ok {
		params.Sampler = v
	}
	if v, ok := data["scheduler"].(string); ok {
		params.Scheduler = v
	}

	params.LoRAs = extractLoRAs(params.PositivePrompt)
	return params, params.PositivePrompt != "" || params.ModelName != ""
}

func extractLoRAs(prompt string) []string {
	matches := loraRegex.FindAllStringSubmatch(prompt, -1)
	var out []string
	seen := make(map[string]bool)
	for _, m := range matches {
		if len(m) > 1 && !seen[m[1]] {
			seen[m[1]] = true
			out = append(out, m[1])
		}
	}
	return out
}

func splitKeyValueCSV(line string) map[string]string {
	res := make(map[string]string)
	parts := strings.Split(line, ",")
	for _, part := range parts {
		kv := strings.SplitN(part, ":", 2)
		if len(kv) == 2 {
			res[strings.TrimSpace(kv[0])] = strings.TrimSpace(kv[1])
		}
	}
	return res
}

func toFloat(v interface{}) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case string:
		f, err := strconv.ParseFloat(strings.TrimSpace(n), 64)
		return f, err == nil
	default:
		return 0, false
	}
}
