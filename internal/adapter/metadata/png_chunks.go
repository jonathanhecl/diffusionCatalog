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

// ParsePNGFile inspects a PNG file's IHDR and chunks (tEXt, zTXt, iTXt, eXIf)
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
	var exifPayloads [][]byte

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

		if length > 64*1024*1024 {
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
		case "zTXt":
			key, val := parseZTXTChunk(data)
			if key != "" {
				chunks[key] = val
			}
		case "iTXt":
			key, val := parseITXTChunk(data)
			if key != "" {
				chunks[key] = val
			}
		case "eXIf":
			exifPayloads = append(exifPayloads, data)
		case "IEND":
			goto done
		}
	}

done:
	params := InterpretPNGMetadata(chunks, exifPayloads)
	return params, width, height, nil
}

func parseTEXTChunk(data []byte) (string, string) {
	sep := bytes.IndexByte(data, 0)
	if sep < 0 {
		return "", ""
	}
	return string(data[:sep]), string(data[sep+1:])
}

func parseZTXTChunk(data []byte) (string, string) {
	sep := bytes.IndexByte(data, 0)
	if sep < 0 || sep+2 >= len(data) {
		return "", ""
	}
	key := string(data[:sep])
	// data[sep+1] is compression method (0 = deflate)
	compressedData := data[sep+2:]

	zr, err := zlib.NewReader(bytes.NewReader(compressedData))
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

// InterpretPNGMetadata extracts diffusion GenerationParams from text chunks and eXIf payloads
// following the exact multi-stage waterfall hierarchy of modern diffusion pipelines.
func InterpretPNGMetadata(chunks map[string]string, exifPayloads [][]byte) domain.GenerationParams {
	// 1. "extraMetadata" JSON
	if em, ok := chunks["extraMetadata"]; ok && strings.TrimSpace(em) != "" {
		var obj interface{}
		if err := json.Unmarshal([]byte(em), &obj); err == nil {
			if p := SanitizePrompt(FindPromptKey(obj)); p != "" {
				return domain.GenerationParams{
					Engine:         domain.EngineUnknown,
					PositivePrompt: p,
					RawMetadata:    em,
					LoRAs:          extractLoRAs(p),
				}
			}
		}
	}

	// 2. "parameters" chunk (A1111 / Forge / Fooocus)
	if par, ok := chunks["parameters"]; ok && strings.TrimSpace(par) != "" {
		st := strings.TrimSpace(par)
		if strings.HasPrefix(st, "{") {
			var obj interface{}
			if err := json.Unmarshal([]byte(st), &obj); err == nil {
				if p := SanitizePrompt(FindPromptKey(obj)); p != "" {
					params, _ := parseJSONMetadata(st)
					params.PositivePrompt = p
					return params
				}
			}
		}
		params := ParseA1111Parameters(par)
		prompt := SanitizePrompt(A1111ParsePrompt(par))
		if prompt != "" {
			params.PositivePrompt = prompt
		}
		if params.PositivePrompt != "" {
			return params
		}
	}

	// 3. EXIF payloads (from eXIf chunks)
	if len(exifPayloads) > 0 {
		exifTexts := extractEXIFStrings(exifPayloads)
		for _, s := range exifTexts {
			if params, found := ClassifyComment(s); found && params.PositivePrompt != "" {
				return params
			}
		}
	}

	// 4. "prompt" chunk (ComfyUI graph)
	if pr, ok := chunks["prompt"]; ok && strings.TrimSpace(pr) != "" {
		st := strings.TrimSpace(pr)
		if strings.HasPrefix(st, "{") {
			params := ExtractComfyGraphParams(st, chunks["workflow"])
			if params.PositivePrompt != "" {
				return params
			}
		} else if len(st) > 20 {
			if p := SanitizePrompt(st); p != "" {
				return domain.GenerationParams{
					Engine:         domain.EngineUnknown,
					PositivePrompt: p,
					RawMetadata:    st,
					LoRAs:          extractLoRAs(p),
				}
			}
		}
	}

	// 5. SwarmUI "sui_image_params"
	if raw, ok := chunks["sui_image_params"]; ok && strings.TrimSpace(raw) != "" {
		if p, ok := parseJSONMetadata(raw); ok {
			p.Engine = domain.EngineSwarmUI
			return p
		}
	}

	// 6. InvokeAI "invokeai_metadata" or "sd-metadata"
	for _, k := range []string{"invokeai_metadata", "sd-metadata"} {
		if raw, ok := chunks[k]; ok && strings.TrimSpace(raw) != "" {
			if p, ok := parseJSONMetadata(raw); ok {
				p.Engine = domain.EngineInvokeAI
				return p
			}
		}
	}

	// 7. "generation_data"
	if gd, ok := chunks["generation_data"]; ok && strings.TrimSpace(gd) != "" {
		var obj interface{}
		if err := json.Unmarshal([]byte(gd), &obj); err == nil {
			if p := SanitizePrompt(FindPromptKey(obj)); p != "" {
				return domain.GenerationParams{
					Engine:         domain.EngineUnknown,
					PositivePrompt: p,
					RawMetadata:    gd,
					LoRAs:          extractLoRAs(p),
				}
			}
		}
	}

	// 8. "workflow"
	if wf, ok := chunks["workflow"]; ok && strings.TrimSpace(wf) != "" {
		var obj interface{}
		if err := json.Unmarshal([]byte(wf), &obj); err == nil {
			if p := SanitizePrompt(FindPromptKey(obj)); p != "" {
				return domain.GenerationParams{
					Engine:         domain.EngineUnknown,
					PositivePrompt: p,
					RawMetadata:    wf,
					LoRAs:          extractLoRAs(p),
				}
			}
		}
	}

	// 9. Fallback comment chunks
	for _, k := range []string{"Comment", "Description", "UserComment", "prompt"} {
		if raw, ok := chunks[k]; ok && strings.TrimSpace(raw) != "" {
			if params, found := ClassifyComment(raw); found && params.PositivePrompt != "" {
				return params
			}
		}
	}

	return domain.GenerationParams{Engine: domain.EngineUnknown}
}

// ParsePNGReader backward compatibility wrapper for existing tests
func InterpretTextChunks(chunks map[string]string) domain.GenerationParams {
	return InterpretPNGMetadata(chunks, nil)
}

// ParseComfyUIPromptGraph backward compatibility wrapper
func ParseComfyUIPromptGraph(rawPromptJSON, rawWorkflowJSON string) domain.GenerationParams {
	return ExtractComfyGraphParams(rawPromptJSON, rawWorkflowJSON)
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

func parseJSONMetadata(raw string) (domain.GenerationParams, bool) {
	var data map[string]interface{}
	if err := json.Unmarshal([]byte(raw), &data); err != nil {
		return domain.GenerationParams{}, false
	}

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
