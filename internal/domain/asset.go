package domain

import "time"

// MediaType distinguishes static diffusion images from generated videos.
type MediaType string

const (
	MediaTypeImage MediaType = "image"
	MediaTypeVideo MediaType = "video"
)

// EngineType identifies the diffusion generator or workflow tool that created an asset.
type EngineType string

const (
	EngineA1111    EngineType = "a1111"
	EngineComfyUI  EngineType = "comfyui"
	EngineFooocus  EngineType = "fooocus"
	EngineInvokeAI EngineType = "invokeai"
	EngineSwarmUI  EngineType = "swarmui"
	EngineUnknown  EngineType = "unknown"
)

// GenerationParams stores normalized diffusion parameters extracted from image/video metadata.
type GenerationParams struct {
	Engine         EngineType `json:"engine"`
	PositivePrompt string     `json:"positive_prompt"`
	NegativePrompt string     `json:"negative_prompt"`
	ModelName      string     `json:"model_name"`
	ModelHash      string     `json:"model_hash"`
	Sampler        string     `json:"sampler"`
	Scheduler      string     `json:"scheduler"`
	Steps          int        `json:"steps"`
	CFGScale       float64    `json:"cfg_scale"`
	Seed           int64      `json:"seed"`
	LoRAs          []string   `json:"loras"`
	RawMetadata    string     `json:"raw_metadata"`
}

// MediaAsset represents a single cataloged file inside the watched source folder.
type MediaAsset struct {
	ID          int64            `json:"id"`
	RelPath     string           `json:"rel_path"`
	FileName    string           `json:"file_name"`
	FileHash    string           `json:"file_hash"`
	MediaType   MediaType        `json:"media_type"`
	Width       int              `json:"width"`
	Height      int              `json:"height"`
	DurationSec float64          `json:"duration_sec"`
	FileSize    int64            `json:"file_size"`
	ViewCount   int64            `json:"view_count"`
	Rating      int              `json:"rating"`
	Metadata    GenerationParams `json:"metadata"`
	Tags        []Tag            `json:"tags"`
	Groups      []Group          `json:"groups"`
	Characters  []Character      `json:"characters"`
	Faces       []DetectedFace   `json:"faces"`
	IndexedAt   time.Time        `json:"indexed_at"`
}
