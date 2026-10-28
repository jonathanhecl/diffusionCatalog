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
	EngineWanGP    EngineType = "wangp"
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

// MediaAsset represents a single cataloged file inside a watched source media root folder.
type MediaAsset struct {
	ID          int64            `json:"id"`
	RootID      int64            `json:"root_id"`
	RootPath    string           `json:"root_path,omitempty"`
	RootLabel   string           `json:"root_label,omitempty"`
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
	Favorite    bool             `json:"favorite"`
	Category    string           `json:"category,omitempty"`
	Subcategory string           `json:"subcategory,omitempty"`
	Metadata    GenerationParams `json:"metadata"`
	Tags        []Tag            `json:"tags"`
	Groups      []Group          `json:"groups"`
	Characters  []Character      `json:"characters"`
	Faces       []DetectedFace   `json:"faces"`
	FileMTime           time.Time        `json:"file_mtime"`
	IndexedAt           time.Time        `json:"indexed_at"`
	MetadataExtractedAt *time.Time       `json:"metadata_extracted_at,omitempty"`
	FacesScannedAt      *time.Time       `json:"faces_scanned_at,omitempty"`
}

// MediaRoot represents a registered source folder on any local directory or external disk/volume.
type MediaRoot struct {
	ID         int64             `json:"id"`
	Path       string            `json:"path"`
	Label      string            `json:"label"`
	Enabled    bool              `json:"enabled"`
	Accessible bool              `json:"accessible"`
	AssetCount int               `json:"asset_count"`
	Categories []DefaultCategory `json:"categories,omitempty"`
	CreatedAt  time.Time         `json:"created_at"`
}

// DefaultCategory defines an agnostic category and its subcategories stored globally in the DB
// and applied to each media root's .diffusion-catalog.json classification manifest.
type DefaultCategory struct {
	ID            int64    `json:"id,omitempty"`
	Name          string   `json:"name"`
	Facet         string   `json:"facet"`
	Color         string   `json:"color"`
	Subcategories []string `json:"subcategories"`
}

// ManifestTag represents a tag entry persisted inside a folder's .diffusion-catalog.json file.
type ManifestTag struct {
	Name  string `json:"name"`
	Facet string `json:"facet"`
	Color string `json:"color,omitempty"`
}

// ManifestAssetEntry stores portable per-asset classification and preprocessed metadata inside <media_root>/.diffusion-catalog.json.
type ManifestAssetEntry struct {
	Category    string            `json:"category,omitempty"`
	Subcategory string            `json:"subcategory,omitempty"`
	Rating      int               `json:"rating,omitempty"`
	Favorite    bool              `json:"favorite,omitempty"`
	Tags        []ManifestTag     `json:"tags,omitempty"`
	Groups      []string          `json:"groups,omitempty"`
	Characters  []string          `json:"characters,omitempty"`
	FileSize    int64             `json:"file_size,omitempty"`
	FileMTime   int64             `json:"file_mtime,omitempty"`
	FileHash    string            `json:"file_hash,omitempty"`
	Width       int               `json:"width,omitempty"`
	Height      int               `json:"height,omitempty"`
	DurationSec float64           `json:"duration_sec,omitempty"`
	Metadata    *GenerationParams `json:"metadata,omitempty"`
}

// FolderManifest represents the .diffusion-catalog.json file stored at the root of each media folder.
type FolderManifest struct {
	Version    int                           `json:"version"`
	FolderName string                        `json:"folder_name"`
	UpdatedAt  string                        `json:"updated_at"`
	Categories []DefaultCategory             `json:"categories"`
	Assets     map[string]ManifestAssetEntry `json:"assets"`
}
