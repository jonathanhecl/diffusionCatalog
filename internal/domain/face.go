package domain

import "time"

// BoundingBox defines a normalized or pixel rectangle inside an image or video frame.
type BoundingBox struct {
	X      int `json:"x"`
	Y      int `json:"y"`
	Width  int `json:"width"`
	Height int `json:"height"`
}

// DetectedFace represents an extracted face or subject region with optional embedding vector for similarity search.
type DetectedFace struct {
	ID            int64       `json:"id"`
	AssetID       int64       `json:"asset_id"`
	AssetRelPath  string      `json:"asset_rel_path"`
	AssetPrompt   string      `json:"asset_prompt"`
	AssetEngine   EngineType  `json:"asset_engine"`
	AssetModel    string      `json:"asset_model"`
	Box           BoundingBox `json:"box"`
	CropRelPath   string      `json:"crop_rel_path"`
	Confidence    float64     `json:"confidence"`
	Embedding     []float32   `json:"embedding,omitempty"`
	ClusterLabel  string      `json:"cluster_label"`
	ExpressionTag string      `json:"expression_tag"`
	CharacterID   *int64      `json:"character_id,omitempty"`
	CharacterName string      `json:"character_name,omitempty"`
	CreatedAt     time.Time   `json:"created_at"`
}
