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
	AssetRootID   int64       `json:"asset_root_id,omitempty"`
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

// FaceCluster represents an aggregated identity group (e.g. Person #1, Elena) with its best cropped face cover.
type FaceCluster struct {
	Label         string         `json:"label"`
	CharacterID   *int64         `json:"character_id,omitempty"`
	CharacterName string         `json:"character_name,omitempty"`
	FaceCount     int            `json:"face_count"`
	AssetCount    int            `json:"asset_count"`
	BestCropPath  string         `json:"best_crop_path"`
	Faces         []DetectedFace `json:"faces,omitempty"`
}
