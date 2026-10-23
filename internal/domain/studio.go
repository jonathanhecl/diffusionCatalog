package domain

import "time"

// StudioComposition represents a saved 2D/2.5D visual staging scene composition.
type StudioComposition struct {
	ID            int64     `json:"id"`
	Name          string    `json:"name"`
	Description   string    `json:"description"`
	SceneJSON     string    `json:"scene_json"`
	ThumbnailData string    `json:"thumbnail_data"`
	PromptOutput  string    `json:"prompt_output"`
	CreatedAt     time.Time `json:"created_at"`
	UpdatedAt     time.Time `json:"updated_at"`
}

// StudioPose represents a saved skeletal character pose preset.
type StudioPose struct {
	ID          int64     `json:"id"`
	Name        string    `json:"name"`
	Category    string    `json:"category"`
	PoseJSON    string    `json:"pose_json"`
	Description string    `json:"description"`
	CreatedAt   time.Time `json:"created_at"`
}
