package domain

import "time"

// Character represents a reusable subject/persona sheet ("ficha de personaje") containing
// physical appearance descriptions, facial/body traits, and optional reference images (0-4).
type Character struct {
	ID                    int64        `json:"id"`
	Name                  string       `json:"name"`
	Slug                  string       `json:"slug"`
	Archetype             string       `json:"archetype"`
	Description           string       `json:"description"`
	TriggerWords          string       `json:"trigger_words"`
	BasePositivePrompt    string       `json:"base_positive_prompt"`
	BaseNegativePrompt    string       `json:"base_negative_prompt"`
	RecommendedCheckpoint string       `json:"recommended_checkpoint"`
	LoRAs                 []string     `json:"loras"`
	AvatarAssetID         *int64       `json:"avatar_asset_id,omitempty"`
	ReferenceImages       []string     `json:"reference_images"`
	ExampleCount          int          `json:"example_count"`
	Examples              []MediaAsset `json:"examples,omitempty"`
	CreatedAt             time.Time    `json:"created_at"`
}
