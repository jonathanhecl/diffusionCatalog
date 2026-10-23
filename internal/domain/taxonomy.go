package domain

// Tag represents an agnostic label organized under a user-defined facet (e.g., "style", "subject", "lighting", "mood", "general").
type Tag struct {
	ID         int64  `json:"id"`
	Name       string `json:"name"`
	Facet      string `json:"facet"`
	Color      string `json:"color"`
	AssetCount int    `json:"asset_count"`
}

// Group represents an agnostic collection or folder of media assets.
type Group struct {
	ID           int64  `json:"id"`
	Name         string `json:"name"`
	Slug         string `json:"slug"`
	Description  string `json:"description"`
	Category     string `json:"category"`
	CoverAssetID *int64 `json:"cover_asset_id,omitempty"`
	AssetCount   int    `json:"asset_count"`
}
