package domain

import "context"

// CatalogFilter specifies search, facet, and pagination criteria for querying the catalog.
type CatalogFilter struct {
	Query         string
	Trigger       string
	RootID        int64
	Category      string
	Subcategory   string
	MinRating     int
	FavoritesOnly bool
	Engine        string
	ModelName     string
	MediaType     string
	GroupID       int64
	CharacterID   int64
	TagNames      []string
	FaceMode      bool
	SortBy        string // "newest", "rating", "favorites", "popular", "views", "seed"
	Page          int
	PerPage       int
}

// CatalogPage contains a paginated slice of assets along with total counts and facet aggregations.
type CatalogPage struct {
	Assets            []MediaAsset
	Faces             []DetectedFace
	TotalCount        int
	Page              int
	PerPage           int
	TotalPages        int
	MediaRoots        []MediaRoot
	DefaultCategories []DefaultCategory
	AvailableTriggers []string
	AvailableTags     []Tag
	FacetTags         map[string][]Tag
	Engines           []string
	Models            []string
	Groups            []Group
	Characters        []Character
}

// CatalogRepository defines the persistence port for SQLite.
type CatalogRepository interface {
	EnsureDefaultAdmin(ctx context.Context, username, passwordHash string) error
	GetUserByUsername(ctx context.Context, username string) (*User, error)

	CurrentDBPath() string
	SwitchDB(ctx context.Context, newDBPath string) error

	UpsertMediaRoot(ctx context.Context, root *MediaRoot) (int64, error)
	DeleteMediaRoot(ctx context.Context, id int64) error
	GetMediaRootByID(ctx context.Context, id int64) (*MediaRoot, error)
	ListMediaRoots(ctx context.Context) ([]MediaRoot, error)

	UpsertDefaultCategory(ctx context.Context, cat *DefaultCategory) (int64, error)
	DeleteDefaultCategory(ctx context.Context, id int64) error
	ListDefaultCategories(ctx context.Context) ([]DefaultCategory, error)

	UpsertAsset(ctx context.Context, asset *MediaAsset) (int64, error)
	DeleteAsset(ctx context.Context, id int64) error
	DeleteAssetsByIDs(ctx context.Context, ids []int64) error
	GetAssetByID(ctx context.Context, id int64) (*MediaAsset, error)
	GetExistingAssetFingerprints(ctx context.Context, rootID int64) (map[string]MediaAsset, error)
	GetAdjacentAssetIDs(ctx context.Context, id int64) (prevID int64, nextID int64, err error)
	UpdateAssetPrompt(ctx context.Context, assetID int64, positivePrompt, engine string) error
	UpdateAssetCategory(ctx context.Context, assetID int64, category, subcategory string) error
	UpdateAssetRating(ctx context.Context, assetID int64, rating int, favorite bool) error
	QueryCatalog(ctx context.Context, filter CatalogFilter) (*CatalogPage, error)
	IncrementAssetView(ctx context.Context, id int64) error

	UpsertTag(ctx context.Context, name, facet, color string) (*Tag, error)
	AttachTagToAsset(ctx context.Context, assetID, tagID int64) error
	DetachTagFromAsset(ctx context.Context, assetID, tagID int64) error
	ListTags(ctx context.Context) ([]Tag, error)

	CreateGroup(ctx context.Context, group *Group) (int64, error)
	AttachAssetToGroup(ctx context.Context, groupID, assetID int64) error
	DetachAssetFromGroup(ctx context.Context, groupID, assetID int64) error
	ListGroups(ctx context.Context) ([]Group, error)

	CreateCharacter(ctx context.Context, char *Character) (int64, error)
	GetCharacterByID(ctx context.Context, id int64) (*Character, error)
	AttachAssetToCharacter(ctx context.Context, characterID, assetID int64) error
	DetachAssetFromCharacter(ctx context.Context, characterID, assetID int64) error
	ListCharacters(ctx context.Context) ([]Character, error)

	SaveDetectedFace(ctx context.Context, face *DetectedFace) (int64, error)
	ListDetectedFaces(ctx context.Context, filter CatalogFilter) ([]DetectedFace, int, error)

	GetSettings(ctx context.Context) (AppSettings, error)
	SaveSettings(ctx context.Context, settings AppSettings) error
	SetDefaultTargetGenerator(ctx context.Context, slug string) error

	UpsertPromptProfile(ctx context.Context, profile *PromptProfile) (int64, error)
	DeletePromptProfile(ctx context.Context, id int64) error
	GetPromptProfileBySlug(ctx context.Context, slug string) (*PromptProfile, error)
	ListPromptProfiles(ctx context.Context) ([]PromptProfile, error)

	UpsertModelLoRA(ctx context.Context, lora *ModelLoRA) (int64, error)
	DeleteModelLoRA(ctx context.Context, id int64) error
	ListModelLoRAs(ctx context.Context, profileSlug string) ([]ModelLoRA, error)
	GetModelLoRAsByIDs(ctx context.Context, ids []int64) ([]ModelLoRA, error)

	SaveGeneratedPrompt(ctx context.Context, gp *GeneratedPrompt) (int64, error)
	ListGeneratedPrompts(ctx context.Context, assetID, characterID int64, limit int) ([]GeneratedPrompt, error)

	SaveStudioComposition(ctx context.Context, comp *StudioComposition) (int64, error)
	GetStudioCompositionByID(ctx context.Context, id int64) (*StudioComposition, error)
	ListStudioCompositions(ctx context.Context) ([]StudioComposition, error)
	DeleteStudioComposition(ctx context.Context, id int64) error

	UpsertStudioPose(ctx context.Context, pose *StudioPose) (int64, error)
	ListStudioPoses(ctx context.Context) ([]StudioPose, error)
	DeleteStudioPose(ctx context.Context, id int64) error
}

// MetadataExtractor defines the port for extracting diffusion prompts and technical parameters from media files.
type MetadataExtractor interface {
	Extract(ctx context.Context, absPath string, mediaType MediaType) (GenerationParams, int, int, error)
}

// VisionEngine defines the agnostic port for face/subject detection, cropping, and vector embedding extraction.
type VisionEngine interface {
	Name() string
	DetectAndCropFaces(ctx context.Context, assetID int64, absImagePath, cacheDir string) ([]DetectedFace, error)
}

// OllamaProvider defines the port for interacting with local or remote Ollama and oMLX (OpenAI-compatible MLX)
// inference servers for capability discovery, vision analysis, image2prompt, and text2prompt generation.
type OllamaProvider interface {
	CheckStatus(ctx context.Context, baseURL string) (version string, online bool)
	ListModels(ctx context.Context, baseURL string) ([]OllamaModelInfo, error)

	CheckOMLXStatus(ctx context.Context, baseURL, apiKey string) (online bool, modelCount int)
	ListOMLXModels(ctx context.Context, baseURL, apiKey string) ([]OllamaModelInfo, error)

	ImageToPrompt(ctx context.Context, provider AIProviderType, baseURL, apiKey, model, systemPrompt, absImagePath, userHint, thinkLevel string) (string, error)
	TextToPrompt(ctx context.Context, provider AIProviderType, baseURL, apiKey, model, systemPrompt, inputConcept, thinkLevel string) (string, error)
}
