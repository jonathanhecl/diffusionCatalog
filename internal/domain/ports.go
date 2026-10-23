package domain

import "context"

// CatalogFilter specifies search, facet, and pagination criteria for querying the catalog.
type CatalogFilter struct {
	Query       string
	Engine      string
	ModelName   string
	MediaType   string
	GroupID     int64
	CharacterID int64
	TagNames    []string
	FaceMode    bool
	SortBy      string // "newest", "popular", "views", "seed"
	Page        int
	PerPage     int
}

// CatalogPage contains a paginated slice of assets along with total counts and facet aggregations.
type CatalogPage struct {
	Assets         []MediaAsset
	Faces          []DetectedFace
	TotalCount     int
	Page           int
	PerPage        int
	TotalPages     int
	AvailableTags  []Tag
	FacetTags      map[string][]Tag
	Engines        []string
	Models         []string
	Groups         []Group
	Characters     []Character
}

// CatalogRepository defines the persistence port for SQLite.
type CatalogRepository interface {
	EnsureDefaultAdmin(ctx context.Context, username, passwordHash string) error
	GetUserByUsername(ctx context.Context, username string) (*User, error)

	UpsertAsset(ctx context.Context, asset *MediaAsset) (int64, error)
	GetAssetByID(ctx context.Context, id int64) (*MediaAsset, error)
	QueryCatalog(ctx context.Context, filter CatalogFilter) (*CatalogPage, error)
	IncrementAssetView(ctx context.Context, id int64) error

	UpsertTag(ctx context.Context, name, facet, color string) (*Tag, error)
	AttachTagToAsset(ctx context.Context, assetID, tagID int64) error
	ListTags(ctx context.Context) ([]Tag, error)

	CreateGroup(ctx context.Context, group *Group) (int64, error)
	AttachAssetToGroup(ctx context.Context, groupID, assetID int64) error
	ListGroups(ctx context.Context) ([]Group, error)

	CreateCharacter(ctx context.Context, char *Character) (int64, error)
	GetCharacterByID(ctx context.Context, id int64) (*Character, error)
	AttachAssetToCharacter(ctx context.Context, characterID, assetID int64) error
	ListCharacters(ctx context.Context) ([]Character, error)

	SaveDetectedFace(ctx context.Context, face *DetectedFace) (int64, error)
	ListDetectedFaces(ctx context.Context, filter CatalogFilter) ([]DetectedFace, int, error)
}

// MetadataExtractor defines the port for extracting diffusion prompts and technical parameters from media files.
type MetadataExtractor interface {
	Extract(ctx context.Context, absPath string, mediaType MediaType) (GenerationParams, int, int, error)
}

// VisionEngine defines the agnostic port for face/subject detection, cropping, and vector embedding extraction.
// It can be backed by a local Python worker (InsightFace/ONNX), an external HTTP service, or a built-in detector.
type VisionEngine interface {
	Name() string
	DetectAndCropFaces(ctx context.Context, assetID int64, absImagePath, cacheDir string) ([]DetectedFace, error)
}
