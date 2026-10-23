package usecase

import (
	"context"
	"strings"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

// CatalogService coordinates catalog querying, agnostic tagging, grouping, and character sheets.
type CatalogService struct {
	repo domain.CatalogRepository
}

// NewCatalogService creates a new CatalogService.
func NewCatalogService(repo domain.CatalogRepository) *CatalogService {
	return &CatalogService{repo: repo}
}

// Query returns a paginated, facet-filtered catalog page.
func (s *CatalogService) Query(ctx context.Context, filter domain.CatalogFilter) (*domain.CatalogPage, error) {
	return s.repo.QueryCatalog(ctx, filter)
}

// GetAsset fetches an asset by ID and increments its view count.
func (s *CatalogService) GetAsset(ctx context.Context, id int64) (*domain.MediaAsset, error) {
	_ = s.repo.IncrementAssetView(ctx, id)
	return s.repo.GetAssetByID(ctx, id)
}

// AddTagToAsset creates or updates an agnostic tag and attaches it to an asset.
func (s *CatalogService) AddTagToAsset(ctx context.Context, assetID int64, name, facet string) error {
	name = strings.TrimSpace(strings.TrimPrefix(name, "#"))
	if name == "" {
		return nil
	}
	if facet == "" {
		facet = "general"
	}
	t, err := s.repo.UpsertTag(ctx, name, strings.ToLower(strings.TrimSpace(facet)), "#2563eb")
	if err != nil {
		return err
	}
	return s.repo.AttachTagToAsset(ctx, assetID, t.ID)
}

// CreateGroup creates a new agnostic group and optionally links an initial asset.
func (s *CatalogService) CreateGroup(ctx context.Context, name, category, description string, initialAssetID int64) (int64, error) {
	g := &domain.Group{
		Name:        strings.TrimSpace(name),
		Category:    strings.TrimSpace(category),
		Description: strings.TrimSpace(description),
	}
	if initialAssetID > 0 {
		g.CoverAssetID = &initialAssetID
	}
	gid, err := s.repo.CreateGroup(ctx, g)
	if err != nil {
		return 0, err
	}
	if initialAssetID > 0 {
		_ = s.repo.AttachAssetToGroup(ctx, gid, initialAssetID)
	}
	return gid, nil
}

// AssignAssetToGroup attaches an existing asset to a group.
func (s *CatalogService) AssignAssetToGroup(ctx context.Context, groupID, assetID int64) error {
	return s.repo.AttachAssetToGroup(ctx, groupID, assetID)
}

// ListGroups returns all agnostic groups.
func (s *CatalogService) ListGroups(ctx context.Context) ([]domain.Group, error) {
	return s.repo.ListGroups(ctx)
}

// CreateCharacter creates a character/concept folder with its canonical prompts and optional example asset.
func (s *CatalogService) CreateCharacter(ctx context.Context, c *domain.Character, exampleAssetID int64) (int64, error) {
	if exampleAssetID > 0 && c.AvatarAssetID == nil {
		c.AvatarAssetID = &exampleAssetID
	}
	cid, err := s.repo.CreateCharacter(ctx, c)
	if err != nil {
		return 0, err
	}
	if exampleAssetID > 0 {
		_ = s.repo.AttachAssetToCharacter(ctx, cid, exampleAssetID)
	}
	return cid, nil
}

// AssignAssetToCharacter links an asset as a reference example for a character.
func (s *CatalogService) AssignAssetToCharacter(ctx context.Context, characterID, assetID int64) error {
	return s.repo.AttachAssetToCharacter(ctx, characterID, assetID)
}

// GetCharacter retrieves a character profile and its example gallery.
func (s *CatalogService) GetCharacter(ctx context.Context, id int64) (*domain.Character, error) {
	return s.repo.GetCharacterByID(ctx, id)
}

// ListCharacters returns all character profiles.
func (s *CatalogService) ListCharacters(ctx context.Context) ([]domain.Character, error) {
	return s.repo.ListCharacters(ctx)
}
