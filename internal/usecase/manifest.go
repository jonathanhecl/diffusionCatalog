package usecase

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

// ManifestFileName is the portable per-folder classification file created inside each media root.
const ManifestFileName = ".diffusion-catalog.json"

// EnsureFolderManifest loads <rootPath>/.diffusion-catalog.json if it exists, or initializes it
// using the global default categories from the database if the folder does not have one yet.
func EnsureFolderManifest(rootPath, label string, defaults []domain.DefaultCategory) (*domain.FolderManifest, error) {
	manifestPath := filepath.Join(rootPath, ManifestFileName)
	data, err := os.ReadFile(manifestPath)
	if err == nil && len(data) > 0 {
		var m domain.FolderManifest
		if uErr := json.Unmarshal(data, &m); uErr == nil {
			if m.Assets == nil {
				m.Assets = make(map[string]domain.ManifestAssetEntry)
			}
			if len(m.Categories) == 0 && len(defaults) > 0 {
				m.Categories = copyDefaultCategories(defaults)
				_ = SaveFolderManifest(rootPath, &m)
			}
			return &m, nil
		}
	}

	if label == "" {
		label = filepath.Base(rootPath)
	}
	m := &domain.FolderManifest{
		Version:    1,
		FolderName: label,
		UpdatedAt:  time.Now().UTC().Format(time.RFC3339),
		Categories: copyDefaultCategories(defaults),
		Assets:     make(map[string]domain.ManifestAssetEntry),
	}
	_ = SaveFolderManifest(rootPath, m)
	return m, nil
}

// SaveFolderManifest writes the FolderManifest to <rootPath>/.diffusion-catalog.json.
func SaveFolderManifest(rootPath string, m *domain.FolderManifest) error {
	if strings.TrimSpace(rootPath) == "" || m == nil {
		return nil
	}
	if m.Version == 0 {
		m.Version = 1
	}
	if m.Assets == nil {
		m.Assets = make(map[string]domain.ManifestAssetEntry)
	}
	m.UpdatedAt = time.Now().UTC().Format(time.RFC3339)

	raw, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	manifestPath := filepath.Join(rootPath, ManifestFileName)
	return os.WriteFile(manifestPath, append(raw, '\n'), 0644)
}

// SyncAssetToFolderManifest updates a single asset's portable classification entry inside its root folder's .diffusion-catalog.json.
func SyncAssetToFolderManifest(rootPath string, asset *domain.MediaAsset, defaults []domain.DefaultCategory) error {
	if strings.TrimSpace(rootPath) == "" || asset == nil {
		return nil
	}
	m, err := EnsureFolderManifest(rootPath, asset.RootLabel, defaults)
	if err != nil || m == nil {
		return err
	}

	key := normalizeManifestRelPath(rootPath, asset.RelPath)
	entry := m.Assets[key]
	if asset.Category != "" {
		entry.Category = asset.Category
	}
	if asset.Subcategory != "" {
		entry.Subcategory = asset.Subcategory
	}
	entry.Rating = asset.Rating
	entry.Favorite = asset.Favorite

	if len(asset.Tags) > 0 {
		var tags []domain.ManifestTag
		seen := make(map[string]bool)
		for _, t := range asset.Tags {
			k := strings.ToLower(t.Name)
			if !seen[k] {
				seen[k] = true
				tags = append(tags, domain.ManifestTag{
					Name:  t.Name,
					Facet: t.Facet,
					Color: t.Color,
				})
			}
		}
		entry.Tags = tags
	}

	if len(asset.Groups) > 0 {
		var groups []string
		seen := make(map[string]bool)
		for _, g := range asset.Groups {
			if !seen[g.Name] {
				seen[g.Name] = true
				groups = append(groups, g.Name)
			}
		}
		entry.Groups = groups
	}

	if len(asset.Characters) > 0 {
		var chars []string
		seen := make(map[string]bool)
		for _, c := range asset.Characters {
			if !seen[c.Name] {
				seen[c.Name] = true
				chars = append(chars, c.Name)
			}
		}
		entry.Characters = chars
	}

	m.Assets[key] = entry

	// Ensure assigned category/subcategory is also listed in the folder's categories list
	if asset.Category != "" {
		ensureManifestCategory(m, asset.Category, asset.Subcategory)
	}

	return SaveFolderManifest(rootPath, m)
}

func ensureManifestCategory(m *domain.FolderManifest, category, subcategory string) {
	category = strings.TrimSpace(category)
	subcategory = strings.TrimSpace(subcategory)
	if category == "" {
		return
	}
	for i := range m.Categories {
		if strings.EqualFold(m.Categories[i].Name, category) {
			if subcategory != "" {
				foundSub := false
				for _, s := range m.Categories[i].Subcategories {
					if strings.EqualFold(s, subcategory) {
						foundSub = true
						break
					}
				}
				if !foundSub {
					m.Categories[i].Subcategories = append(m.Categories[i].Subcategories, subcategory)
				}
			}
			return
		}
	}
	var subs []string
	if subcategory != "" {
		subs = []string{subcategory}
	}
	m.Categories = append(m.Categories, domain.DefaultCategory{
		Name:          category,
		Facet:         "category",
		Color:         "#2563eb",
		Subcategories: subs,
	})
}

func normalizeManifestRelPath(rootPath, relPath string) string {
	// Strip optional "@root_N/" storage prefix if present
	p := filepath.ToSlash(relPath)
	if strings.HasPrefix(p, "@root_") {
		if idx := strings.IndexByte(p, '/'); idx > 0 {
			p = p[idx+1:]
		}
	}
	return strings.TrimPrefix(p, "/")
}

func copyDefaultCategories(defaults []domain.DefaultCategory) []domain.DefaultCategory {
	out := make([]domain.DefaultCategory, 0, len(defaults))
	for _, c := range defaults {
		subs := make([]string, len(c.Subcategories))
		copy(subs, c.Subcategories)
		out = append(out, domain.DefaultCategory{
			Name:          c.Name,
			Facet:         c.Facet,
			Color:         c.Color,
			Subcategories: subs,
		})
	}
	return out
}
