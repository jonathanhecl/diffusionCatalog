package web

import (
	"context"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/metadata"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/storage/sqlite"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/vision"
	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
	"github.com/jonathanhecl/diffusionCatalog/internal/usecase"
)

func TestPageURL(t *testing.T) {
	tests := []struct {
		name     string
		filter   domain.CatalogFilter
		page     int
		contains []string
	}{
		{
			name:   "basic page 2",
			filter: domain.CatalogFilter{},
			page:   2,
			contains: []string{
				"page=2",
			},
		},
		{
			name: "preserves all query filters",
			filter: domain.CatalogFilter{
				Query:         "cyberpunk warrior",
				Trigger:       "kaelen7",
				RootID:        3,
				Category:      "Characters",
				Subcategory:   "Concept",
				MinRating:     8,
				FavoritesOnly: true,
				Engine:        "Flux",
				ModelName:     "flux1-dev.safetensors",
				MediaType:     "image",
				GroupID:       5,
				CharacterID:   2,
				TagNames:      []string{"cyberpunk", "neon"},
				FaceMode:      true,
				SortBy:        "rating",
				PerPage:       100,
			},
			page: 4,
			contains: []string{
				"page=4",
				"per_page=100",
				"q=cyberpunk+warrior",
				"trigger=kaelen7",
				"root_id=3",
				"category=Characters",
				"subcategory=Concept",
				"min_rating=8",
				"favorites=1",
				"engine=Flux",
				"model=flux1-dev.safetensors",
				"media_type=image",
				"group_id=5",
				"character_id=2",
				"tag=cyberpunk",
				"tag=neon",
				"face_mode=1",
				"sort=rating",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			url := pageURL(tt.filter, tt.page)
			for _, exp := range tt.contains {
				if !strings.Contains(url, exp) {
					t.Errorf("pageURL() = %q, expected to contain %q", url, exp)
				}
			}
		})
	}
}

func TestSortURL(t *testing.T) {
	filter := domain.CatalogFilter{
		Query:   "portrait",
		SortBy:  "rating",
		Page:    5,
		PerPage: 25,
	}

	url := sortURL(filter, "favorites")
	if !strings.Contains(url, "sort=favorites") {
		t.Errorf("sortURL() expected sort=favorites, got %q", url)
	}
	if !strings.Contains(url, "page=1") {
		t.Errorf("sortURL() expected page to reset to 1, got %q", url)
	}
	if !strings.Contains(url, "q=portrait") {
		t.Errorf("sortURL() expected query filter to be retained, got %q", url)
	}
}

func TestAssetDetailURL(t *testing.T) {
	filter := domain.CatalogFilter{
		Query: "scifi",
		Page:  3,
	}
	url := assetDetailURL(filter, 42)
	if !strings.Contains(url, "asset_id=42") {
		t.Errorf("assetDetailURL() expected asset_id=42, got %q", url)
	}
	if !strings.Contains(url, "page=3") {
		t.Errorf("assetDetailURL() expected page=3, got %q", url)
	}
}

func TestPaginationList(t *testing.T) {
	t.Run("single page returns nil", func(t *testing.T) {
		items := paginationList(1, 1)
		if len(items) != 0 {
			t.Errorf("expected 0 items for total=1, got %d", len(items))
		}
	})

	t.Run("small total 3 pages", func(t *testing.T) {
		items := paginationList(2, 3)
		if len(items) != 3 {
			t.Fatalf("expected 3 items, got %d", len(items))
		}
		if items[0].Page != 1 || items[1].Page != 2 || items[2].Page != 3 {
			t.Errorf("unexpected pages: %+v", items)
		}
		if !items[1].IsActive {
			t.Errorf("expected page 2 to be active")
		}
	})

	t.Run("large total 15 pages current 1", func(t *testing.T) {
		items := paginationList(1, 15)
		// Should include 1, 2, 3, 4, 5, gap, 15
		var labels []string
		for _, it := range items {
			labels = append(labels, it.Label)
		}
		expected := []string{"1", "2", "3", "4", "5", "…", "15"}
		if len(labels) != len(expected) {
			t.Fatalf("expected %v, got %v", expected, labels)
		}
		for i := range expected {
			if labels[i] != expected[i] {
				t.Errorf("at index %d expected %q, got %q", i, expected[i], labels[i])
			}
		}
		if !items[0].IsActive {
			t.Errorf("expected page 1 to be active")
		}
	})

	t.Run("large total 15 pages current 8", func(t *testing.T) {
		items := paginationList(8, 15)
		// Should include 1, gap, 6, 7, 8, 9, 10, gap, 15
		var labels []string
		for _, it := range items {
			labels = append(labels, it.Label)
		}
		expected := []string{"1", "…", "6", "7", "8", "9", "10", "…", "15"}
		if len(labels) != len(expected) {
			t.Fatalf("expected %v, got %v", expected, labels)
		}
		for i := range expected {
			if labels[i] != expected[i] {
				t.Errorf("at index %d expected %q, got %q", i, expected[i], labels[i])
			}
		}
	})

	t.Run("large total 15 pages current 15", func(t *testing.T) {
		items := paginationList(15, 15)
		// Should include 1, gap, 11, 12, 13, 14, 15
		var labels []string
		for _, it := range items {
			labels = append(labels, it.Label)
		}
		expected := []string{"1", "…", "11", "12", "13", "14", "15"}
		if len(labels) != len(expected) {
			t.Fatalf("expected %v, got %v", expected, labels)
		}
		for i := range expected {
			if labels[i] != expected[i] {
				t.Errorf("at index %d expected %q, got %q", i, expected[i], labels[i])
			}
		}
		if !items[len(items)-1].IsActive {
			t.Errorf("expected last item to be active")
		}
	})
}

func TestParseCatalogFilter(t *testing.T) {
	req := httptest.NewRequest("GET", "/?page=3&per_page=25&q=robot&tag=scifi&tag=mecha&face_mode=1&sort=rating", nil)
	f := parseCatalogFilter(req)
	if f.Page != 3 {
		t.Errorf("expected Page 3, got %d", f.Page)
	}
	if f.PerPage != 25 {
		t.Errorf("expected PerPage 25, got %d", f.PerPage)
	}
	if f.Query != "robot" {
		t.Errorf("expected Query robot, got %q", f.Query)
	}
	if len(f.TagNames) != 2 || f.TagNames[0] != "scifi" || f.TagNames[1] != "mecha" {
		t.Errorf("expected TagNames [scifi, mecha], got %v", f.TagNames)
	}
	if !f.FaceMode {
		t.Errorf("expected FaceMode true")
	}
	if f.SortBy != "rating" {
		t.Errorf("expected SortBy rating, got %q", f.SortBy)
	}
}

func TestDirectorTabRendering(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	mediaDir := filepath.Join(tmpDir, "media")
	cacheDir := filepath.Join(tmpDir, "cache")
	dbPath := filepath.Join(tmpDir, "test.db")

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("failed to open sqlite: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db)
	authSvc, _ := usecase.NewAuthService(ctx, repo, "admin", "admin123")
	extractor := metadata.NewExtractor()
	scannerSvc := usecase.NewScannerService(mediaDir, cacheDir, repo, extractor, vision.NewNoopEngine())
	catalogSvc := usecase.NewCatalogService(mediaDir, repo, nil)

	srv, err := NewServer(mediaDir, cacheDir, "NoopVision", catalogSvc, scannerSvc, authSvc)
	if err != nil {
		t.Fatalf("failed to construct web server: %v", err)
	}

	req := httptest.NewRequest("GET", "/?tab=director", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("expected HTTP 200, got %d", rec.Code)
	}

	body := rec.Body.String()
	if !strings.Contains(body, "director-container") {
		t.Errorf("expected body to contain director-container, got:\n%s", body[:min(len(body), 500)])
	}
	if !strings.Contains(body, "directorSvg") {
		t.Errorf("expected body to contain directorSvg canvas element")
	}
	if !strings.Contains(body, "CHARACTERS &amp; FIGURES") {
		t.Errorf("expected body to contain character palette")
	}
	if !strings.Contains(body, "aspect-btn") {
		t.Errorf("expected body to contain aspect ratio buttons")
	}
	if strings.Contains(body, `Search prompts, trigger words, checkpoints`) {
		t.Errorf("director tab should not render the catalog search grid!")
	}
}
