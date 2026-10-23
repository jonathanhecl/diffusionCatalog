package usecase

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/metadata"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/storage/sqlite"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/vision"
	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

func TestEndToEndSeedScanAndCatalogQuery(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	mediaDir := filepath.Join(tmpDir, "media")
	cacheDir := filepath.Join(tmpDir, ".cache")
	dbPath := filepath.Join(tmpDir, "test_catalog.db")

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("failed to open sqlite db: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db)

	authSvc, err := NewAuthService(ctx, repo, "admin", "admin123")
	if err != nil {
		t.Fatalf("failed to bootstrap default admin: %v", err)
	}
	token, err := authSvc.Authenticate(ctx, "admin", "admin123")
	if err != nil || token == "" {
		t.Fatalf("expected default admin login to succeed, got err=%v", err)
	}

	extractor := metadata.NewExtractor()
	scannerSvc := NewScannerService(mediaDir, cacheDir, repo, extractor, vision.NewNoopEngine())
	catalogSvc := NewCatalogService(repo)

	if err := SeedDemoFolderIfEmpty(ctx, mediaDir, cacheDir, repo, scannerSvc); err != nil {
		t.Fatalf("failed to seed demo folder: %v", err)
	}

	page, err := catalogSvc.Query(ctx, domain.CatalogFilter{PerPage: 50})
	if err != nil {
		t.Fatalf("failed to query catalog: %v", err)
	}
	if page.TotalCount != 6 {
		t.Fatalf("expected 6 seeded diffusion assets, got %d", page.TotalCount)
	}
	if len(page.Characters) != 3 {
		t.Fatalf("expected 3 character folders, got %d", len(page.Characters))
	}
	if len(page.Groups) != 3 {
		t.Fatalf("expected 3 auto-indexed folder groups, got %d", len(page.Groups))
	}

	// Verify Face Explorer mode
	facePage, err := catalogSvc.Query(ctx, domain.CatalogFilter{FaceMode: true, PerPage: 50})
	if err != nil {
		t.Fatalf("failed to query face explorer mode: %v", err)
	}
	if facePage.TotalCount != 6 {
		t.Fatalf("expected 6 extracted face crops, got %d", facePage.TotalCount)
	}
}
