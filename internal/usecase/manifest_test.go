package usecase

import (
	"context"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/metadata"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/storage/sqlite"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/vision"
	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

func TestMultiRootManifestSidecarAndDBSwitch(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()

	db1Path := filepath.Join(tmpDir, "global_one.db")
	db2Path := filepath.Join(tmpDir, "global_two.db")
	root1Dir := filepath.Join(tmpDir, "disk_one_media")
	root2Dir := filepath.Join(tmpDir, "disk_two_media")
	cacheDir := filepath.Join(tmpDir, "cache")

	_ = os.MkdirAll(root1Dir, 0755)
	_ = os.MkdirAll(root2Dir, 0755)

	db, err := sqlite.Open(ctx, db1Path)
	if err != nil {
		t.Fatalf("failed to open db1: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db, db1Path)
	extractor := metadata.NewExtractor()
	scannerSvc := NewScannerService(root1Dir, cacheDir, repo, extractor, vision.NewNoopEngine())
	catalogSvc := NewCatalogService(root1Dir, repo, nil)

	// 1. Seed demo files in root1Dir and verify .txt sidecars and .diffusion-catalog.json are created
	if err := SeedDemoFolderIfEmpty(ctx, root1Dir, cacheDir, repo, scannerSvc); err != nil {
		t.Fatalf("SeedDemoFolderIfEmpty failed: %v", err)
	}

	manifest1Path := filepath.Join(root1Dir, ManifestFileName)
	if _, err := os.Stat(manifest1Path); err != nil {
		t.Fatalf("expected %s to exist in root1Dir: %v", ManifestFileName, err)
	}

	txtSidecar := filepath.Join(root1Dir, "Cyberpunk_Chronicles", "01_kaelen_neon_visor.txt")
	if _, err := os.Stat(txtSidecar); err != nil {
		t.Fatalf("expected auto-created .txt sidecar at %s: %v", txtSidecar, err)
	}

	// 2. Create a second media folder on another path with a .txt sidecar + image, register it, and scan
	img2Path := filepath.Join(root2Dir, "mecha_test.png")
	txt2Path := filepath.Join(root2Dir, "mecha_test.txt")
	img := renderSyntheticArtwork(200, 200, color.RGBA{20, 30, 40, 255}, color.RGBA{100, 200, 250, 255}, 1)
	imgFile, err := os.Create(img2Path)
	if err != nil {
		t.Fatalf("create img2: %v", err)
	}
	if err := png.Encode(imgFile, img); err != nil {
		imgFile.Close()
		t.Fatalf("encode img2: %v", err)
	}
	imgFile.Close()
	if err := os.WriteFile(txt2Path, []byte("orbital mech drop from txt sidecar\nSteps: 20, Seed: 12345, Model: flux1-dev\n"), 0644); err != nil {
		t.Fatalf("write txt2: %v", err)
	}

	root2ID, err := catalogSvc.AddMediaRoot(ctx, root2Dir, "External SSD #2")
	if err != nil || root2ID <= 1 {
		t.Fatalf("AddMediaRoot failed: id=%d err=%v", root2ID, err)
	}

	report, err := scannerSvc.Scan(ctx)
	if err != nil {
		t.Fatalf("Scan failed: %v", err)
	}
	if report.ScannedRoots < 2 {
		t.Errorf("expected at least 2 scanned roots, got %d", report.ScannedRoots)
	}

	// Verify filtering by RootID = root2ID returns the asset with the prompt loaded from mecha_test.txt
	pageR2, err := catalogSvc.Query(ctx, domain.CatalogFilter{RootID: root2ID})
	if err != nil {
		t.Fatalf("Query root2 failed: %v", err)
	}
	if pageR2.TotalCount != 1 || len(pageR2.Assets) != 1 {
		t.Fatalf("expected 1 asset in root2, got %d", pageR2.TotalCount)
	}
	assetR2 := pageR2.Assets[0]
	if !strings.Contains(assetR2.Metadata.PositivePrompt, "orbital mech drop from txt sidecar") {
		t.Errorf("expected prompt from .txt sidecar, got %q", assetR2.Metadata.PositivePrompt)
	}

	// 3. Classify the asset, tag it, and rate it (score 9/10 + favorite); verify it persists to root2Dir/.diffusion-catalog.json
	if err := catalogSvc.SetAssetCategory(ctx, assetR2.ID, "Mecha, Vehicles & Props", "Heavy Orbital Mecha"); err != nil {
		t.Fatalf("SetAssetCategory failed: %v", err)
	}
	if err := catalogSvc.AddTagToAsset(ctx, assetR2.ID, "orbital-drop-v2", "action"); err != nil {
		t.Fatalf("AddTagToAsset failed: %v", err)
	}
	if err := catalogSvc.RateAsset(ctx, assetR2.ID, 9, true); err != nil {
		t.Fatalf("RateAsset failed: %v", err)
	}

	manifest2Path := filepath.Join(root2Dir, ManifestFileName)
	manifestData, err := os.ReadFile(manifest2Path)
	if err != nil {
		t.Fatalf("failed to read root2 manifest: %v", err)
	}
	if !strings.Contains(string(manifestData), `"rating": 9`) || !strings.Contains(string(manifestData), `"favorite": true`) {
		t.Fatalf("expected manifest to contain rating 9 and favorite true, got: %s", string(manifestData))
	}

	// 4. Switch to a brand new global SQLite database (db2Path), register root2Dir, scan, and verify classifications, ratings, and favorites are restored from .diffusion-catalog.json!
	if err := catalogSvc.SwitchDatabase(ctx, db2Path); err != nil {
		t.Fatalf("SwitchDatabase failed: %v", err)
	}
	if catalogSvc.CurrentDBPath() != db2Path {
		t.Errorf("expected CurrentDBPath %s, got %s", db2Path, catalogSvc.CurrentDBPath())
	}

	_, _ = catalogSvc.AddMediaRoot(ctx, root2Dir, "External SSD #2 Restored")
	if _, err := scannerSvc.Scan(ctx); err != nil {
		t.Fatalf("Scan on db2 failed: %v", err)
	}

	restoredPage, err := catalogSvc.Query(ctx, domain.CatalogFilter{Category: "Mecha, Vehicles & Props", MinRating: 8, FavoritesOnly: true})
	if err != nil {
		t.Fatalf("Query on db2 failed: %v", err)
	}
	if len(restoredPage.Assets) != 1 {
		t.Fatalf("expected 1 asset restored with rating >= 8 and favorites_only in db2, got %d", len(restoredPage.Assets))
	}
	if restoredPage.Assets[0].Subcategory != "Heavy Orbital Mecha" {
		t.Errorf("expected subcategory 'Heavy Orbital Mecha' restored from manifest, got %q", restoredPage.Assets[0].Subcategory)
	}
	if restoredPage.Assets[0].Rating != 9 || !restoredPage.Assets[0].Favorite {
		t.Errorf("expected rating 9 and favorite true restored from manifest, got rating=%d favorite=%v", restoredPage.Assets[0].Rating, restoredPage.Assets[0].Favorite)
	}

	// Verify sorting by rating and favorites
	favSortPage, err := catalogSvc.Query(ctx, domain.CatalogFilter{SortBy: "favorites"})
	if err != nil || len(favSortPage.Assets) == 0 || !favSortPage.Assets[0].Favorite {
		t.Fatalf("expected top asset in favorites sort to be favorite, got: %+v (err=%v)", favSortPage, err)
	}

	rateSortPage, err := catalogSvc.Query(ctx, domain.CatalogFilter{SortBy: "rating"})
	if err != nil || len(rateSortPage.Assets) < 2 {
		t.Fatalf("expected at least 2 assets in rating sort, got: %+v (err=%v)", rateSortPage, err)
	}
	if rateSortPage.Assets[0].Rating < rateSortPage.Assets[1].Rating {
		t.Errorf("expected descending rating order, got %d then %d", rateSortPage.Assets[0].Rating, rateSortPage.Assets[1].Rating)
	}
	if rateSortPage.Assets[0].Rating != 10 {
		t.Errorf("expected top asset in rating sort to have rating 10, got %d", rateSortPage.Assets[0].Rating)
	}

	// 5. Verify Caption Trigger Filter and Related Assets by Category
	trigPage, err := catalogSvc.Query(ctx, domain.CatalogFilter{Trigger: "orbital mech drop"})
	if err != nil || len(trigPage.Assets) != 1 {
		t.Fatalf("expected 1 asset matching caption trigger 'orbital mech drop', got %d (err=%v)", len(trigPage.Assets), err)
	}

	// 6. Test Differential Delta Scan: a second scan of the same unchanged files must skip them all
	rescanReport, err := scannerSvc.Scan(ctx)
	if err != nil {
		t.Fatalf("Rescan failed: %v", err)
	}
	if rescanReport.UnchangedFiles == 0 {
		t.Errorf("expected differential scanner to detect unchanged files, got %d", rescanReport.UnchangedFiles)
	}
	if rescanReport.ImagesFound > 0 {
		t.Errorf("expected 0 new images indexed on unchanged rescan, got %d", rescanReport.ImagesFound)
	}
}
