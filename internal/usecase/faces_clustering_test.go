package usecase

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/metadata"
	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/storage/sqlite"
	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

func TestClusterDetectedFacesAndRenaming(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test_faces.db")
	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("failed to init test db: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db)

	// Create test asset
	assetID, err := repo.UpsertAsset(ctx, &domain.MediaAsset{
		RootID:    1,
		RelPath:   "test_image_1.png",
		FileName:  "test_image_1.png",
		MediaType: domain.MediaTypeImage,
	})
	if err != nil {
		t.Fatalf("failed to insert asset: %v", err)
	}

	// Create two faces with very similar embeddings (>95% cosine similarity)
	// and one face with an orthogonal embedding (~0% similarity)
	embPersonA1 := []float32{0.9, 0.4, 0.1, 0.0, 0.0}
	embPersonA2 := []float32{0.88, 0.42, 0.11, 0.02, 0.0}
	embPersonB := []float32{0.0, 0.0, 0.0, 0.8, 0.6}

	f1 := &domain.DetectedFace{
		AssetID:      assetID,
		CropRelPath:  "crops/face1.png",
		Confidence:   0.95,
		Embedding:    embPersonA1,
		ClusterLabel: "Unassigned",
	}
	f2 := &domain.DetectedFace{
		AssetID:      assetID,
		CropRelPath:  "crops/face2.png",
		Confidence:   0.94,
		Embedding:    embPersonA2,
		ClusterLabel: "Unassigned",
	}
	f3 := &domain.DetectedFace{
		AssetID:      assetID,
		CropRelPath:  "crops/face3.png",
		Confidence:   0.92,
		Embedding:    embPersonB,
		ClusterLabel: "Unassigned",
	}

	id1, _ := repo.SaveDetectedFace(ctx, f1)
	id2, _ := repo.SaveDetectedFace(ctx, f2)
	id3, _ := repo.SaveDetectedFace(ctx, f3)

	scanner := NewScannerService(tmpDir, tmpDir, repo, metadata.NewExtractor(), nil)

	// Run clustering with threshold 0.50
	clusterCount, err := scanner.ClusterDetectedFaces(ctx, 0.50)
	if err != nil {
		t.Fatalf("ClusterDetectedFaces failed: %v", err)
	}
	if clusterCount != 2 {
		t.Fatalf("expected 2 clusters, got %d", clusterCount)
	}

	faces, err := repo.ListFacesWithEmbeddings(ctx)
	if err != nil {
		t.Fatalf("ListFacesWithEmbeddings failed: %v", err)
	}

	var face1Label, face2Label, face3Label string
	for _, f := range faces {
		if f.ID == id1 {
			face1Label = f.ClusterLabel
		} else if f.ID == id2 {
			face2Label = f.ClusterLabel
		} else if f.ID == id3 {
			face3Label = f.ClusterLabel
		}
	}

	if face1Label == "" || face2Label == "" || face3Label == "" {
		t.Fatalf("all faces must have cluster labels assigned, got %q, %q, %q", face1Label, face2Label, face3Label)
	}
	if face1Label != face2Label {
		t.Fatalf("face1 and face2 must belong to the same cluster, got %q vs %q", face1Label, face2Label)
	}
	if face1Label == face3Label {
		t.Fatalf("face1 and face3 must belong to different clusters, got %q", face1Label)
	}

	// Test renaming the cluster to a custom character name
	catalogSvc := NewCatalogService(tmpDir, repo, nil)
	if err := catalogSvc.RenameFaceCluster(ctx, face1Label, "Elena Rostova", true); err != nil {
		t.Fatalf("RenameFaceCluster failed: %v", err)
	}

	// Verify faces are updated
	facesAfter, _ := repo.ListFacesWithEmbeddings(ctx)
	for _, f := range facesAfter {
		if f.ID == id1 || f.ID == id2 {
			if f.ClusterLabel != "Elena Rostova" {
				t.Fatalf("expected face %d to have label 'Elena Rostova', got %q", f.ID, f.ClusterLabel)
			}
			if f.CharacterID == nil || *f.CharacterID <= 0 {
				t.Fatalf("expected face %d to be linked to new Character", f.ID)
			}
		}
	}

	// Verify Character was created
	chars, _ := catalogSvc.ListCharacters(ctx)
	foundChar := false
	for _, c := range chars {
		if c.Name == "Elena Rostova" {
			foundChar = true
			break
		}
	}
	if !foundChar {
		t.Fatalf("expected Character 'Elena Rostova' to be created")
	}
}

func TestReextractMissingMetadataPersistence(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test_missing_meta.db")
	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("failed to init test db: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db)

	// 1. Asset with empty prompt and no extraction timestamp
	assetID, err := repo.UpsertAsset(ctx, &domain.MediaAsset{
		RootID:    1,
		RelPath:   "legacy_asset.png",
		FileName:  "legacy_asset.png",
		MediaType: domain.MediaTypeImage,
	})
	if err != nil {
		t.Fatalf("failed to insert legacy asset: %v", err)
	}

	missing, err := repo.ListAssetsMissingMetadata(ctx)
	if err != nil {
		t.Fatalf("ListAssetsMissingMetadata failed: %v", err)
	}
	if len(missing) != 1 || missing[0].ID != assetID {
		t.Fatalf("expected legacy asset to be returned in missing list, got: %+v", missing)
	}

	// Update metadata and set extraction date
	now := time.Now()
	err = repo.UpdateAssetMetadata(ctx, assetID, domain.GenerationParams{
		Engine:         domain.EngineComfyUI,
		PositivePrompt: "1girl, futuristic city, cybernetic aesthetic",
	}, now)
	if err != nil {
		t.Fatalf("UpdateAssetMetadata failed: %v", err)
	}

	// Verify it has extraction date
	updated, err := repo.GetAssetByID(ctx, assetID)
	if err != nil {
		t.Fatalf("GetAssetByID failed: %v", err)
	}
	if updated.MetadataExtractedAt == nil {
		t.Fatalf("expected MetadataExtractedAt to be recorded")
	}
	if updated.Metadata.PositivePrompt != "1girl, futuristic city, cybernetic aesthetic" {
		t.Fatalf("prompt was not updated properly: %q", updated.Metadata.PositivePrompt)
	}

	// Verify it no longer appears in missing list
	missingAfter, err := repo.ListAssetsMissingMetadata(ctx)
	if err != nil {
		t.Fatalf("ListAssetsMissingMetadata failed: %v", err)
	}
	if len(missingAfter) != 0 {
		t.Fatalf("expected 0 assets missing metadata, got %d", len(missingAfter))
	}
}
