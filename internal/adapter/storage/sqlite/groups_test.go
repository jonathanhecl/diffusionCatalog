package sqlite_test

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/storage/sqlite"
	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

func TestListGroupsPreviewAssets(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test_groups.db")

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("sqlite.Open error: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db)

	// Create 2 groups
	g1ID, err := repo.CreateGroup(ctx, &domain.Group{Name: "Cyberpunk Collection", Description: "Futuristic cityscapes"})
	if err != nil {
		t.Fatalf("CreateGroup g1 error: %v", err)
	}
	g2ID, err := repo.CreateGroup(ctx, &domain.Group{Name: "Mecha Units", Description: "Heavy combat rigs"})
	if err != nil {
		t.Fatalf("CreateGroup g2 error: %v", err)
	}

	// Insert 4 assets for Group 1 with distinct ratings
	now := time.Now().UTC()
	a1 := &domain.MediaAsset{RootID: 1, RelPath: "c1.png", FileName: "c1.png", MediaType: domain.MediaTypeImage, Rating: 5, FileMTime: now.Add(-10 * time.Minute)}
	a2 := &domain.MediaAsset{RootID: 1, RelPath: "c2.png", FileName: "c2.png", MediaType: domain.MediaTypeImage, Rating: 10, FileMTime: now.Add(-20 * time.Minute)} // Highest rating
	a3 := &domain.MediaAsset{RootID: 1, RelPath: "c3.png", FileName: "c3.png", MediaType: domain.MediaTypeImage, Rating: 8, FileMTime: now.Add(-30 * time.Minute)}  // Second highest
	a4 := &domain.MediaAsset{RootID: 1, RelPath: "c4.png", FileName: "c4.png", MediaType: domain.MediaTypeImage, Rating: 7, FileMTime: now.Add(-5 * time.Minute)}   // Third highest
	a5 := &domain.MediaAsset{RootID: 1, RelPath: "c5.png", FileName: "c5.png", MediaType: domain.MediaTypeImage, Rating: 1, FileMTime: now.Add(-40 * time.Minute)}  // Lowest (should not be in top 3)

	id1, _ := repo.UpsertAsset(ctx, a1)
	id2, _ := repo.UpsertAsset(ctx, a2)
	id3, _ := repo.UpsertAsset(ctx, a3)
	id4, _ := repo.UpsertAsset(ctx, a4)
	id5, _ := repo.UpsertAsset(ctx, a5)

	_ = repo.AttachAssetToGroup(ctx, g1ID, id1)
	_ = repo.AttachAssetToGroup(ctx, g1ID, id2)
	_ = repo.AttachAssetToGroup(ctx, g1ID, id3)
	_ = repo.AttachAssetToGroup(ctx, g1ID, id4)
	_ = repo.AttachAssetToGroup(ctx, g1ID, id5)

	// Insert 2 assets for Group 2
	m1 := &domain.MediaAsset{RootID: 1, RelPath: "m1.png", FileName: "m1.png", MediaType: domain.MediaTypeImage, Rating: 4, FileMTime: now}
	m2 := &domain.MediaAsset{RootID: 1, RelPath: "m2.png", FileName: "m2.png", MediaType: domain.MediaTypeImage, Rating: 9, FileMTime: now}
	mid1, _ := repo.UpsertAsset(ctx, m1)
	mid2, _ := repo.UpsertAsset(ctx, m2)
	_ = repo.AttachAssetToGroup(ctx, g2ID, mid1)
	_ = repo.AttachAssetToGroup(ctx, g2ID, mid2)

	// Fetch groups
	groups, err := repo.ListGroups(ctx)
	if err != nil {
		t.Fatalf("ListGroups error: %v", err)
	}

	if len(groups) != 2 {
		t.Fatalf("expected 2 groups, got %d", len(groups))
	}

	// Verify Group 1 (Cyberpunk Collection)
	var g1 *domain.Group
	var g2 *domain.Group
	for i := range groups {
		if groups[i].ID == g1ID {
			g1 = &groups[i]
		}
		if groups[i].ID == g2ID {
			g2 = &groups[i]
		}
	}

	if g1 == nil || len(g1.PreviewAssets) != 3 {
		t.Fatalf("expected Group 1 to have exactly 3 preview assets, got %d", len(g1.PreviewAssets))
	}
	// Verify order: rating 10 (id2), rating 8 (id3), rating 7 (id4)
	if g1.PreviewAssets[0].ID != id2 || g1.PreviewAssets[0].Rating != 10 {
		t.Errorf("expected 1st preview to be id2 (rating 10), got ID=%d Rating=%d", g1.PreviewAssets[0].ID, g1.PreviewAssets[0].Rating)
	}
	if g1.PreviewAssets[1].ID != id3 || g1.PreviewAssets[1].Rating != 8 {
		t.Errorf("expected 2nd preview to be id3 (rating 8), got ID=%d Rating=%d", g1.PreviewAssets[1].ID, g1.PreviewAssets[1].Rating)
	}
	if g1.PreviewAssets[2].ID != id4 || g1.PreviewAssets[2].Rating != 7 {
		t.Errorf("expected 3rd preview to be id4 (rating 7), got ID=%d Rating=%d", g1.PreviewAssets[2].ID, g1.PreviewAssets[2].Rating)
	}

	// Verify Group 2 (Mecha Units) has exactly 2 preview assets
	if g2 == nil || len(g2.PreviewAssets) != 2 {
		t.Fatalf("expected Group 2 to have exactly 2 preview assets, got %d", len(g2.PreviewAssets))
	}
	if g2.PreviewAssets[0].ID != mid2 || g2.PreviewAssets[0].Rating != 9 {
		t.Errorf("expected 1st preview of Group 2 to be mid2 (rating 9), got ID=%d", g2.PreviewAssets[0].ID)
	}
}
