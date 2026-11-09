package sqlite_test

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/jonathanhecl/diffusionCatalog/internal/adapter/storage/sqlite"
	"github.com/jonathanhecl/diffusionCatalog/internal/domain"
)

func TestTagsLifecycleAndOrphanPruning(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test_tags.db")

	db, err := sqlite.Open(ctx, dbPath)
	if err != nil {
		t.Fatalf("sqlite.Open failed: %v", err)
	}
	defer db.Close()

	repo := sqlite.NewRepository(db)

	// Create a media root
	rootID, err := repo.UpsertMediaRoot(ctx, &domain.MediaRoot{
		Path:    "/tmp/media_test",
		Label:   "Test Media",
		Enabled: true,
	})
	if err != nil {
		t.Fatalf("UpsertMediaRoot failed: %v", err)
	}

	// Insert an asset
	assetID, err := repo.UpsertAsset(ctx, &domain.MediaAsset{
		RootID:    rootID,
		RelPath:   "sub/image1.png",
		FileName:  "image1.png",
		MediaType: domain.MediaTypeImage,
		FileMTime: time.Now().UTC(),
	})
	if err != nil {
		t.Fatalf("UpsertAsset failed: %v", err)
	}

	// Create two tags: one will be attached, the other will be orphaned
	tag1, err := repo.UpsertTag(ctx, "lora-flux-realism", "lora", "#7c3aed")
	if err != nil {
		t.Fatalf("UpsertTag 1 failed: %v", err)
	}
	_, err = repo.UpsertTag(ctx, "lora-unused", "lora", "#7c3aed")
	if err != nil {
		t.Fatalf("UpsertTag 2 failed: %v", err)
	}

	// Attach tag1 to asset
	if err := repo.AttachTagToAsset(ctx, assetID, tag1.ID); err != nil {
		t.Fatalf("AttachTagToAsset failed: %v", err)
	}

	// ListTags should only return tag1 (cnt > 0), omitting lora-unused
	tags, err := repo.ListTags(ctx)
	if err != nil {
		t.Fatalf("ListTags failed: %v", err)
	}
	if len(tags) != 1 || tags[0].Name != "lora-flux-realism" || tags[0].AssetCount != 1 {
		t.Fatalf("expected 1 tag with count 1, got %+v", tags)
	}

	// Prune orphaned tags explicitly
	if err := repo.PruneOrphanedTags(ctx); err != nil {
		t.Fatalf("PruneOrphanedTags failed: %v", err)
	}

	// Delete media root (which cascades to asset and triggers PruneOrphanedTags)
	if err := repo.DeleteMediaRoot(ctx, rootID); err != nil {
		t.Fatalf("DeleteMediaRoot failed: %v", err)
	}

	// Now there should be 0 tags in ListTags
	remainingTags, err := repo.ListTags(ctx)
	if err != nil {
		t.Fatalf("ListTags after delete failed: %v", err)
	}
	if len(remainingTags) != 0 {
		t.Fatalf("expected 0 tags after deleting media root, got %+v", remainingTags)
	}

	// Verify the database table 'tags' is completely empty
	var tagCount int
	if err := db.QueryRowContext(ctx, "SELECT COUNT(*) FROM tags").Scan(&tagCount); err != nil {
		t.Fatalf("query tags count failed: %v", err)
	}
	if tagCount != 0 {
		t.Fatalf("expected 0 rows in tags table, got %d", tagCount)
	}
}
