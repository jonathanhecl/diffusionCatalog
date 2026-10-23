package sqlite

import (
	"context"
	"database/sql"
	"fmt"

	_ "modernc.org/sqlite"
)

// Open initializes the local SQLite database with WAL mode, foreign keys, and schema migrations.
func Open(ctx context.Context, dbPath string) (*sql.DB, error) {
	dsn := fmt.Sprintf("file:%s?_pragma=busy_timeout(5000)&_pragma=journal_mode(WAL)&_pragma=foreign_keys(ON)", dbPath)
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, err
	}
	if err := db.PingContext(ctx); err != nil {
		_ = db.Close()
		return nil, err
	}
	if err := migrate(ctx, db); err != nil {
		_ = db.Close()
		return nil, err
	}
	return db, nil
}

func migrate(ctx context.Context, db *sql.DB) error {
	schema := `
	CREATE TABLE IF NOT EXISTS users (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		username TEXT NOT NULL UNIQUE,
		password_hash TEXT NOT NULL,
		role TEXT NOT NULL DEFAULT 'admin',
		created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
	);

	CREATE TABLE IF NOT EXISTS assets (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		rel_path TEXT NOT NULL UNIQUE,
		file_name TEXT NOT NULL,
		file_hash TEXT NOT NULL,
		media_type TEXT NOT NULL DEFAULT 'image',
		width INTEGER NOT NULL DEFAULT 0,
		height INTEGER NOT NULL DEFAULT 0,
		duration_sec REAL NOT NULL DEFAULT 0,
		file_size INTEGER NOT NULL DEFAULT 0,
		view_count INTEGER NOT NULL DEFAULT 0,
		rating INTEGER NOT NULL DEFAULT 0,
		engine TEXT NOT NULL DEFAULT 'unknown',
		positive_prompt TEXT NOT NULL DEFAULT '',
		negative_prompt TEXT NOT NULL DEFAULT '',
		model_name TEXT NOT NULL DEFAULT '',
		model_hash TEXT NOT NULL DEFAULT '',
		sampler TEXT NOT NULL DEFAULT '',
		scheduler TEXT NOT NULL DEFAULT '',
		steps INTEGER NOT NULL DEFAULT 0,
		cfg_scale REAL NOT NULL DEFAULT 0,
		seed INTEGER NOT NULL DEFAULT 0,
		loras_json TEXT NOT NULL DEFAULT '[]',
		raw_metadata TEXT NOT NULL DEFAULT '',
		indexed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
	);

	CREATE VIRTUAL TABLE IF NOT EXISTS assets_fts USING fts5(
		file_name,
		positive_prompt,
		negative_prompt,
		model_name,
		engine,
		content='assets',
		content_rowid='id'
	);

	CREATE TRIGGER IF NOT EXISTS assets_ai AFTER INSERT ON assets BEGIN
		INSERT INTO assets_fts(rowid, file_name, positive_prompt, negative_prompt, model_name, engine)
		VALUES (new.id, new.file_name, new.positive_prompt, new.negative_prompt, new.model_name, new.engine);
	END;

	CREATE TRIGGER IF NOT EXISTS assets_ad AFTER DELETE ON assets BEGIN
		INSERT INTO assets_fts(assets_fts, rowid, file_name, positive_prompt, negative_prompt, model_name, engine)
		VALUES ('delete', old.id, old.file_name, old.positive_prompt, old.negative_prompt, old.model_name, old.engine);
	END;

	CREATE TRIGGER IF NOT EXISTS assets_au AFTER UPDATE ON assets BEGIN
		INSERT INTO assets_fts(assets_fts, rowid, file_name, positive_prompt, negative_prompt, model_name, engine)
		VALUES ('delete', old.id, old.file_name, old.positive_prompt, old.negative_prompt, old.model_name, old.engine);
		INSERT INTO assets_fts(rowid, file_name, positive_prompt, negative_prompt, model_name, engine)
		VALUES (new.id, new.file_name, new.positive_prompt, new.negative_prompt, new.model_name, new.engine);
	END;

	CREATE TABLE IF NOT EXISTS tags (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		name TEXT NOT NULL UNIQUE,
		facet TEXT NOT NULL DEFAULT 'general',
		color TEXT NOT NULL DEFAULT '#2563eb'
	);

	CREATE TABLE IF NOT EXISTS asset_tags (
		asset_id INTEGER NOT NULL,
		tag_id INTEGER NOT NULL,
		PRIMARY KEY (asset_id, tag_id),
		FOREIGN KEY (asset_id) REFERENCES assets(id) ON DELETE CASCADE,
		FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
	);

	CREATE TABLE IF NOT EXISTS groups (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		name TEXT NOT NULL UNIQUE,
		slug TEXT NOT NULL UNIQUE,
		description TEXT NOT NULL DEFAULT '',
		category TEXT NOT NULL DEFAULT 'Collection',
		cover_asset_id INTEGER,
		FOREIGN KEY (cover_asset_id) REFERENCES assets(id) ON DELETE SET NULL
	);

	CREATE TABLE IF NOT EXISTS group_assets (
		group_id INTEGER NOT NULL,
		asset_id INTEGER NOT NULL,
		PRIMARY KEY (group_id, asset_id),
		FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE,
		FOREIGN KEY (asset_id) REFERENCES assets(id) ON DELETE CASCADE
	);

	CREATE TABLE IF NOT EXISTS characters (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		name TEXT NOT NULL UNIQUE,
		slug TEXT NOT NULL UNIQUE,
		archetype TEXT NOT NULL DEFAULT 'Concept',
		description TEXT NOT NULL DEFAULT '',
		trigger_words TEXT NOT NULL DEFAULT '',
		base_positive_prompt TEXT NOT NULL DEFAULT '',
		base_negative_prompt TEXT NOT NULL DEFAULT '',
		recommended_checkpoint TEXT NOT NULL DEFAULT '',
		loras_json TEXT NOT NULL DEFAULT '[]',
		avatar_asset_id INTEGER,
		created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
		FOREIGN KEY (avatar_asset_id) REFERENCES assets(id) ON DELETE SET NULL
	);

	CREATE TABLE IF NOT EXISTS character_examples (
		character_id INTEGER NOT NULL,
		asset_id INTEGER NOT NULL,
		PRIMARY KEY (character_id, asset_id),
		FOREIGN KEY (character_id) REFERENCES characters(id) ON DELETE CASCADE,
		FOREIGN KEY (asset_id) REFERENCES assets(id) ON DELETE CASCADE
	);

	CREATE TABLE IF NOT EXISTS faces (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		asset_id INTEGER NOT NULL,
		bbox_x INTEGER NOT NULL DEFAULT 0,
		bbox_y INTEGER NOT NULL DEFAULT 0,
		bbox_w INTEGER NOT NULL DEFAULT 0,
		bbox_h INTEGER NOT NULL DEFAULT 0,
		crop_rel_path TEXT NOT NULL DEFAULT '',
		confidence REAL NOT NULL DEFAULT 0,
		embedding_json TEXT NOT NULL DEFAULT '[]',
		cluster_label TEXT NOT NULL DEFAULT '',
		expression_tag TEXT NOT NULL DEFAULT '',
		character_id INTEGER,
		created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
		FOREIGN KEY (asset_id) REFERENCES assets(id) ON DELETE CASCADE,
		FOREIGN KEY (character_id) REFERENCES characters(id) ON DELETE SET NULL
	);
	`
	_, err := db.ExecContext(ctx, schema)
	return err
}
