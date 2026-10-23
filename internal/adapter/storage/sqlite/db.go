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

	CREATE TABLE IF NOT EXISTS app_settings (
		key TEXT PRIMARY KEY,
		value TEXT NOT NULL
	);

	CREATE TABLE IF NOT EXISTS prompt_profiles (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		slug TEXT NOT NULL UNIQUE,
		name TEXT NOT NULL,
		target_modality TEXT NOT NULL DEFAULT 'image',
		description TEXT NOT NULL DEFAULT '',
		system_prompt TEXT NOT NULL,
		updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
	);
	`
	if _, err := db.ExecContext(ctx, schema); err != nil {
		return err
	}
	return seedDefaultSettingsAndProfiles(ctx, db)
}

func seedDefaultSettingsAndProfiles(ctx context.Context, db *sql.DB) error {
	defaults := map[string]string{
		"ollama_base_url":          "http://localhost:11434",
		"vision_model":             "llama3.2-vision:latest",
		"image2prompt_model":       "llama3.2-vision:latest",
		"text2prompt_model":        "qwen3:8b",
		"default_target_generator": "flux",
		"auto_caption_on_scan":     "false",
	}
	for k, v := range defaults {
		_, _ = db.ExecContext(ctx, `INSERT OR IGNORE INTO app_settings (key, value) VALUES (?, ?)`, k, v)
	}

	var count int
	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM prompt_profiles`).Scan(&count); err != nil {
		return err
	}
	if count > 0 {
		return nil
	}

	profiles := []struct {
		Slug        string
		Name        string
		Modality    string
		Description string
		Prompt      string
	}{
		{
			Slug:        "flux",
			Name:        "Flux.1 Dev / Pro",
			Modality:    "image",
			Description: "Natural-language T5 prose optimized for Flux.1 architecture without weight brackets.",
			Prompt:      "You are an expert prompt engineer for Flux.1 diffusion models. Write rich, natural-language descriptive prose (3 to 5 sentences). Describe the primary subject, facial expression or focal geometry, attire/materials, spatial composition, camera lens (e.g., 85mm f/1.4, anamorphic), lighting direction, color palette, and surface textures. Do NOT use Danbooru tag weights like (word:1.3). Output only the final prompt.",
		},
		{
			Slug:        "krea2",
			Name:        "Krea 2",
			Modality:    "image",
			Description: "High-aesthetic editorial art direction, micro-texture fidelity, and organic film lighting.",
			Prompt:      "You are a senior art director crafting prompts for Krea 2. Focus on hyper-tactile surface micro-textures, skin/material translucency, editorial studio or golden-hour lighting, medium-format analog film stock aesthetics (Kodak Portra 400, Hasselblad H6D), and deliberate depth-of-field. Output only the final prompt.",
		},
		{
			Slug:        "moss",
			Name:        "MOSS Diffusion",
			Modality:    "image",
			Description: "Structured multi-layered scene composition with explicit foreground, subject, and atmosphere.",
			Prompt:      "You are a specialist prompt architect for MOSS Diffusion. Structure the prompt into clear descriptive layers: [Subject & Pose], [Wardrobe & Materials], [Foreground & Midground Environment], [Volumetric Lighting & Atmosphere], and [Camera & Color Grading]. Output a cohesive paragraph with high semantic density and zero conversational filler.",
		},
		{
			Slug:        "minimaxh3",
			Name:        "MiniMax Hailuo H3 (Video)",
			Modality:    "video",
			Description: "Cinematic video generation prompt specifying camera trajectory, subject motion, and physics.",
			Prompt:      "You are a cinematic AI video prompt director for MiniMax Hailuo H3 (video generation). Structure every prompt with: 1) Explicit camera movement (e.g., slow dolly-in, orbital tracking shot, handheld pan), 2) Subject motion and temporal micro-actions, 3) Environmental physics (wind, rain ripples, shifting neon reflections, volumetric dust), and 4) Lighting and film aesthetic. Output only the final video prompt.",
		},
		{
			Slug:        "sdxl",
			Name:        "SDXL / Illustrious / Pony",
			Modality:    "image",
			Description: "Hybrid comma-separated tag and phrase structure for SDXL checkpoints.",
			Prompt:      "You are an SDXL prompt specialist. Generate a comma-separated prompt starting with quality and medium tokens (masterpiece, best quality, highres), followed by detailed subject description, clothing, pose, background environment, lighting tags, and camera perspective. Output only the final comma-separated prompt.",
		},
	}

	for _, p := range profiles {
		_, _ = db.ExecContext(ctx, `
			INSERT OR IGNORE INTO prompt_profiles (slug, name, target_modality, description, system_prompt)
			VALUES (?, ?, ?, ?, ?)`,
			p.Slug, p.Name, p.Modality, p.Description, p.Prompt,
		)
	}
	return nil
}
