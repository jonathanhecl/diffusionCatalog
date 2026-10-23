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

	CREATE TABLE IF NOT EXISTS model_loras (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		profile_slug TEXT NOT NULL,
		name TEXT NOT NULL,
		syntax_tag TEXT NOT NULL DEFAULT '',
		trigger_words TEXT NOT NULL DEFAULT '',
		recommended_weight TEXT NOT NULL DEFAULT '0.8',
		description TEXT NOT NULL DEFAULT '',
		usage_examples TEXT NOT NULL DEFAULT '',
		created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
		UNIQUE(profile_slug, name)
	);

	CREATE TABLE IF NOT EXISTS generated_prompts (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		asset_id INTEGER,
		character_id INTEGER,
		target_profile_slug TEXT NOT NULL DEFAULT 'flux',
		scene_direction TEXT NOT NULL DEFAULT '',
		output_prompt TEXT NOT NULL,
		provider TEXT NOT NULL DEFAULT 'ollama',
		model_name TEXT NOT NULL DEFAULT '',
		created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
		FOREIGN KEY (asset_id) REFERENCES assets(id) ON DELETE CASCADE,
		FOREIGN KEY (character_id) REFERENCES characters(id) ON DELETE CASCADE
	);
	`
	if _, err := db.ExecContext(ctx, schema); err != nil {
		return err
	}
	return seedDefaultSettingsAndProfiles(ctx, db)
}

func seedDefaultSettingsAndProfiles(ctx context.Context, db *sql.DB) error {
	defaults := map[string]string{
		"ollama_base_url":          "http://mac-mini.local:11434",
		"omlx_base_url":            "http://mac-mini.local:8000",
		"omlx_api_key":             "omlx",
		"vision_provider":          "ollama",
		"vision_model":             "llama3.2-vision:latest",
		"vision_think":             "off",
		"image2prompt_provider":    "ollama",
		"image2prompt_model":       "llama3.2-vision:latest",
		"image2prompt_think":       "off",
		"text2prompt_provider":     "omlx",
		"text2prompt_model":        "SuperQwen3.8-27b-abliterated-MLX-4bit",
		"text2prompt_think":        "low",
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
			Slug:        "qwen-image",
			Name:        "Qwen-Image",
			Modality:    "image",
			Description: "Dense semantic description with accurate typography, layout spatial cues, and stylistic precision.",
			Prompt:      "You are a specialist prompt engineer for Qwen-Image. Write a clear, spatially structured prompt specifying foreground/background placement, exact subject attributes, lighting atmosphere, and any embedded signage or typography in quotes. Output only the final prompt.",
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

	if count == 0 {
		for _, p := range profiles {
			_, _ = db.ExecContext(ctx, `
				INSERT OR IGNORE INTO prompt_profiles (slug, name, target_modality, description, system_prompt)
				VALUES (?, ?, ?, ?, ?)`,
				p.Slug, p.Name, p.Modality, p.Description, p.Prompt,
			)
		}
	} else {
		// Ensure qwen-image exists on upgraded databases
		for _, p := range profiles {
			if p.Slug == "qwen-image" {
				_, _ = db.ExecContext(ctx, `
					INSERT OR IGNORE INTO prompt_profiles (slug, name, target_modality, description, system_prompt)
					VALUES (?, ?, ?, ?, ?)`,
					p.Slug, p.Name, p.Modality, p.Description, p.Prompt,
				)
			}
		}
	}

	var loraCount int
	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM model_loras`).Scan(&loraCount); err != nil || loraCount > 0 {
		return nil
	}

	seedLoRAs := []struct {
		ProfileSlug string
		Name        string
		SyntaxTag   string
		Trigger     string
		Weight      string
		Desc        string
		Examples    string
	}{
		{
			ProfileSlug: "flux",
			Name:        "Dark Sci-Fi Anamorphic 70mm",
			SyntaxTag:   "<lora:flux_anamorphic_70mm:0.85>",
			Trigger:     "anamorphic70mm, horizontal cyan lens flare",
			Weight:      "0.85",
			Desc:        "Trained on 70mm IMAX and Panavision sci-fi film stills for Flux.1. Adds oval bokeh, film grain, and subtle cyan/amber highlight halation.",
			Examples:    "A lone netrunner standing on a rain-slicked brutalist balcony, anamorphic70mm, horizontal cyan lens flare, sodium vapor streetlights reflecting on wet concrete <lora:flux_anamorphic_70mm:0.85>",
		},
		{
			ProfileSlug: "krea2",
			Name:        "Tactile Editorial Skin & Fabric (Triggerless)",
			SyntaxTag:   "<lora:krea2_editorial_microtexture:0.7>",
			Trigger:     "",
			Weight:      "0.70",
			Desc:        "Triggerless detail enhancer specifically tuned for Krea 2. Boosts natural skin pores, woven fabric threads, and subsurface scattering under studio softboxes without plastic over-smoothing.",
			Examples:    "Editorial medium-format close-up portrait, natural freckled skin translucency, heavy charcoal wool collar, soft key light from 45 degrees, Kodak Portra 400 <lora:krea2_editorial_microtexture:0.7>",
		},
		{
			ProfileSlug: "qwen-image",
			Name:        "Neon Cyber-Signage & Crisp Typography",
			SyntaxTag:   "<lora:qwen_neon_signage_v2:0.9>",
			Trigger:     "cyber_signage, glowing acrylic lettering",
			Weight:      "0.90",
			Desc:        "Qwen-Image LoRA specialized in rendering legible futuristic storefront signs, HUD telemetry overlays, and glowing glass typography.",
			Examples:    "Nighttime alley in Neo-Veridia, cyber_signage, glowing acrylic lettering reading \"ARCHIVE SECTOR 09\" above a rain-misted glass doorway <lora:qwen_neon_signage_v2:0.9>",
		},
		{
			ProfileSlug: "minimaxh3",
			Name:        "FPV Orbital Tracking Shot (Video Motion LoRA)",
			SyntaxTag:   "<lora:minimax_fpv_orbital:0.8>",
			Trigger:     "smooth orbital camera arc, dynamic parallax motion",
			Weight:      "0.80",
			Desc:        "Motion LoRA for MiniMax Hailuo H3 video generation. Drives a continuous 180-degree orbital camera trajectory around the main subject with strong foreground/background parallax.",
			Examples:    "[Camera: smooth orbital camera arc, dynamic parallax motion] A heavy orbital mecha unit ignites its thrusters inside a subterranean hangar as sparks and volumetric steam swirl across the lens <lora:minimax_fpv_orbital:0.8>",
		},
	}

	for _, l := range seedLoRAs {
		_, _ = db.ExecContext(ctx, `
			INSERT OR IGNORE INTO model_loras (
				profile_slug, name, syntax_tag, trigger_words, recommended_weight, description, usage_examples
			) VALUES (?, ?, ?, ?, ?, ?, ?)`,
			l.ProfileSlug, l.Name, l.SyntaxTag, l.Trigger, l.Weight, l.Desc, l.Examples,
		)
	}
	return nil
}
