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

	CREATE TABLE IF NOT EXISTS media_roots (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		path TEXT NOT NULL UNIQUE,
		label TEXT NOT NULL,
		enabled INTEGER NOT NULL DEFAULT 1,
		created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
	);

	CREATE TABLE IF NOT EXISTS default_categories (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		name TEXT NOT NULL UNIQUE,
		facet TEXT NOT NULL DEFAULT 'category',
		color TEXT NOT NULL DEFAULT '#2563eb',
		subcategories_json TEXT NOT NULL DEFAULT '[]'
	);

	CREATE TABLE IF NOT EXISTS assets (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		root_id INTEGER NOT NULL DEFAULT 1,
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
		favorite INTEGER NOT NULL DEFAULT 0,
		category TEXT NOT NULL DEFAULT '',
		subcategory TEXT NOT NULL DEFAULT '',
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
		file_mtime DATETIME,
		indexed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
		metadata_extracted_at DATETIME
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
		reference_images_json TEXT NOT NULL DEFAULT '[]',
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
		duration_sec REAL NOT NULL DEFAULT 0.0,
		created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
		FOREIGN KEY (asset_id) REFERENCES assets(id) ON DELETE CASCADE,
		FOREIGN KEY (character_id) REFERENCES characters(id) ON DELETE CASCADE
	);

	CREATE TABLE IF NOT EXISTS studio_compositions (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		name TEXT NOT NULL,
		description TEXT NOT NULL DEFAULT '',
		scene_json TEXT NOT NULL DEFAULT '{}',
		thumbnail_data TEXT NOT NULL DEFAULT '',
		prompt_output TEXT NOT NULL DEFAULT '',
		created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
		updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
	);

	CREATE TABLE IF NOT EXISTS asset_embeddings (
		asset_id INTEGER PRIMARY KEY,
		model TEXT NOT NULL,
		embedding_json TEXT NOT NULL DEFAULT '[]',
		created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
		FOREIGN KEY (asset_id) REFERENCES assets(id) ON DELETE CASCADE
	);
	CREATE INDEX IF NOT EXISTS idx_asset_embeddings_model ON asset_embeddings(model);

	CREATE TABLE IF NOT EXISTS studio_poses (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		name TEXT NOT NULL UNIQUE,
		category TEXT NOT NULL DEFAULT 'Standing',
		pose_json TEXT NOT NULL DEFAULT '{}',
		description TEXT NOT NULL DEFAULT '',
		created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
	);
	`
	if _, err := db.ExecContext(ctx, schema); err != nil {
		return err
	}

	// Safe column migrations for existing databases
	_, _ = db.ExecContext(ctx, `ALTER TABLE assets ADD COLUMN root_id INTEGER NOT NULL DEFAULT 1`)
	_, _ = db.ExecContext(ctx, `ALTER TABLE assets ADD COLUMN category TEXT NOT NULL DEFAULT ''`)
	_, _ = db.ExecContext(ctx, `ALTER TABLE assets ADD COLUMN subcategory TEXT NOT NULL DEFAULT ''`)
	_, _ = db.ExecContext(ctx, `ALTER TABLE assets ADD COLUMN favorite INTEGER NOT NULL DEFAULT 0`)
	_, _ = db.ExecContext(ctx, `ALTER TABLE assets ADD COLUMN file_mtime DATETIME`)
	_, _ = db.ExecContext(ctx, `ALTER TABLE generated_prompts ADD COLUMN duration_sec REAL NOT NULL DEFAULT 0.0`)
	_, _ = db.ExecContext(ctx, `ALTER TABLE characters ADD COLUMN reference_images_json TEXT NOT NULL DEFAULT '[]'`)
	_, _ = db.ExecContext(ctx, `ALTER TABLE assets ADD COLUMN metadata_extracted_at DATETIME`)

	return seedDefaultSettingsAndProfiles(ctx, db)
}

func seedDefaultSettingsAndProfiles(ctx context.Context, db *sql.DB) error {
	defaults := map[string]string{
		"ollama_base_url":          "http://127.0.0.1:11434",
		"omlx_base_url":            "",
		"omlx_api_key":             "",
		"vision_provider":          "ollama",
		"vision_model":             "",
		"vision_think":             "default",
		"image2prompt_provider":    "ollama",
		"image2prompt_model":       "",
		"image2prompt_think":       "default",
		"text2prompt_provider":     "ollama",
		"text2prompt_model":        "",
		"text2prompt_think":        "default",
		"default_target_generator": "",
		"auto_caption_on_scan":     "false",
	}
	for k, v := range defaults {
		_, _ = db.ExecContext(ctx, `INSERT OR IGNORE INTO app_settings (key, value) VALUES (?, ?)`, k, v)
	}
	return seedDefaultPoses(ctx, db)
}

func seedDefaultPoses(ctx context.Context, db *sql.DB) error {
	poses := []struct {
		name     string
		category string
		desc     string
		poseJSON string
	}{
		{
			name:     "Standing Idle",
			category: "Standing",
			desc:     "Neutral standing pose with arms relaxed at sides",
			poseJSON: `{"nose":{"x":50,"y":14},"neck":{"x":50,"y":24},"r_shoulder":{"x":40,"y":26},"r_elbow":{"x":36,"y":42},"r_wrist":{"x":34,"y":57},"l_shoulder":{"x":60,"y":26},"l_elbow":{"x":64,"y":42},"l_wrist":{"x":66,"y":57},"mid_hip":{"x":50,"y":54},"r_hip":{"x":44,"y":56},"r_knee":{"x":43,"y":74},"r_ankle":{"x":42,"y":95},"l_hip":{"x":56,"y":56},"l_knee":{"x":57,"y":74},"l_ankle":{"x":58,"y":95}}`,
		},
		{
			name:     "Walking / Striding",
			category: "Standing",
			desc:     "Mid-stride dynamic walking motion with swinging arms",
			poseJSON: `{"nose":{"x":52,"y":14},"neck":{"x":51,"y":24},"r_shoulder":{"x":42,"y":26},"r_elbow":{"x":33,"y":38},"r_wrist":{"x":27,"y":50},"l_shoulder":{"x":60,"y":26},"l_elbow":{"x":67,"y":38},"l_wrist":{"x":74,"y":49},"mid_hip":{"x":50,"y":54},"r_hip":{"x":44,"y":56},"r_knee":{"x":35,"y":72},"r_ankle":{"x":28,"y":92},"l_hip":{"x":56,"y":56},"l_knee":{"x":63,"y":73},"l_ankle":{"x":72,"y":93}}`,
		},
		{
			name:     "Sitting Relaxed",
			category: "Sitting",
			desc:     "Seated posture with bent knees and hands resting",
			poseJSON: `{"nose":{"x":48,"y":18},"neck":{"x":48,"y":28},"r_shoulder":{"x":38,"y":30},"r_elbow":{"x":34,"y":46},"r_wrist":{"x":40,"y":58},"l_shoulder":{"x":58,"y":30},"l_elbow":{"x":62,"y":46},"l_wrist":{"x":56,"y":58},"mid_hip":{"x":48,"y":58},"r_hip":{"x":42,"y":60},"r_knee":{"x":66,"y":65},"r_ankle":{"x":67,"y":92},"l_hip":{"x":54,"y":60},"l_knee":{"x":74,"y":66},"l_ankle":{"x":75,"y":93}}`,
		},
		{
			name:     "Crouching Action",
			category: "Action",
			desc:     "Low dynamic combat/stealth crouch with bent legs",
			poseJSON: `{"nose":{"x":50,"y":32},"neck":{"x":50,"y":42},"r_shoulder":{"x":38,"y":44},"r_elbow":{"x":28,"y":58},"r_wrist":{"x":32,"y":75},"l_shoulder":{"x":62,"y":44},"l_elbow":{"x":70,"y":55},"l_wrist":{"x":64,"y":72},"mid_hip":{"x":50,"y":65},"r_hip":{"x":42,"y":67},"r_knee":{"x":32,"y":82},"r_ankle":{"x":40,"y":95},"l_hip":{"x":58,"y":67},"l_knee":{"x":72,"y":78},"l_ankle":{"x":65,"y":95}}`,
		},
		{
			name:     "Arms Crossed",
			category: "Standing",
			desc:     "Confident standing pose with arms folded across chest",
			poseJSON: `{"nose":{"x":50,"y":14},"neck":{"x":50,"y":24},"r_shoulder":{"x":39,"y":26},"r_elbow":{"x":36,"y":42},"r_wrist":{"x":55,"y":42},"l_shoulder":{"x":61,"y":26},"l_elbow":{"x":64,"y":42},"l_wrist":{"x":45,"y":42},"mid_hip":{"x":50,"y":54},"r_hip":{"x":44,"y":56},"r_knee":{"x":43,"y":74},"r_ankle":{"x":42,"y":95},"l_hip":{"x":56,"y":56},"l_knee":{"x":57,"y":74},"l_ankle":{"x":58,"y":95}}`,
		},
		{
			name:     "Hands on Hips",
			category: "Standing",
			desc:     "Heroic wide stance with hands resting on waist",
			poseJSON: `{"nose":{"x":50,"y":14},"neck":{"x":50,"y":24},"r_shoulder":{"x":39,"y":26},"r_elbow":{"x":28,"y":38},"r_wrist":{"x":40,"y":54},"l_shoulder":{"x":61,"y":26},"l_elbow":{"x":72,"y":38},"l_wrist":{"x":60,"y":54},"mid_hip":{"x":50,"y":54},"r_hip":{"x":44,"y":56},"r_knee":{"x":42,"y":74},"r_ankle":{"x":40,"y":95},"l_hip":{"x":56,"y":56},"l_knee":{"x":58,"y":74},"l_ankle":{"x":60,"y":95}}`,
		},
		{
			name:     "Reaching Dynamic",
			category: "Action",
			desc:     "Dramatic reach with one arm pointing up and forward",
			poseJSON: `{"nose":{"x":54,"y":18},"neck":{"x":52,"y":28},"r_shoulder":{"x":42,"y":30},"r_elbow":{"x":32,"y":44},"r_wrist":{"x":26,"y":58},"l_shoulder":{"x":62,"y":28},"l_elbow":{"x":72,"y":18},"l_wrist":{"x":82,"y":8},"mid_hip":{"x":50,"y":56},"r_hip":{"x":44,"y":58},"r_knee":{"x":42,"y":76},"r_ankle":{"x":40,"y":95},"l_hip":{"x":56,"y":58},"l_knee":{"x":62,"y":75},"l_ankle":{"x":68,"y":94}}`,
		},
		{
			name:     "Running / Sprint",
			category: "Action",
			desc:     "High-speed sprint with angled torso and pumping limbs",
			poseJSON: `{"nose":{"x":58,"y":22},"neck":{"x":54,"y":30},"r_shoulder":{"x":46,"y":32},"r_elbow":{"x":36,"y":24},"r_wrist":{"x":30,"y":16},"l_shoulder":{"x":62,"y":32},"l_elbow":{"x":70,"y":46},"l_wrist":{"x":78,"y":58},"mid_hip":{"x":50,"y":56},"r_hip":{"x":44,"y":58},"r_knee":{"x":26,"y":68},"r_ankle":{"x":18,"y":84},"l_hip":{"x":56,"y":58},"l_knee":{"x":72,"y":70},"l_ankle":{"x":82,"y":88}}`,
		},
		{
			name:     "Lying Down",
			category: "Lying",
			desc:     "Horizontal recline resting flat",
			poseJSON: `{"nose":{"x":15,"y":52},"neck":{"x":25,"y":54},"r_shoulder":{"x":27,"y":46},"r_elbow":{"x":40,"y":44},"r_wrist":{"x":52,"y":46},"l_shoulder":{"x":27,"y":62},"l_elbow":{"x":40,"y":64},"l_wrist":{"x":52,"y":62},"mid_hip":{"x":55,"y":55},"r_hip":{"x":56,"y":48},"r_knee":{"x":75,"y":49},"r_ankle":{"x":94,"y":50},"l_hip":{"x":56,"y":62},"l_knee":{"x":75,"y":61},"l_ankle":{"x":94,"y":60}}`,
		},
		{
			name:     "Kneeling (Both Knees)",
			category: "Kneeling",
			desc:     "Kneeling frontally with both knees on the floor",
			poseJSON: `{"nose":{"x":50,"y":28},"neck":{"x":50,"y":38},"r_shoulder":{"x":39,"y":40},"r_elbow":{"x":34,"y":54},"r_wrist":{"x":38,"y":68},"l_shoulder":{"x":61,"y":40},"l_elbow":{"x":66,"y":54},"l_wrist":{"x":62,"y":68},"mid_hip":{"x":50,"y":64},"r_hip":{"x":44,"y":65},"r_knee":{"x":42,"y":85},"r_ankle":{"x":44,"y":95},"l_hip":{"x":56,"y":65},"l_knee":{"x":58,"y":85},"l_ankle":{"x":56,"y":95}}`,
		},
		{
			name:     "Kneeling (One Knee)",
			category: "Kneeling",
			desc:     "Genuflecting / one knee on the ground with forward bent knee",
			poseJSON: `{"nose":{"x":50,"y":26},"neck":{"x":50,"y":36},"r_shoulder":{"x":40,"y":38},"r_elbow":{"x":34,"y":52},"r_wrist":{"x":40,"y":66},"l_shoulder":{"x":60,"y":38},"l_elbow":{"x":66,"y":52},"l_wrist":{"x":60,"y":66},"mid_hip":{"x":50,"y":60},"r_hip":{"x":44,"y":61},"r_knee":{"x":42,"y":86},"r_ankle":{"x":42,"y":95},"l_hip":{"x":56,"y":61},"l_knee":{"x":66,"y":72},"l_ankle":{"x":66,"y":95}}`,
		},
		{
			name:     "Crawling / All Fours",
			category: "Action",
			desc:     "Crawling low on hands and knees",
			poseJSON: `{"nose":{"x":50,"y":42},"neck":{"x":50,"y":52},"r_shoulder":{"x":38,"y":54},"r_elbow":{"x":32,"y":68},"r_wrist":{"x":30,"y":86},"l_shoulder":{"x":62,"y":54},"l_elbow":{"x":68,"y":68},"l_wrist":{"x":70,"y":86},"mid_hip":{"x":50,"y":58},"r_hip":{"x":44,"y":60},"r_knee":{"x":40,"y":76},"r_ankle":{"x":42,"y":90},"l_hip":{"x":56,"y":60},"l_knee":{"x":60,"y":76},"l_ankle":{"x":58,"y":90}}`,
		},
		{
			name:     "Hanging / Suspended",
			category: "Action",
			desc:     "Hanging suspended with hands overhead and feet in air",
			poseJSON: `{"nose":{"x":50,"y":28},"neck":{"x":50,"y":36},"r_shoulder":{"x":42,"y":34},"r_elbow":{"x":38,"y":20},"r_wrist":{"x":38,"y":6},"l_shoulder":{"x":58,"y":34},"l_elbow":{"x":62,"y":20},"l_wrist":{"x":62,"y":6},"mid_hip":{"x":50,"y":60},"r_hip":{"x":45,"y":61},"r_knee":{"x":43,"y":76},"r_ankle":{"x":41,"y":92},"l_hip":{"x":55,"y":61},"l_knee":{"x":57,"y":77},"l_ankle":{"x":59,"y":92}}`,
		},
		{
			name:     "Climbing (Ladder / Stairs)",
			category: "Action",
			desc:     "Climbing ladder or stairs with alternating dynamic limb reach",
			poseJSON: `{"nose":{"x":50,"y":20},"neck":{"x":50,"y":28},"r_shoulder":{"x":40,"y":30},"r_elbow":{"x":34,"y":18},"r_wrist":{"x":36,"y":6},"l_shoulder":{"x":60,"y":30},"l_elbow":{"x":68,"y":40},"l_wrist":{"x":66,"y":52},"mid_hip":{"x":50,"y":56},"r_hip":{"x":44,"y":58},"r_knee":{"x":40,"y":72},"r_ankle":{"x":42,"y":88},"l_hip":{"x":56,"y":58},"l_knee":{"x":66,"y":64},"l_ankle":{"x":64,"y":78}}`,
		},
		{
			name:     "Standing Idle (Side)",
			category: "Standing",
			desc:     "Neutral standing pose in profile / side view",
			poseJSON: `{"nose":{"x":62,"y":14},"neck":{"x":50,"y":24},"r_shoulder":{"x":52,"y":26},"r_elbow":{"x":50,"y":43},"r_wrist":{"x":48,"y":58},"l_shoulder":{"x":48,"y":26},"l_elbow":{"x":46,"y":43},"l_wrist":{"x":45,"y":58},"mid_hip":{"x":48,"y":54},"r_hip":{"x":50,"y":56},"r_knee":{"x":50,"y":75},"r_ankle":{"x":52,"y":95},"l_hip":{"x":46,"y":56},"l_knee":{"x":46,"y":75},"l_ankle":{"x":48,"y":95}}`,
		},
		{
			name:     "Walking (Side)",
			category: "Standing",
			desc:     "Profile side-view stride with swinging limbs",
			poseJSON: `{"nose":{"x":64,"y":14},"neck":{"x":52,"y":24},"r_shoulder":{"x":53,"y":26},"r_elbow":{"x":38,"y":38},"r_wrist":{"x":28,"y":48},"l_shoulder":{"x":49,"y":26},"l_elbow":{"x":65,"y":38},"l_wrist":{"x":76,"y":48},"mid_hip":{"x":50,"y":54},"r_hip":{"x":51,"y":56},"r_knee":{"x":34,"y":72},"r_ankle":{"x":24,"y":92},"l_hip":{"x":49,"y":56},"l_knee":{"x":68,"y":73},"l_ankle":{"x":78,"y":93}}`,
		},
		{
			name:     "Running (Side)",
			category: "Action",
			desc:     "Dynamic sprint in profile side view",
			poseJSON: `{"nose":{"x":70,"y":20},"neck":{"x":58,"y":28},"r_shoulder":{"x":58,"y":30},"r_elbow":{"x":42,"y":24},"r_wrist":{"x":32,"y":18},"l_shoulder":{"x":54,"y":30},"l_elbow":{"x":72,"y":44},"l_wrist":{"x":84,"y":56},"mid_hip":{"x":48,"y":56},"r_hip":{"x":50,"y":58},"r_knee":{"x":22,"y":68},"r_ankle":{"x":12,"y":84},"l_hip":{"x":46,"y":58},"l_knee":{"x":78,"y":70},"l_ankle":{"x":90,"y":88}}`,
		},
		{
			name:     "Sitting (Side)",
			category: "Sitting",
			desc:     "Seated profile view with 90-degree bent knees",
			poseJSON: `{"nose":{"x":62,"y":18},"neck":{"x":50,"y":28},"r_shoulder":{"x":50,"y":30},"r_elbow":{"x":58,"y":46},"r_wrist":{"x":66,"y":58},"l_shoulder":{"x":46,"y":30},"l_elbow":{"x":54,"y":46},"l_wrist":{"x":62,"y":58},"mid_hip":{"x":44,"y":58},"r_hip":{"x":45,"y":60},"r_knee":{"x":74,"y":62},"r_ankle":{"x":74,"y":92},"l_hip":{"x":42,"y":60},"l_knee":{"x":72,"y":64},"l_ankle":{"x":72,"y":93}}`,
		},
		{
			name:     "Crouching (Side)",
			category: "Action",
			desc:     "Stealth crouch in profile view",
			poseJSON: `{"nose":{"x":66,"y":32},"neck":{"x":54,"y":42},"r_shoulder":{"x":54,"y":44},"r_elbow":{"x":62,"y":58},"r_wrist":{"x":70,"y":74},"l_shoulder":{"x":48,"y":44},"l_elbow":{"x":44,"y":58},"l_wrist":{"x":40,"y":74},"mid_hip":{"x":42,"y":66},"r_hip":{"x":44,"y":68},"r_knee":{"x":70,"y":80},"r_ankle":{"x":48,"y":95},"l_hip":{"x":40,"y":68},"l_knee":{"x":66,"y":82},"l_ankle":{"x":44,"y":95}}`,
		},
		{
			name:     "Arms Crossed (Side)",
			category: "Standing",
			desc:     "Arms folded across chest in profile",
			poseJSON: `{"nose":{"x":62,"y":14},"neck":{"x":50,"y":24},"r_shoulder":{"x":52,"y":26},"r_elbow":{"x":64,"y":40},"r_wrist":{"x":58,"y":38},"l_shoulder":{"x":48,"y":26},"l_elbow":{"x":60,"y":42},"l_wrist":{"x":54,"y":40},"mid_hip":{"x":48,"y":54},"r_hip":{"x":50,"y":56},"r_knee":{"x":50,"y":75},"r_ankle":{"x":52,"y":95},"l_hip":{"x":46,"y":56},"l_knee":{"x":46,"y":75},"l_ankle":{"x":48,"y":95}}`,
		},
		{
			name:     "Hands on Hips (Side)",
			category: "Standing",
			desc:     "Hands on hips in profile stance",
			poseJSON: `{"nose":{"x":62,"y":14},"neck":{"x":50,"y":24},"r_shoulder":{"x":52,"y":26},"r_elbow":{"x":36,"y":38},"r_wrist":{"x":48,"y":54},"l_shoulder":{"x":48,"y":26},"l_elbow":{"x":38,"y":38},"l_wrist":{"x":46,"y":54},"mid_hip":{"x":48,"y":54},"r_hip":{"x":50,"y":56},"r_knee":{"x":50,"y":75},"r_ankle":{"x":52,"y":95},"l_hip":{"x":46,"y":56},"l_knee":{"x":46,"y":75},"l_ankle":{"x":48,"y":95}}`,
		},
		{
			name:     "Reaching (Side)",
			category: "Action",
			desc:     "Reaching up and forward in profile",
			poseJSON: `{"nose":{"x":66,"y":16},"neck":{"x":54,"y":26},"r_shoulder":{"x":56,"y":28},"r_elbow":{"x":70,"y":16},"r_wrist":{"x":84,"y":6},"l_shoulder":{"x":48,"y":28},"l_elbow":{"x":40,"y":44},"l_wrist":{"x":34,"y":58},"mid_hip":{"x":46,"y":56},"r_hip":{"x":48,"y":58},"r_knee":{"x":54,"y":76},"r_ankle":{"x":56,"y":94},"l_hip":{"x":44,"y":58},"l_knee":{"x":42,"y":76},"l_ankle":{"x":40,"y":94}}`,
		},
		{
			name:     "Lying Down (Side / Lateral)",
			category: "Lying",
			desc:     "Lateral side-sleeping / fetal rest in profile view",
			poseJSON: `{"nose":{"x":18,"y":42},"neck":{"x":28,"y":48},"r_shoulder":{"x":32,"y":46},"r_elbow":{"x":42,"y":54},"r_wrist":{"x":48,"y":46},"l_shoulder":{"x":30,"y":50},"l_elbow":{"x":38,"y":58},"l_wrist":{"x":46,"y":52},"mid_hip":{"x":56,"y":50},"r_hip":{"x":58,"y":49},"r_knee":{"x":76,"y":62},"r_ankle":{"x":92,"y":72},"l_hip":{"x":56,"y":51},"l_knee":{"x":72,"y":65},"l_ankle":{"x":88,"y":76}}`,
		},
		{
			name:     "Kneeling Both (Side)",
			category: "Kneeling",
			desc:     "Profile view kneeling with both knees on ground",
			poseJSON: `{"nose":{"x":60,"y":28},"neck":{"x":50,"y":38},"r_shoulder":{"x":51,"y":40},"r_elbow":{"x":47,"y":56},"r_wrist":{"x":56,"y":68},"l_shoulder":{"x":49,"y":40},"l_elbow":{"x":45,"y":56},"l_wrist":{"x":54,"y":68},"mid_hip":{"x":46,"y":62},"r_hip":{"x":47,"y":63},"r_knee":{"x":64,"y":85},"r_ankle":{"x":36,"y":87},"l_hip":{"x":45,"y":63},"l_knee":{"x":62,"y":85},"l_ankle":{"x":34,"y":87}}`,
		},
		{
			name:     "Kneeling One (Side)",
			category: "Kneeling",
			desc:     "Profile view genuflect / proposal kneeling on one knee",
			poseJSON: `{"nose":{"x":62,"y":26},"neck":{"x":52,"y":36},"r_shoulder":{"x":54,"y":38},"r_elbow":{"x":56,"y":52},"r_wrist":{"x":66,"y":64},"l_shoulder":{"x":48,"y":38},"l_elbow":{"x":46,"y":52},"l_wrist":{"x":56,"y":64},"mid_hip":{"x":48,"y":60},"r_hip":{"x":48,"y":61},"r_knee":{"x":46,"y":88},"r_ankle":{"x":24,"y":88},"l_hip":{"x":48,"y":61},"l_knee":{"x":72,"y":72},"l_ankle":{"x":72,"y":92}}`,
		},
		{
			name:     "Crawling (Side)",
			category: "Action",
			desc:     "Crawling on all fours in profile view",
			poseJSON: `{"nose":{"x":80,"y":42},"neck":{"x":70,"y":46},"r_shoulder":{"x":66,"y":48},"r_elbow":{"x":67,"y":66},"r_wrist":{"x":68,"y":86},"l_shoulder":{"x":62,"y":48},"l_elbow":{"x":61,"y":66},"l_wrist":{"x":60,"y":86},"mid_hip":{"x":38,"y":50},"r_hip":{"x":39,"y":52},"r_knee":{"x":42,"y":76},"r_ankle":{"x":24,"y":86},"l_hip":{"x":37,"y":52},"l_knee":{"x":34,"y":76},"l_ankle":{"x":16,"y":86}}`,
		},
		{
			name:     "Hanging (Side)",
			category: "Action",
			desc:     "Hanging by hands in profile view",
			poseJSON: `{"nose":{"x":58,"y":28},"neck":{"x":50,"y":36},"r_shoulder":{"x":52,"y":34},"r_elbow":{"x":56,"y":20},"r_wrist":{"x":58,"y":6},"l_shoulder":{"x":48,"y":34},"l_elbow":{"x":52,"y":20},"l_wrist":{"x":54,"y":6},"mid_hip":{"x":48,"y":60},"r_hip":{"x":49,"y":61},"r_knee":{"x":43,"y":76},"r_ankle":{"x":38,"y":91},"l_hip":{"x":47,"y":61},"l_knee":{"x":53,"y":77},"l_ankle":{"x":50,"y":91}}`,
		},
		{
			name:     "Climbing (Side)",
			category: "Action",
			desc:     "Climbing ladder or stairs in profile view",
			poseJSON: `{"nose":{"x":64,"y":22},"neck":{"x":54,"y":30},"r_shoulder":{"x":56,"y":32},"r_elbow":{"x":66,"y":20},"r_wrist":{"x":76,"y":10},"l_shoulder":{"x":50,"y":32},"l_elbow":{"x":62,"y":42},"l_wrist":{"x":70,"y":50},"mid_hip":{"x":46,"y":58},"r_hip":{"x":47,"y":60},"r_knee":{"x":40,"y":76},"r_ankle":{"x":40,"y":92},"l_hip":{"x":45,"y":60},"l_knee":{"x":68,"y":68},"l_ankle":{"x":68,"y":82}}`,
		},
	}
	for _, p := range poses {
		_, _ = db.ExecContext(ctx, `INSERT OR IGNORE INTO studio_poses (name, category, description, pose_json) VALUES (?, ?, ?, ?)`,
			p.name, p.category, p.desc, p.poseJSON)
	}
	return nil
}
