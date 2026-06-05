-- UFC tables
CREATE TABLE IF NOT EXISTS ufc_fighters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ufcstats_id VARCHAR(32) UNIQUE NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    nickname VARCHAR(200),
    height VARCHAR(20),
    weight VARCHAR(20),
    reach VARCHAR(20),
    stance VARCHAR(20),
    dob DATE,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    draws INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ufc_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ufcstats_id VARCHAR(32) UNIQUE NOT NULL,
    name VARCHAR(300) NOT NULL,
    date DATE NOT NULL,
    location VARCHAR(200),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ufc_fights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ufcstats_id VARCHAR(32) UNIQUE NOT NULL,
    event_id INTEGER NOT NULL REFERENCES ufc_events(id),
    red_fighter_id INTEGER NOT NULL REFERENCES ufc_fighters(id),
    blue_fighter_id INTEGER NOT NULL REFERENCES ufc_fighters(id),
    winner_id INTEGER REFERENCES ufc_fighters(id),
    red_result VARCHAR(10),
    blue_result VARCHAR(10),
    weight_class VARCHAR(100),
    method VARCHAR(100),
    finish_round INTEGER,
    finish_time VARCHAR(10),
    time_format VARCHAR(50),
    fight_time_seconds INTEGER,
    max_fight_time_seconds INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ufc_fight_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fight_id INTEGER NOT NULL REFERENCES ufc_fights(id),
    fighter_id INTEGER NOT NULL REFERENCES ufc_fighters(id),
    corner VARCHAR(4) NOT NULL,
    kd INTEGER DEFAULT 0,
    sig_str_landed INTEGER DEFAULT 0,
    sig_str_attempted INTEGER DEFAULT 0,
    total_str_landed INTEGER DEFAULT 0,
    total_str_attempted INTEGER DEFAULT 0,
    td_landed INTEGER DEFAULT 0,
    td_attempted INTEGER DEFAULT 0,
    sub_att INTEGER DEFAULT 0,
    rev INTEGER DEFAULT 0,
    ctrl_seconds INTEGER DEFAULT 0,
    head_landed INTEGER DEFAULT 0,
    head_attempted INTEGER DEFAULT 0,
    body_landed INTEGER DEFAULT 0,
    body_attempted INTEGER DEFAULT 0,
    leg_landed INTEGER DEFAULT 0,
    leg_attempted INTEGER DEFAULT 0,
    distance_landed INTEGER DEFAULT 0,
    distance_attempted INTEGER DEFAULT 0,
    clinch_landed INTEGER DEFAULT 0,
    clinch_attempted INTEGER DEFAULT 0,
    ground_landed INTEGER DEFAULT 0,
    ground_attempted INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(fight_id, fighter_id)
);

-- Shared tables
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport VARCHAR(50) NOT NULL,
    event_id INTEGER NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    predicted_outcome VARCHAR(200) NOT NULL,
    confidence REAL NOT NULL,
    actual_outcome VARCHAR(200),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    run_date TIMESTAMP NOT NULL,
    accuracy REAL NOT NULL,
    notes VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport VARCHAR(50) NOT NULL,
    event_id INTEGER NOT NULL,
    source VARCHAR(100) NOT NULL,
    home_odds REAL,
    away_odds REAL,
    draw_odds REAL,
    over_under REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
