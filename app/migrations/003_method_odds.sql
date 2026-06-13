CREATE TABLE IF NOT EXISTS ufc_method_odds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fight_id INTEGER NOT NULL REFERENCES ufc_fights(id),
    bookmaker VARCHAR(100) NOT NULL,
    ko_odds INTEGER,
    sub_odds INTEGER,
    dec_odds INTEGER,
    ko_prob REAL,
    sub_prob REAL,
    dec_prob REAL,
    red_ko_odds INTEGER,
    red_sub_odds INTEGER,
    red_dec_odds INTEGER,
    blue_ko_odds INTEGER,
    blue_sub_odds INTEGER,
    blue_dec_odds INTEGER,
    UNIQUE(fight_id, bookmaker)
);

CREATE INDEX IF NOT EXISTS ix_ufc_method_odds_fight_id ON ufc_method_odds(fight_id);
