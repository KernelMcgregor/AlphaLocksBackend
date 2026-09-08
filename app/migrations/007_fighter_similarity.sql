-- Top-K stylistic comparables per fighter, backing the "Similar Fighters" panel.
--
-- DISPLAY-ONLY AND RETRODICTIVE. Rows are built from career-to-date stats and the final
-- Glicko ratings, so a fighter's vector reflects fights that had not yet happened at the
-- time of any given historical bout. Same hazard class as whr_ranker: never a model
-- feature. tests/test_leakage.py asserts no style_/similar_ column reaches
-- winner_feature_columns().
--
-- Top-K, not pairwise. 4,496 fighters is ~20M ordered pairs and nothing in the product
-- asks for an arbitrary pair's similarity, so we store the 20 rows per fighter that will
-- actually be served -- the same call ufc_matchup_predictions makes.
--
-- previous_rank records where the pair sat in the prior run and is carried across the
-- delete-and-reinsert. It is what makes "whose comparables changed after this event"
-- answerable; NULL means the pair is newly in the top-K.
--
-- This is a new table, so Base.metadata.create_all creates it on both SQLite and
-- Cockroach. Unlike 005/006 there are no additive ALTERs, so run_migrations() in main.py
-- needs no entry for it. This file documents the shape.

CREATE TABLE IF NOT EXISTS ufc_fighter_similarity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fighter_id INTEGER NOT NULL REFERENCES ufc_fighters(id),
    similar_fighter_id INTEGER NOT NULL REFERENCES ufc_fighters(id),
    rank INTEGER NOT NULL,
    similarity REAL NOT NULL,
    same_division BOOLEAN NOT NULL DEFAULT 0,
    top_drivers TEXT NOT NULL,
    previous_rank INTEGER,
    computed_at TIMESTAMP,

    UNIQUE (fighter_id, similar_fighter_id)
);

CREATE INDEX IF NOT EXISTS ix_similarity_fighter ON ufc_fighter_similarity (fighter_id);
