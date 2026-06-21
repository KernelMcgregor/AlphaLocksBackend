CREATE TABLE IF NOT EXISTS ufc_fighter_career_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fighter_id INTEGER NOT NULL UNIQUE REFERENCES ufc_fighters(id),

    -- Foundation
    fight_count INTEGER NOT NULL DEFAULT 0,
    total_fight_min REAL,
    est_standing_min REAL,
    est_ground_min REAL,

    -- Striking: Overall
    slpm REAL,
    sapm REAL,
    sl_diff REAL,
    sig_acc REAL,
    sig_def REAL,
    tslpm REAL,

    -- Striking: Head (offense + defense)
    head_pct REAL,
    head_pm REAL,
    head_acc REAL,
    head_abs_pct REAL,
    head_abs_pm REAL,
    head_def REAL,

    -- Striking: Body (offense + defense)
    body_pct REAL,
    body_pm REAL,
    body_acc REAL,
    body_abs_pct REAL,
    body_abs_pm REAL,
    body_def REAL,

    -- Striking: Legs (offense + defense)
    leg_pct REAL,
    leg_pm REAL,
    leg_acc REAL,
    leg_abs_pct REAL,
    leg_abs_pm REAL,
    leg_def REAL,

    -- Striking: Distance position (offense + defense)
    dist_pct REAL,
    dist_pm REAL,
    dist_acc REAL,
    dist_abs_pct REAL,
    dist_abs_pm REAL,
    dist_def REAL,

    -- Striking: Clinch position (offense + defense)
    clinch_pct REAL,
    clinch_pm REAL,
    clinch_acc REAL,
    clinch_abs_pct REAL,
    clinch_abs_pm REAL,
    clinch_def REAL,

    -- Striking: Ground position (offense + defense + position-aware)
    ground_pct REAL,
    ground_pm REAL,
    ground_acc REAL,
    ground_abs_pct REAL,
    ground_abs_pm REAL,
    ground_def REAL,
    gnp15g REAL,
    gnp_abs15g REAL,

    -- Knockdowns
    kd15 REAL,
    kd15s REAL,
    kd_abs15 REAL,
    kd_abs15s REAL,

    -- Takedowns
    td15 REAL,
    td15s REAL,
    td_acc REAL,
    td_abs15 REAL,
    td_abs15s REAL,
    td_def REAL,

    -- Control time
    ctrl15 REAL,
    ctrl15g REAL,
    ctrl_abs15 REAL,
    ctrl_abs15g REAL,

    -- Submissions
    sub_att15 REAL,
    sub_att15g REAL,
    sub_abs15 REAL,
    sub_abs15g REAL,

    -- Reversals
    rev15 REAL,
    rev_abs15 REAL,

    -- Outcomes
    ko_wins INTEGER NOT NULL DEFAULT 0,
    sub_wins INTEGER NOT NULL DEFAULT 0,
    dec_wins INTEGER NOT NULL DEFAULT 0,
    finish_rate REAL,
    win_pct REAL,
    avg_fight_sec REAL,

    -- Metadata
    computed_at TIMESTAMP
);
