"""Point-in-time ("as of before the fight") feature blocks for winner-model experiments.

Each module exposes PREFIX and compute(ctx) -> DataFrame keyed (fight_id, fighter_id), with
every value computed only from bouts strictly before the fight's date. scripts/feature_wf.py
merges a block into the walk-forward matrix and runs the paired test; tests/test_pit_leakage.py
checks the as-of property for every module.

Nothing here imports app.services.ufc.model: the blocks are built from the raw bout tables
plus columns the caller passes in (elo_expected, xsres_*), so they cannot pick up a
full-history value from the model's feature frame by accident.
"""
