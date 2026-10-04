# Method model forward test: pre-registration

Log: `forward_log_method_v1.jsonl`, written by `scripts/method_forward_track.py`
(`--log` / `--settle` / `--report`). Log version `method-v1-2026-09-30`.

## What is logged (since 2026-09-30)

Each upcoming fight is logged once, at its opening BestFightOdds prop line (or the day before
the card if no props appear), with:
- the served method grid (`ufc_method_predictions`: six winner x method cells, KO/Sub/Dec,
  goes the distance) and the method model version;
- the opening props (de-vigged probabilities; from 2026-10-04 also each market's typical
  (median) American price).

Settlement adds the result and the closing props. The main question is accuracy: six-way log
loss of the model against the props at open and close.

## Secondary hypothesis H3: "favourite by decision" is underpriced

Registered 2026-10-04, before any H3 row was logged.

**Why.** In backtests, books de-vig "favourite wins by decision" at 27.7% and it happens 32.8%
(2,831 priced fights; `scripts/build_grade_table.py`). The family's realised ROI is +27% on 683
bets (SE 9%). On held-out fights (2024-09 to 2026-09, 909 priced) it was underpriced in every
favourite band. This pattern was found by looking at the data, so it is a hypothesis, not an
edge, until this test settles it.

**The rule.** Per fight, frozen at log time in the row's `h3` block (`h3_decision`):
- favourite = the corner whose three opening six-way cells sum to the higher de-vigged
  probability;
- p = the served model's favourite-by-decision cell, blended 50/50 in log-odds with the
  de-vigged opening market (`grading.blend_prob`, `PROP_MODEL_WEIGHT` 0.5): the probability
  the picks page uses;
- bet 1 unit at the opening typical (median) price when EV = p x decimal - 1 >= 3%
  (`H3_MIN_EV`, the picks threshold).

**Scored** (`--report`): ROI at the logged price, hit rate, and closing-line value (de-vigged
close minus de-vigged open, in points). Draws, no-contests and DQs void.

**Judged at 150 settled bets** (`H3_CHECKPOINT`; roughly 6-9 months of cards):
- **Confirmed** if the one-sided 95% bootstrap lower bound on ROI is above 0 and mean CLV is
  >= 0.
- **Rejected** if ROI is below 0. The picks page then stops treating the angle as a pick
  family (its grade backtest is re-examined).
- Otherwise inconclusive: keep logging to 300 bets and apply the same test once.

**Why it matters for the picks page.** Favourite by decision is the headline pick on most
fights (UFC 332: 10 of 12). The picks are left as the model ranks them. This test is the
check on whether that is a real edge or a bias: part of it is a known model bias, since the
model overstates the cell when an 85%+ favourite wins (it says 38.5% where 32.5% happen).

## Rules

- Append-only. `--settle` fills only the result and close fields of existing rows.
- Any change to the served method grid (model artifact, features, or a correction layer such
  as `app/services/ufc/method_market.py`, built 2026-10-04 and **not served**: it failed its
  walk-forward gate) starts a new log version with its own addendum here, written before it is
  served. Retraining with unchanged code does not.
- Changing H3's rule constants (`H3_MIN_EV`, `PROP_MODEL_WEIGHT` or the favourite definition)
  ends H3. A new hypothesis gets a new name and starts at 0 bets.
- The log and this file are committed, and commit timestamps prove a row predates its fight.
