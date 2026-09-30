# Pre-registration v2.0: ensemble forward test

**Registered 2026-09-28, before any fight it applies to.**
**Implementation: `scripts/forward_track.py`. Log: `forward_log_v2.jsonl`.**
**Settled fights under v2.0: 0. The count starts here.**

## Why a new log

v1.2 (`PREREGISTRATION.md`) had 2 settled picks when the winner-model pipeline was
rebuilt on 2026-09-28: draws/no-contests stopped counting as losses, serving started
including each fighter's latest fight, Elo and the market devig changed, and new features
were added. The frozen v1.2 picks model depends on the old feature definitions, so
continuing it would feed it inputs it was never trained on (the v1.0 skew again). v1.2 is
therefore **retired, not reset**: its log (`picks_log.jsonl`) is kept as-is, its open
picks are still settled, and no new v1.2 picks are made. v2.0 starts a separate log, as
v1.2's own audit clause requires once a pick has settled.

## What is logged

Every UFC fight that is priced and within 7 days of its event, **once**, on the first
nightly run that finds it. A logged row is never re-priced. Each row records:

- `model_prob_red` — the ensemble's own probability (no odds in its inputs)
- `final_prob_red` — the model blended with the market (the site's displayed number)
- `market_prob_red` — goto-devigged consensus of FanDuel, DraftKings, BetMGM, Bovada,
  BetRivers and Caesars at logging time, with the American prices themselves

Settlement adds the result, the sportsbook closing consensus (last capture before the
event), the Kalshi/Polymarket closing price where available, profit and CLV.

## The questions, in order of how fast they can be answered

1. **Accuracy (primary).** Log loss over all settled priced fights: model alone, blend,
   market at log time, market close. The blend beating the market *at log time* is the
   claim under test (backtest: 0.5863 vs 0.5885 on 1,817 fights, CI including zero).
2. **Closing line value.** Mean CLV of the bets below: `p_close_fair(side) × decimal
   price taken − 1`. Positive mean CLV is the fastest-converging evidence that the prices
   taken were good, independent of who won.
3. **Profit.** Flat-stake ROI of the bets. Noisy; do not act on it below ~500 bets.

## The betting rule

Bet the side with the larger expected value if and only if

    EV = final_prob(side) × decimal_price(side) − 1  >  0

at the logged consensus price. Flat $100. One bet per fight at most.

The threshold is 0 on purpose: it is the one value that was not chosen by looking at a
backtest. (The backtest was scored at 0/2/3/5/8%. The +12% at 8% was the best of five
cuts and is exactly the kind of number this document exists to distrust.)

## Secondary hypothesis H2: fading short-notice replacements

Registered with the rule above, before any data. Every logged fight is tagged from
`ufc_cancelled_bouts` with which corner (if any) stepped in as a short-notice replacement
and whose opponent changed. In 2022-26 backtests replacements won 30% of 102 fights while
the model said 41% and the closing line about 38%, so even the close may overrate them.

Tracked separately, not part of the betting rule: flat bets AGAINST the replacement at the
logged consensus price, graded on ROI and CLV. Only 5-10 such fights a month, so this needs
most of a season before it says anything. The short-notice features stay off in the model
until a walk-forward on about 200 examples confirms them.

## Rules for the log

- Append-only. `--settle` fills only the result/close columns of existing rows.
- Draws, no-contests and DQs are voided (profit 0, excluded from log loss).
- Any change to the model artifact, the feature pipeline, the book set or the rule
  constants opens `forward_log_v3.jsonl` with its own pre-registration. Retraining the
  ensemble on new fights with **unchanged code** is allowed and does not reset the count;
  each row records the rule version, and the artifact's `trained_at` is in git history.
- The log is committed by CI after each run; the commit timestamp is what proves a row
  predates its fight.
