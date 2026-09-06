# Pre-registration: UFC moneyline picks rule v1.2

**Registered 2026-09-06, before any fight it applies to.**
**First card under this rule: Noche UFC: Silva vs. Delgado, 2026-09-12.**
**Settled picks under v1.2: 0. The count starts here.**

> ### Two resets on 2026-09-06, both at zero settled picks
>
> The rule was registered and four picks logged before an audit of the ranking system
> found two defects in the pipeline underneath it. Both resets happened the same day,
> before any pick could settle, so both were free.
>
> **v1.0 → v1.1 — serving fix, no model change.** `ufc_glicko_snapshots` had no columns
> for the four Glicko confidence features, so `build_features` fell back to sentinels and
> the picks model received **3 of its 39 features as constants that never appeared in
> training** (the served MLP: 3 of 46). Measured on the training artifact,
> `blue_glicko_meta_rounds_seen` was standardised with mu=12.5, sd=15.1 — a constant 0.0
> fed **−0.83σ into every fight** regardless of who was competing. Migration 006 persists
> the columns; the stored snapshots were then verified byte-identical to the in-memory
> ones training uses (21,312 rows, 0 mismatches). The weights were never wrong.
>
> **v1.1 → v1.2 — model change.** The Glicko dimension semantics were corrected. The
> finish bonus was creating the entire offence/defence antisymmetry (it transferred
> between *different* dimensions, so `ko` inflated while `kod` deflated even after being
> made zero-sum); `str_acc` and `str_def` were algebraic duplicates at r=+0.93; the `td`
> rate branch was centred on the mean of a different observable. Correlation of each
> dimension with a fighter's fight count, before → after:
> `ko +0.735 → +0.335`, `kod −0.590 → +0.163`, `td +0.505 → +0.135`,
> `tdd −0.496 → −0.011`, `corr(str_acc, str_def) +0.93 → +0.165`.
> That redefines this model's inputs, so leaving the frozen artifact in place would have
> recreated the v1.0 skew exactly. Both models were retrained on the corrected features.
> Walk-forward over 8 folds showed the change is performance-neutral (all arm deltas
> within the ±0.006 fold-noise floor, measured from the arm that uses no Glicko at all)
> with slightly better Brier in every Glicko-using arm.
>
> **The rule itself has never been touched** — both thresholds, the staking scheme and the
> book set are as first registered. Only the pipeline beneath it was repaired.
>
> Per "Audit trail" below, rewriting the log voids the experiment and restarts it. The log
> was rewritten rather than annotated on both occasions, so that clause is invoked
> deliberately and the count resets to zero. Nothing was lost: no pick had settled. Prior
> entries remain in git history (v1.0 at commit `155c612`) and are **not** carried
> forward, because they were priced by pipelines that no longer exist.
>
> All model-affecting work identified in the audit is now complete. This is intended to be
> the last reset; any further one after a pick has settled must open a new log alongside
> the old rather than replace it.

This document exists because the rule below was found by searching a backtest, and a rule
found that way is worth nothing until it survives data it was not chosen on. Writing the
parameters down and freezing them is the only mechanism that turns future cards into
evidence instead of into more searching.

---

## The rule

Bet a fight if and only if **both** gates pass:

| Gate | Threshold | Meaning |
|---|---|---|
| Market uncertainty | `abs(devig(market_red) - 0.5) < 0.08` | The market has no strong opinion |
| Model edge | `model_prob(side) - market_prob(side) > 0.05` | The model disagrees enough to matter |

- **Side**: whichever fighter the model favours, determined *before* the edge test.
- **Staking**: flat, $100 per pick. Not Kelly — see "Why flat stakes" below.
- **Model**: `models/ufc/h2h/picks_noodds_gbt_v1.pkl`, a gradient-boosted tree trained
  with `include_odds=False`. It has never seen a betting line.
- **Market price**: mean American odds across FanDuel, DraftKings, BetMGM, Bovada,
  BetRivers, de-vigged by normalising the two implied probabilities.
- **Implementation**: `app/services/ufc/picks.py::select_picks`. That function *is* the
  rule.

Expected volume: about 15% of priced fights, ~70 per year, 2–4 per card.

---

## Why this rule and not "bet where we see edge"

The obvious rule loses money, badly. On the 2,082 priced walk-forward fights:

| Naive rule | n | Accuracy | ROI |
|---|---|---|---|
| edge > 0.05, any fight | 757 | 0.464 | −0.35% |
| edge > 0.10, any fight | 560 | 0.418 | −2.87% |
| edge > 0.20, any fight | 218 | 0.294 | −14.69% |

Accuracy drops *further below chance* the more strongly the model disagrees. Confidently
contradicting a confident market is this model's single worst behaviour.

Adding the market-uncertainty gate reverses the sign:

| Gated rule | n | Accuracy | ROI | 95% CI |
|---|---|---|---|---|
| near-even only | 497 | 0.575 | +8.57% | [+0.6, +17.0] |
| **near-even AND edge > 0.05** | **321** | **0.586** | **+14.16%** | **[+3.3, +24.7]** |

The proposed mechanism: when the de-vigged line sits near 50/50, the market has priced in
everything it knows and arrived at "no opinion". That is the one region where a weaker but
genuinely independent signal has room to be right. Where the market *is* confident, it is
confident for reasons the model does not have access to, and disagreeing is a mistake.

Supporting detail — the effect decays smoothly with the window rather than sitting on a
knife edge:

```
|mkt-.5| < 0.03   n=147   acc=0.605   ROI=+16.14%
|mkt-.5| < 0.05   n=274   acc=0.577   ROI= +9.97%
|mkt-.5| < 0.08   n=497   acc=0.575   ROI= +8.57%
|mkt-.5| < 0.15   n=1021  acc=0.548   ROI= +0.45%
```

Year by year, under the registered rule:

```
2022  n=46  acc=0.478  ROI= -4.26%
2023  n=79  acc=0.620  ROI=+19.72%
2024  n=78  acc=0.513  ROI= -0.76%
2025  n=70  acc=0.686  ROI=+31.66%
2026  n=48  acc=0.604  ROI=+21.41%
```

Positive in both halves of the eval window. Picks split 140 red / 181 blue, so this is not
a disguised underdog bias.

---

## Why the picks model is not the model the site serves

Production serves `mlp_v1.pkl`, which takes betting odds as input features. That model
cannot measure edge against the market: it has already seen the line, so
`model_prob − market_prob` is partly the model echoing its own input. It correlates 0.83
with the closing line.

The picks model is trained without odds, so its disagreement with the market is a real
disagreement. This also matches the `no_odds` walk-forward arm the rule was measured on,
which is what makes live results comparable to the backtest.

## Why flat stakes

Hubáček & Šír (*Int. J. Forecasting* 2022), §3.1: under growth-optimal (Kelly) staking,
profitability requires strictly lower cross-entropy than the market. This model does not
have that — it is 6 points less accurate overall. The favourable result exists **only**
under uniform staking. Kelly sizing on an uncalibrated model with an unproven edge would
compound estimation error, not returns.

---

## What this backtest is NOT

**The +14.16% is not evidence of an edge.** It was produced by evaluating roughly 26 rule
variants across two models on the same eval set and reporting the ones whose confidence
interval excluded zero. With that many correlated comparisons, at least one such result is
expected by chance. Anyone quoting +14.16% as an expected return — including a future
version of this project — is misreading it.

What makes it worth testing rather than discarding:

1. The market-uncertainty gate came from a mechanism argument stated *before* the search.
2. The effect is monotone along both axes independently, not a single lucky cell.
3. The failure of the naive rule and the success of the gated one are the same coherent
   story, not two unrelated findings.

Those make it a reasonable hypothesis. They do not make it a result.

**The sample is also small.** At n=321 the 95% CI is roughly ±11 points, and the earlier
γ-ablation showed run-to-run swings larger than the seed noise implied. Treat every figure
here as soft.

---

## Success and stop conditions

Declared now so they cannot be adjusted later to fit whatever happens.

- **Success**: flat-stake ROI over **at least 150 settled live picks** whose 95% bootstrap
  CI excludes zero. At ~70 picks/year that is roughly two seasons.
- **Failure**: after 150 settled picks the CI includes zero, or the point estimate is
  negative. Report the null, retire the rule.
- **Below 150 picks, no conclusion may be drawn in either direction.** A hot start is not
  validation and a cold start is not refutation. The running ROI will be displayed, but it
  is not to be acted on.

**Forbidden without registering a new version and resetting the count to zero:** changing
either threshold, changing the staking scheme, changing the model, changing the book set,
adding a filter, or excluding a logged pick after the fact.

If the rule is changed, the old log stays and a new one begins. Prior picks never
retroactively join a revised rule.

---

## Audit trail

`picks_log.jsonl`, append-only and git-tracked. Each entry records the timestamp, rule
version, both probabilities, the edge, the odds taken, and the stake — written **before**
the fight. Settlement (`--settle`) fills in only `result` and `profit`; no other field may
be edited afterwards. Re-running the generator skips fights already logged, so the first
recorded opinion is the one that counts.

The git history of that file is the actual evidence. If it is ever rewritten, the
experiment is void and must restart.

This clause has been invoked twice, both on 2026-09-06 (v1.0 → v1.1 → v1.2) — see the note
at the top. Both invocations were at zero settled picks, which is the only situation in
which a restart costs nothing. It is not a precedent for editing the log once results
exist: after a pick settles, a rewrite destroys the evidence rather than correcting an
input, and the correct response to a defect found then is a new rule version logged
alongside the old one, not in place of it.
