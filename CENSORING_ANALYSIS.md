# Replicate Censoring / Aggregation — Investigation Notes

Working notes for addressing the round-2 review concern about how replicate
yield measurements are aggregated (median-of-non-zero-yields) before being
turned into reaction classes. Captures the key question, a bug found while
tracing the code, and the direction we're taking to answer it quantitatively.

## The reviewer concern

Two reviewers independently raised the same underlying issue in round 2:

- **Reviewer #1 (comment 1):** the aggregation rule takes the median of the
  *non-zero* replicate yields, silently excluding zero-yield ("failed")
  replicates. This is an outcome-dependent choice — data is dropped because
  of what it showed — and it isn't justified anywhere in the manuscript. A
  duplicate pair with one failed replicate and one positive replicate above
  the 40% threshold can be assigned the high-yield class despite an observed
  failure. Because these classes define BB–BB similarity, network edges,
  pilot selection, model inputs, and evaluation labels, this choice is
  load-bearing for essentially every downstream result.

- **Reviewer #3 (comments 1 and 2):** independently flagged the same
  concept from the code side. Comment 1 found that the released
  `src/dataset.py` didn't even implement the *stated* median rule (see bug
  below). Comment 2 quantified the effect of positive-only vs. all-replicate
  medians: 40 aggregated yields and 12 class labels change, and — more
  importantly — 99 of 100 six-pilot sets change between the two rules,
  meaning the choice materially reshapes experimental design, not just
  reported numbers.

Source docs: `Chem Review round 2.docx` (reviewer comments), `Response_v2.docx`
(prior response letter), `RMap_Manuscript_R2_v1.docx` ("Reaction Data
Generation" section — the passage under fire).

## The key question we're asking

This is a chicken-and-egg problem: we can't know how to censor or bin yield
outcomes into classes at the start of a campaign, yet how we slice those
classes materially shapes every downstream result (network edges, pilot
selection, LP predictions). There's no single provably-correct answer to
appeal to in advance.

What we *can* do is look at how the yield distributions shift under
different censoring/aggregation choices and let that shift justify (or
revise) the decision. The tails of the distribution — clearly dead-on-arrival
BBs vs. clearly highly reactive ones — are stable under any reasonable rule;
they aren't where the real risk is. The decision that matters is the boundary
between "moderately reactive" and "barely reactive," because that's the
terrain where a library ends up either full of dead building blocks or too
sparse to get usable SAR out of the downstream assay. The "best" aggregation
method, by this logic, is the one that creates the sharpest contrast right at
that boundary — i.e., maximizes separation between the moderate and marginal
classes rather than optimizing some global distributional statistic.

The goal is not to claim one universally correct censoring rule, but to make
the choice quantitatively: pick a default based on a stated, defensible
criterion (maximizing class separation where it matters, i.e. balancing the
risk of over-censoring against under-censoring), show how sensitive
conclusions are to that choice, and present it to reviewers as "here is one
way to think about this, chosen for balanced risk, and here is how flexible
it is" rather than an unexamined default.

## Current bug in the repo

`src/dataset.py`, `SuzukiDataset.__init__`, lines ~314–343 (aggregation
loop) and line 343 (`self.df = pd.DataFrame(each_row, columns=sub_df.columns).iloc[:, :-1]`).

The code computes the intended value correctly (median of positive replicate
yields, `y = np.median(pos_yvals)` at line 338), but then mislabels it away:

```python
row = list(sub_df.iloc[0, :-1].values) + [y]          # line 339
...
self.df = pd.DataFrame(each_row, columns=sub_df.columns).iloc[:, :-1]   # line 343
```

`sub_df.columns` ends in `[..., 'suzuki_product_CAD_yield', 'plate']`.
`sub_df.iloc[0, :-1]` grabs the first replicate's row *including its raw,
unaggregated yield* in the `suzuki_product_CAD_yield` position, then appends
the correct median `y` as a 6th value — which lands in the `'plate'`
column slot once `columns=sub_df.columns` is applied. `.iloc[:, :-1]` then
drops the *last* column (`'plate'`) — which is where the correct median
actually ended up — and keeps the mislabeled `suzuki_product_CAD_yield`
column, which holds the **first raw replicate's yield**, not the median.

This is exactly the bug Reviewer #3 found (comment 1) and that the authors'
response letter (`Response_v2.docx`) claims was fixed and used to regenerate
the entire paper. **The fix does not appear to be present in this repo** —
the code here matches the "previously" (buggy) snippet quoted in the
response letter, not the "corrected" one. Confirmed empirically by
replicating the exact column-order/labeling logic against the raw plate CSVs
in `data/*.csv`:

- 280 of 996 multi-replicate substrate pairs get the wrong retained value
  (vs. the reviewer's reported 268 of 897 — close match; the gap is expected
  since core-filtering and the 8b/010–013 stoichiometry-adjustment special
  case weren't replicated in the check).
- 25 of those change yield class (vs. reviewer's reported 23).

**Open question to resolve with coauthors:** is this repo simply behind the
fix described in the response letter, or did the fix not get committed? This
needs correcting regardless of the outcome of the censoring-policy question
below — it's a separate, purely mechanical bug.

## Data available

Everything needed to redo this analysis from raw replicates is in the repo —
nothing is missing relative to what was used for the publication:

- `data/{008b,010,011,012,013}_final_report.csv` — raw well-level data for
  the final 13-core × 69-BB combinatorial matrix (897 pairs), all replicates,
  across and within plates. Yield column: `suzuki_product_CAD_yield`.
- `data/preliminary/00[1-8]_final_report.csv` — earlier condition-screening
  rounds (different assay column, different reaction condition per round;
  not part of the final 897-pair dataset).
- `data/boronic_descriptors.csv`, `data/halide_descriptors.csv` — physical
  descriptors used for the RFC baseline.
- `src/dataset.py` — raw → aggregated collapsing (the code under dispute).
- `src/rmap.py`, `src/first_selection.py`, `src/second_selection.py`,
  `src/analyzer.py` — network construction, pilot selection, LP/RFC/baseline
  evaluation, figures.
- `saved_results/*.joblib` — cached outputs from a prior run; stale as soon
  as aggregation changes.

## Where we're aiming

1. Fix the mechanical bug in `dataset.py` (independent of the policy
   question).
2. Rebuild a fully raw, uncollapsed dataset (one row per replicate, tagged
   core/BB/plate/yield) directly from `data/*.csv`, bypassing
   `dataset.py`'s collapsing step for this analysis.
3. Define a small family of candidate aggregation/censoring rules to compare:
   positive-only median (current stated rule), all-replicate median (zeros
   included), mean, max/"any positive," and a probabilistic success-rate /
   replicate-confidence flag (as Reviewer #3 suggested) instead of a single
   point estimate.
4. Operationalize "contrast" quantitatively, focused on the moderate-vs-
   marginal boundary (not the tails): a separation/overlap metric between
   class-conditional yield distributions (e.g. Bhattacharyya distance, or a
   density-valley / gap statistic), evaluated per candidate rule.
5. Let the class threshold float with the aggregation rule rather than
   fixing 20%/40% a priori — see how much the natural density minimum moves
   under each rule.
6. Propagate each candidate rule downstream (pilot-set stability, per-core
   LP AUPRC) the same way Reviewer #3's own sensitivity check did, so the
   chosen rule is justified by more than distribution shape alone.
7. Write up as a risk-balance argument for the response letter: state the
   default and the criterion that picked it, show the sensitivity band, and
   frame it explicitly as one reasonable choice among a flexible family
   rather than an unexamined default.
