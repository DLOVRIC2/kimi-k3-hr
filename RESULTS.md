# Results

Four models, 200 Belebele items per language plus all 164 HumanEval problems each.
Greedy decoding (temperature 0), same prompts, per-backend token budgets sized so every
model can finish.

**Every number below is emitted by `uv run eval/analysis.py`** from the per-item records.
Nothing here is typed in by hand — an earlier draft of this file was, and several of its
figures were wrong (`LOG.md` #25–#28).

Two corrections separate this from the first published version, and both were found
*after* the run finished, by writing tooling to reproduce numbers that already existed:

1. **The language comparison was not paired.** Belebele is a parallel benchmark — the
   same 900 passages in every language — but the sweep sampled each language
   independently and overlapped on 48 of 200 items.
2. **The echo detector had no minimum length.** A bare `2` was classified as a
   restatement of the passage whenever `2` appeared in its first 200 characters, and
   discarded. This misclassified 140 items and, for the three comparators, was 100% of
   their reported echoes.

Both are quantified in [What the two corrections cost](#what-the-two-corrections-cost).
Original and corrected data are both kept: `results/full/` (as run), `results/paired/`
(English re-scored on the Croatian item set), `results/rescored/` (corrected echo rule).

## Headline

| model | on disk | Croatian | English | gap | HumanEval | peak RSS | tok/s | wall |
|---|---|---|---|---|---|---|---|---|
| **K3-REAP80-hr-code** | 326 GB | **45.0%** | 38.0% | −7.0 | 25.0% | **325 GB** | 5.4 | 133 min |
| gpt-oss:120b | 65 GB | 91.0% | 96.0% | +5.0 | 92.1% | 74 GB | 76.6 | 44 min |
| gemma4:31b | 19 GB | **93.0%** | 95.5% | +2.5 | **98.2%** | 56 GB | 28.0 | 32 min |
| gpt-oss:20b | 13 GB | 89.0% | 94.5% | +5.5 | 92.7% | **25 GB** | **106.8** | 39 min |

Chance on Belebele is 25%.

**The pruned 326 GB build loses to a 13 GB one on every axis.** Against gemma4:31b —
17× smaller on disk, 6× smaller resident, 5× faster — K3 is worse by 48 points on
Croatian (p=3e-25), 57.5 on English (p=3e-34), and 73 on HumanEval (p=3e-42). There is no
reading of this where the pruned model is competitive as a system.

## Finding 1 — the damage is to response initiation, not capability

K3 frequently produces **no answer at all**: the first token it predicts is a stop
token. Splitting each score into "did it respond" and "was it right when it did":

| | responded | correct *when* it responded |
|---|---|---|
| **K3** Croatian | 85.5% | 52.6% |
| **K3** English | 68.0% | 55.9% |
| **K3** HumanEval | 49.4% | **50.6%** |
| gpt-oss:120b (all three tasks) | 100% | 91–96% |
| gemma4:31b (all three tasks) | 100% | 93–98% |
| gpt-oss:20b (all three tasks) | 98–100% | 91–95% |

On HumanEval the split is exact: **83 of 164 prompts produced zero tokens and scored 0;
the other 81 produced code and passed 50.6%.** The three comparators produced output on
essentially 100% of prompts and, after the echo fix, never once restated a passage
instead of answering it.

So roughly half of K3's competence survives pruning intact. What collapses is the
ability to *begin* a response. The same failure appears in three other places:

- the smoke test wrote fluent, grammatical Croatian prose while scoring near chance on
  Croatian comprehension
- thinking mode intermittently emitted no `<|open|>response<|sep|>` markup at all,
  rambling instead of opening its own response channel
- 8–23% of K3's Belebele answers are echoes — the passage restated rather than answered —
  against 0% for every comparator

**Raw pass@1 of 25% reads as "this model cannot code." It codes at 51% and stays silent
half the time.** Those are different failures and only one is about coding. Any
benchmark reporting a single aggregate hides this entirely.

Full generations for each failure mode — including the zero-token responses, with the
same prompts answered by gemma4:31b for contrast — are in `results/examples/`. The three
silent items reproduced as silent on re-run, which under greedy decoding confirms a
stable property of the model rather than a transient.

## Finding 2 — the Croatian targeting moved compliance, not comprehension

**This is materially weaker than the first draft claimed, and in a more interesting
direction.** That draft reported K3 as 16 points *better* at Croatian than English. On
matched items, with the echo rule fixed, the accuracy gap is −7.0 and not significant:

| measure (paired, n=200) | Croatian | English | p (McNemar) |
|---|---|---|---|
| **responded at all** | 85.5% | 68.0% | **5.1e-05** *** |
| **echoed the passage** | 8.0% | 22.5% | **8.2e-05** *** |
| accuracy | 45.0% | 38.0% | 0.13 n.s. |
| accuracy *given* it responded | 52.6% | **55.9%** | — |

Conditional on answering at all, K3 is marginally *better* in English. The Croatian
calibration corpus did not buy Croatian comprehension. What it bought is **willingness
to respond in Croatian** — +17.5 points of response rate and 14.5 points less
passage-echoing, on identical passages.

This is not an artifact of Croatian being easier, because the comparators go the other
way on accuracy:

| model | Croatian | English | gap | p (McNemar) |
|---|---|---|---|---|
| K3-REAP80-hr-code | 45.0% | 38.0% | −7.0 | 0.13 n.s. |
| gpt-oss:120b | 91.0% | 96.0% | +5.0 | 0.013 * |
| gemma4:31b | 93.0% | 95.5% | +2.5 | 0.18 n.s. |
| gpt-oss:20b | 89.0% | 94.5% | +5.5 | 0.019 * |

And on response rate the difference is not merely an absence but a reversal: K3 leans
Croatian by 17.5 points while every comparator sits at 0–1.5 points in the other
direction, at a ceiling of ~100%. A per-item sign test on `d_K3 − d_comparator` confirms
the asymmetry is specific to this build:

```
K3 vs gpt-oss:120b   K3 more Croatian-favouring on 54 items, less on 19   p 5.06e-05  ***
K3 vs gemma4:31b     K3 more Croatian-favouring on 54 items, less on 19   p 5.06e-05  ***
K3 vs gpt-oss:20b    K3 more Croatian-favouring on 55 items, less on 19   p 3.38e-05  ***
```

Read alongside Finding 1, these are the same result. Pruning damages response initiation
rather than comprehension, and **the calibration corpus determines which languages
retain the ability to initiate.** Our corpus was 22% Croatian and 16% English, and
Croatian is where the model still answers.

What this does *not* establish is that the Croatian data caused it, rather than pruning
damage generally — that needs the control arm in limitation 1.

## Finding 3 — cost

| model | peak RSS | peak wired | tok/s | sustained page-in |
|---|---|---|---|---|
| K3-REAP80-hr-code | 325 GB | 469 GB | 5.4 | 0.040 GB/min |
| gpt-oss:120b | 74 GB | 78 GB | 76.6 | 0.007 |
| gemma4:31b | 56 GB | 55 GB | 28.0 | 0.007 |
| gpt-oss:20b | 25 GB | 28 GB | 106.8 | 0.009 |

All four ran fully resident — no thrashing (sustained page-in below 0.05 GB/min
throughout; the threshold for concern is ~0.5). Compressed memory held at 3.4 GB for
every arm, so none of these numbers are contaminated by memory pressure.

K3 needs **6× the resident memory and 17× the disk of gemma4:31b to run 5× slower and
score worse on every task.**

> These tok/s figures supersede an earlier draft that reported 4.3 for K3 and 44.5 for
> gemma4:31b. That version averaged per-item rates, which counts an item that generated
> nothing as "0 tokens per second" — and K3 generated nothing on 83 of 164 coding
> prompts. Total tokens over total decode seconds is the honest aggregate; it moved K3
> up and gemma4:31b down.

## What the two corrections cost

Isolating them matters, because quoting an old-rule unpaired number against a new-rule
paired one would credit the entire shift to whichever correction was under discussion.

**Correction 1 — unmatched item sets.** Both columns below use the corrected echo rule;
only the item set differs.

| model | gap, unpaired items | gap, paired items | shift |
|---|---|---|---|
| K3-REAP80-hr-code | −14.5 | −7.0 | +7.5 |
| gpt-oss:120b | +6.5 | +5.0 | −1.5 |
| gemma4:31b | +4.0 | +2.5 | −1.5 |
| gpt-oss:20b | +7.0 | +5.5 | −1.5 |

`shuffle(seed)` fixes the permutation, not the outcome — each language config stores its
rows in a different order — so the two languages were scored on different passages. The
English items it drew were harder for every model, which inflated K3's apparent Croatian
advantage and masked the comparators' real English advantage at the same time. Half of
the original −16.0 was this.

**Correction 2 — the echo rule had no minimum length.** Same items, same responses,
different classification.

| model | Croatian, old rule | corrected | English, old rule | corrected |
|---|---|---|---|---|
| K3-REAP80-hr-code | 40.5% | 45.0% | 24.5% | 30.5% |
| gpt-oss:120b | 81.0% | 91.0% | 82.0% | 97.5% |
| gemma4:31b | 82.5% | 93.0% | 81.5% | 97.0% |
| gpt-oss:20b | 78.5% | 89.0% | 80.5% | 96.0% |

It suppressed accuracy for every model and manufactured an echo rate for three models
that never echoed once. Note the direction: correcting it raises the comparators by
8–11 points against K3's 3–6, so it *widens* the gap this report is about. Fixing it
made our own result look worse, which is the only reason to trust the other corrections.

Rescoring required no GPU time. The checkpoint happens to store `raw` as exactly
`text.strip()[:40]` and the echo rule reads `head[:40]`, so the stored prefix is
precisely the rule's input and the decision reproduces exactly. 253 items reclassified,
230 correct answers recovered, none skipped.

## Is the build sound?

The conversion is verified. `scripts/verify.py` from the toolchain compares the artifact
against the source without loading it: 59/59 shards mapped, zero missing tensors, zero
orphans, and 24 sampled experts across layers 8–86 dequantized and compared against the
same experts in the source — **all bit-exact, `cos 1.00000`, worst error 0.0** — with the
expert remapping followed through the keep map. `build/verify.log` is committed.

That rules out the failure this result would otherwise be most likely to be: a router
renumbering fault, where surviving experts are reindexed but the router still points at
the old rows. It produces a model that loads, generates fluent text, and routes every
token to the wrong specialists — indistinguishable from our result by inspection. It is
not what happened.

It does **not** rule out that the chat and generation path is subtly wrong, and that is
where the residual risk sits: the entire Finding 1 claim is about *generation* behaviour
(the first token predicted is a stop token), which depends on the chat template, the
control-token ids and the stop set. Two of this project's worst bugs lived in exactly
that code. The measurement that would route around it is held-out perplexity, which
bypasses the chat stack entirely and buckets by source language — but the calibration
corpus was sized to what calibration consumed, leaving 844 held-out tokens against the
65,536 required (`LOG.md` #29). It needs a fresh corpus first.

**No unpruned baseline exists anywhere.** The smallest full K3 tier is 883 GB against 512
GB of unified memory, so no unpruned tier has ever produced a token on this hardware —
not here, and not in the toolchain that published these builds. "How much did pruning
cost" is not directly answerable. The nearest control is the published
`Kimi-K3-REAP80-MLX-mxfp4-q8` (179/896 experts, mxfp4+q8 — the same ratio and profile as
ours, differing only in calibration corpus), run through this harness. That is
limitation 1.

## Known limitations

1. **No control arm.** Two candidates, and the external one is now clearly better.
   `reap_subset --keep-sources code,en,zh,de,ru,fr` derives the reference mix from the
   *same* calibration run — cheap, and it isolates the corpus effect, but it shares our
   entire pipeline so it cannot tell us whether the absolute scores are real. Running
   the **published `Kimi-K3-REAP80-MLX-mxfp4-q8`** through this harness tests both at
   once: same prune ratio, same quantization profile, different corpus, third-party
   build. If it scores ~85% here, the collapse is ours; if it scores ~40%, the collapse
   is REAP's at this ratio. **This is the gap between "a model" and "a claim."**
2. **`hr-heavy` and `hr-only` arms unbuilt**, so the language-plus-code question — does
   Croatian+code beat Croatian-only, as Chinese+code beat Chinese-only in the published
   ablation — is untested.
3. **Thinking mode is under-measured** at n=20. Enough to rule out the confound that
   K3's weak scores were an artifact of running it non-thinking, not enough to
   characterise thinking-mode behaviour.
4. **One build, one seed.** No variance estimate across prune plans.
5. **Response rate is measured, not explained.** We can show K3 stays silent far more
   often in English, but not why — whether the stop token is being predicted at the
   first position, or the response channel never opens. Logit inspection at position 0
   would settle it and costs one short run.
6. **Echo detection is still a heuristic.** A 20-character floor removes the false
   positives that mattered, but "restated the passage" has no exact definition. The
   comparators' 0% is a useful sanity check that it is no longer firing spuriously.

## Reproducing

```bash
uv sync
$K3_MLX/scripts/install_model.sh .venv/bin/python

bash eval/sweep.sh                # ~5h, resumable, writes results/full/
bash eval/pair_up.sh              # ~55m, re-scores English on the Croatian item set
uv run eval/rescore.py            # seconds, no GPU: corrected echo rule
uv run eval/analysis.py           # regenerates every table above
uv run eval/capture_examples.py   # full text for a stratified sample of items
```

See `LOG.md` for the 29 bugs found on the way here, several of which produced
plausible-looking numbers that were wrong — including four found after the run finished,
by building the tooling to reproduce its own results.
