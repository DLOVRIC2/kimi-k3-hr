# Results

Four models, 200 Belebele items per language plus all 164 HumanEval problems each.
Greedy decoding (temperature 0), same prompts, per-backend token budgets sized so every
model can finish. Raw per-item records in `results/full/checkpoints/` and
`results/paired/checkpoints/`; memory traces alongside them.

**Every number below is emitted by `uv run eval/analysis.py`** from the per-item records.
Nothing here is typed in by hand — an earlier draft of this file was, and two of its
figures were wrong (see `LOG.md` #25–#27).

Belebele is a parallel benchmark: the same 900 passages exist in every language, so the
same items are scored in Croatian and English and the comparison is paired (McNemar).
The first version of this analysis sampled each language independently and got only 48
items in common; correcting that changed the headline finding. See
[What the unpaired sampling reported](#what-the-unpaired-sampling-reported).

## Headline

| model | on disk | Croatian | English | gap | HumanEval | peak RSS | tok/s | wall |
|---|---|---|---|---|---|---|---|---|
| **K3-REAP80-hr-code** | 326 GB | **40.5%** | 35.0% | −5.5 | 25.0% | **325 GB** | 5.4 | 133 min |
| gpt-oss:120b | 65 GB | 81.0% | 88.0% | +7.0 | 92.1% | 74 GB | 76.6 | 44 min |
| gemma4:31b | 19 GB | 82.5% | 87.5% | +5.0 | **98.2%** | 56 GB | 28.0 | 32 min |
| gpt-oss:20b | 13 GB | 78.5% | 86.5% | +8.0 | 92.7% | **25 GB** | **106.8** | 39 min |

Chance on Belebele is 25%.

**The pruned 326 GB build loses to a 13 GB one on every axis.** Against gemma4:31b —
17× smaller on disk, 6× smaller resident, 5× faster — K3 is worse by 42 points on
Croatian (p=6e-18), 52.5 on English (p=4e-27), and 73 on HumanEval (p=3e-42). There is no
reading of this where the pruned model is competitive as a system.

## Finding 1 — the damage is to response initiation, not capability

K3 frequently produces **no answer at all**: the first token it predicts is a stop
token. Splitting each score into "did it respond" and "was it right when it did":

| | responded | correct *when* it responded |
|---|---|---|
| **K3** Croatian | 78.5% | 51.6% |
| **K3** English | 63.5% | 55.1% |
| **K3** HumanEval | 49.4% | **50.6%** |
| gemma4:31b Croatian | 89.5% | 92.2% |
| gemma4:31b HumanEval | 100% | 98.2% |
| gpt-oss:120b HumanEval | 100% | 92.1% |
| gpt-oss:20b HumanEval | 100% | 92.7% |

On HumanEval the split is exact: **83 of 164 prompts produced zero tokens and scored 0;
the other 81 produced code and passed 50.6%.** The three comparators responded to 100%
of prompts.

So roughly half of K3's competence survives pruning intact. What collapses is the
ability to *begin* a response. This is the same failure seen in three other places:

- the smoke test wrote fluent, grammatical Croatian prose while scoring near chance on
  Croatian comprehension
- thinking mode intermittently emitted no `<|open|>response<|sep|>` markup at all,
  rambling instead of opening its own response channel
- 15–27% of Belebele answers are echoes — the passage restated rather than answered

**Raw pass@1 of 25% reads as "this model cannot code." It codes at 51% and stays silent
half the time.** Those are different failures and only one is about coding. Any
benchmark reporting a single aggregate hides this entirely.

Sample generations for each failure mode — including the zero-token responses, with the
same prompts answered by gemma4:31b for contrast — are in `results/examples/`.

## Finding 2 — the Croatian targeting moved compliance, not comprehension

**This finding is materially weaker than the first draft of this document claimed, and
in a more interesting direction.** The original claim was that K3 was 16 points *better*
at Croatian than English. On matched items that gap is −5.5 and not significant:

| measure (paired, n=200) | Croatian | English | p (McNemar) |
|---|---|---|---|
| **responded at all** | 78.5% | 63.5% | **0.00077** *** |
| **echoed the passage** | 15.0% | 27.0% | **0.0027** ** |
| accuracy | 40.5% | 35.0% | 0.228 n.s. |
| accuracy *given* it responded | 51.6% | **55.1%** | — |

Conditional on answering at all, K3 is marginally *better* in English. The Croatian
calibration corpus did not buy Croatian comprehension. What it bought is **willingness
to respond in Croatian** — +15 points of response rate and 12 points less
passage-echoing, on identical passages.

That is not an artifact of Croatian being easier, because the comparators go the other
way. On accuracy, all three general models are significantly better at English:

| model | Croatian | English | gap | p (McNemar) |
|---|---|---|---|---|
| K3-REAP80-hr-code | 40.5% | 35.0% | −5.5 | 0.228 n.s. |
| gpt-oss:120b | 81.0% | 88.0% | +7.0 | 0.0013 ** |
| gemma4:31b | 82.5% | 87.5% | +5.0 | 0.0064 ** |
| gpt-oss:20b | 78.5% | 86.5% | +8.0 | 0.00086 *** |

**K3 is the only model in the set without a significant English advantage** — a relative
swing of 10.5 to 13.5 points against the comparators. And on response rate the
difference is not merely an absence but a reversal: K3 leans Croatian by +15.0 points
where every comparator leans English by 2.0–4.5. A per-item sign test on
`d_K3 − d_comparator` confirms the asymmetry is specific to this build:

```
K3 vs gpt-oss:120b   K3 more Croatian-favouring on 57 items, less on 22   p 0.000103  ***
K3 vs gemma4:31b     K3 more Croatian-favouring on 56 items, less on 22   p 0.000149  ***
K3 vs gpt-oss:20b    K3 more Croatian-favouring on 59 items, less on 22   p 4.77e-05  ***
```

Read alongside Finding 1, the two results are the same result. Pruning damages response
initiation rather than comprehension, and **the calibration corpus determines which
languages retain the ability to initiate.** Our corpus was 22% Croatian and 16% English,
and Croatian is where the model still answers.

What this does *not* establish is that the Croatian data caused it, rather than pruning
damage generally — that needs the control arm in limitation 2.

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

## What the unpaired sampling reported

The first sweep drew 200 items per language with the same seed. `shuffle(seed)` fixes
the permutation, not the outcome — each language config stores its rows in a different
order — so the two languages were scored on **different passages, overlapping in only 48
of 200**. Re-scoring English on the Croatian item set moves every model:

| model | gap, unpaired | gap, paired (n=200) | shift |
|---|---|---|---|
| K3-REAP80-hr-code | −16.0 | −5.5 | +10.5 |
| gpt-oss:120b | +1.0 | +7.0 | +6.0 |
| gemma4:31b | −1.0 | +5.0 | +6.0 |
| gpt-oss:20b | +2.0 | +8.0 | +6.0 |

The English set drawn by the original sweep was harder for every model, by about six
points. That single fact did two things at once: it inflated K3's apparent Croatian
advantage from −5.5 to −16.0, *and* it masked the comparators' real English advantage,
flattening +5 to +8 down to roughly zero. The result was a table where K3 looked
uniquely Croatian-favouring against a set of models that looked perfectly balanced —
and both halves of that picture were wrong.

The correction cost 53 minutes of compute, because the Croatian side did not need
rerunning and 48 English items were already in hand. Both datasets are kept:
`results/full/` is the original, `results/paired/` is the correction.

## Known limitations

1. **No control arm.** `reap_subset --keep-sources code,en,zh,de,ru,fr` reproduces the
   reference mix from the *same* calibration run. Without it, Finding 2 shows that this
   build initiates in Croatian where general models do not, but cannot attribute that to
   the Croatian calibration rather than to pruning damage generally. **This is the gap
   between "a model" and "a claim,"** and it is now the single most valuable next
   experiment.
2. **`hr-heavy` and `hr-only` arms unbuilt**, so the language-plus-code question — does
   Croatian+code beat Croatian-only, as Chinese+code beat Chinese-only in the published
   ablation — is untested.
3. **Thinking mode is under-measured** at n=20 (35.0% Croatian, against 40.5%
   non-thinking). Enough to rule out the confound that K3's weak scores were an artifact
   of running it non-thinking, not enough to characterise thinking-mode behaviour.
4. **One build, one seed.** No variance estimate across prune plans.
5. **Response rate is measured, not explained.** We can show K3 stays silent far more
   often in English, but not why — whether the stop token is being predicted at the
   first position, or the response channel never opens. Logit inspection at position 0
   would settle it and costs one short run.

## Reproducing

```bash
uv sync
$K3_MLX/scripts/install_model.sh .venv/bin/python

bash eval/sweep.sh          # ~5h, resumable, writes results/full/
bash eval/pair_up.sh        # ~55m, re-scores English on the Croatian item set
uv run eval/analysis.py     # regenerates every table above
uv run eval/capture_examples.py   # full text for a stratified sample of items
```

See `LOG.md` for the 27 bugs found on the way here, several of which produced
plausible-looking numbers that were wrong — including three found by writing the
analysis script itself.
