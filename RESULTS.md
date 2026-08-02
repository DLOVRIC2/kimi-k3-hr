# Results

Full sweep, 2026-08-02. Four models, 200 Belebele items per language plus all 164
HumanEval problems each. Greedy decoding (temperature 0), same prompts, per-backend
token budgets sized so every model can finish. Raw per-item records in
`results/full/checkpoints/`, memory traces in `results/full/memtrace-*.jsonl`.

## Headline

| model | on disk | Croatian | English | gap | HumanEval | peak RSS | tok/s | wall |
|---|---|---|---|---|---|---|---|---|
| **K3-REAP80-hr-code** | 326 GB | **40.5%** | 24.5% | **−16.0** | 25.0% | **325 GB** | 4.3 | 133 min |
| gpt-oss:120b | 65 GB | 81.0% | 82.0% | +1.0 | 92.1% | 74 GB | 76.7 | 44 min |
| gemma4:31b | 19 GB | 82.5% | 81.5% | −1.0 | **98.2%** | 56 GB | 44.5 | 32 min |
| gpt-oss:20b | 13 GB | 78.5% | 80.5% | +2.0 | 92.7% | **25 GB** | **107.1** | 39 min |

Chance on Belebele is 25%.

**The pruned 326 GB build loses to a 13 GB one on every axis.** Against gemma4:31b —
17× smaller on disk, 6× smaller resident, 10× faster — K3 is worse by 42 points on
Croatian (p=6e-18), 57 on English (p=3e-30), and 73 on HumanEval (p=3e-42). There is no
reading of this where the pruned model is competitive as a system.

## Finding 1 — the damage is to response initiation, not capability

K3 frequently produces **no answer at all**: the first token it predicts is a stop
token. Splitting each score into "did it respond" and "was it right when it did":

| | responded | correct *when* it responded |
|---|---|---|
| **K3** Croatian | 78.5% | 51.6% |
| **K3** English | 61.5% | 39.8% |
| **K3** HumanEval | 49.4% | **50.6%** |
| gemma4:31b Croatian | 89.5% | 92.2% |
| gemma4:31b HumanEval | 100% | 98.2% |
| gpt-oss:120b HumanEval | 100% | 92.1% |
| gpt-oss:20b HumanEval | 100% | 92.7% |

On HumanEval the split is exact: **50 of 99 sampled prompts produced zero tokens and
scored 0; the other 49 produced code and passed 53%.** The three comparators responded
to 100% of prompts.

So roughly half of K3's competence survives pruning intact. What collapses is the
ability to *begin* a response. This is the same failure seen in three other places:

- the smoke test wrote fluent, grammatical Croatian prose while scoring near chance on
  Croatian comprehension
- thinking mode intermittently emitted no `<|open|>response<|sep|>` markup at all,
  rambling instead of opening its own response channel
- 15–29% of Belebele answers are echoes — the passage restated rather than answered

**Raw pass@1 of 25% reads as "this model cannot code." It codes at 51% and stays silent
half the time.** Those are different failures and only one is about coding. Any
benchmark reporting a single aggregate hides this entirely.

## Finding 2 — the Croatian targeting worked, and it is the only asymmetry in the set

K3 is the **only model that is better at Croatian than English**, by 16 points:

```
K3-REAP80-hr-code   hrv 40.5%  eng 24.5%   gap -16.0   z  3.42   p 0.00064  ***
gemma4:31b          hrv 82.5%  eng 81.5%   gap  -1.0   z  0.26   p 0.79     n.s.
gpt-oss:120b        hrv 81.0%  eng 82.0%   gap  +1.0   z -0.26   p 0.80     n.s.
gpt-oss:20b         hrv 78.5%  eng 80.5%   gap  +2.0   z -0.50   p 0.62     n.s.
```

All three comparators sit at zero gap, so this is not a property of the benchmark or of
Croatian being unusually easy — it is a property of **this build**. Our calibration
corpus was 22% Croatian and 16% English, and the resulting model inverted the ordering
that every general model shows.

It holds after conditioning on compliance too — 51.6% vs 39.8% among items that produced
a parseable answer — so it is not merely a parse-rate artifact.

The retention analysis predicted degradation rather than death (Croatian retaining 54.2%
under a reference-mix plan against an 82.5% own-distribution ceiling). At 40.5% against a
25% floor, that is what happened.

## Finding 3 — cost

| model | peak RSS | peak wired | tok/s | sustained page-in |
|---|---|---|---|---|
| K3-REAP80-hr-code | 325 GB | 469 GB | 4.3 | 0.040 GB/min |
| gpt-oss:120b | 74 GB | 78 GB | 76.7 | 0.007 |
| gemma4:31b | 56 GB | 55 GB | 44.5 | 0.007 |
| gpt-oss:20b | 25 GB | 28 GB | 107.1 | 0.009 |

All four ran fully resident — no thrashing (sustained page-in below 0.05 GB/min
throughout; the threshold for concern is ~0.5). Compressed memory held at 3.4 GB for
every arm, so none of these numbers are contaminated by memory pressure.

K3 needs **13× the memory of gemma4:31b to run 10× slower and score worse.**

## Known limitations

1. **The language comparison is unpaired.** Belebele's design is fully parallel — the
   same 900 items in every language — which is its main methodological value. Sampling
   200 items per language independently produced only **48 items in common**, so the
   headline gap rests on an unpaired two-proportion test rather than the much tighter
   paired McNemar. The gap is significant anyway (p=0.00064), but a matched-subset rerun
   would be strictly better and costs ~35 minutes for K3 plus ~30 for the comparators.
2. **No control arm.** `reap_subset --keep-sources code,en,zh,de,ru,fr` reproduces the
   reference mix from the *same* calibration run. Without it, Finding 2 shows that this
   build favours Croatian, but not how much of that is attributable to the calibration
   corpus versus to pruning damage generally.
3. **`hr-heavy` and `hr-only` arms unbuilt**, so the language-plus-code question — does
   Croatian+code beat Croatian-only, as Chinese+code beat Chinese-only in the published
   ablation — is untested.
4. **Thinking mode is under-measured** at n=20 (35.0%). Enough to rule out the confound
   that K3's weak scores were an artifact of running it non-thinking, not enough to
   characterise thinking-mode behaviour.
5. **One build, one seed.** No variance estimate across prune plans.

## Reproducing

```bash
uv sync
$K3_MLX/scripts/install_model.sh .venv/bin/python
bash eval/sweep.sh     # ~5h, resumable
```

See `LOG.md` for the 22 bugs found on the way here, several of which produced
plausible-looking numbers that were wrong.
