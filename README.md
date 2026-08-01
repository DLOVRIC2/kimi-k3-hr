# kimi-k3-hr

Croatian-targeted expert pruning of Kimi K3, on a single Mac Studio (M3 Ultra, 512GB).

## What this is

Kimi K3 is 2.8T parameters — too large for any single Apple silicon machine, even at
1-bit. The only way to run it locally is REAP expert pruning: score every expert against
a calibration corpus, delete the low scorers.

Which means **the calibration corpus decides what the model can still do.**

The published builds used roughly 40% code, 30% English web, 15% Chinese, and 15% split
across ja/ru/ko/de/fr/es/ar. **Croatian appears nowhere in it.** So the experts serving
Croatian scored low and were deleted.

This repo tests whether calibrating on Croatian recovers that — and measures what it
costs elsewhere.

## The questions

1. **Does Croatian actually die in the published builds?** Expected but unverified.
2. **Does Croatian ride the Slavic/European cluster?** German-Spanish overlapped at
   2.2x chance, so related languages share experts. Russian *is* in the reference mix.
   Croatian may partially survive on its neighbours' coattails rather than dying outright.
   Nobody has measured this.
3. **Does language+code beat language-only?** Their own ablation found Chinese+code beat
   Chinese-only for Chinese generation. If that replicates on a Slavic low-resource
   language, it's a general result rather than a quirk of Chinese.

## Arms

| Recipe | Mix | Role |
|---|---|---|
| `control.yaml` | 40% code / 30% en / 15% zh / 15% other | reproduces the reference mix — the baseline |
| `hr-code.yaml` | 35% hr / 35% code / 15% sr / 15% en | the hypothesis |
| `hr-heavy.yaml` | 70% hr / 15% sr / 10% code / 5% en | ablation — should *underperform* hr-code if their finding generalises |

The control arm matters. Without it there is no claim, only a model.

## Method

```bash
# 1. build a tagged, interleaved corpus
uv run corpus/build_corpus.py corpus/recipes/hr-code.yaml --out build/hr-code

# 2. score every expert against it  (~30min-2h, 58GB peak RAM at 128x2048)
python scripts/reap_calibrate.py --src $K3_SRC --out build/hr-code.saliency.npz \
    --calib-text build/hr-code.txt --seqs 128 --seqlen 2048

# 3. choose survivors
python scripts/reap_plan.py --saliency build/hr-code.saliency.npz \
    --mode uniform --out build/hr-code.plan.json

# 4. build it  (~30-90min)
python scripts/convert.py --src $K3_SRC --out models/K3-REAP80-hr-code \
    --profile mxfp4 --prune-plan build/hr-code.plan.json

# 5. probe it
uv run eval/run_probes.py --model models/K3-REAP80-hr-code --probes eval/probes.yaml
```

Steps 2-4 use the toolchain from [PipeNetwork/kimi-k3-mlx](https://github.com/PipeNetwork/kimi-k3-mlx).
This repo supplies the corpus design, the probe set, and the measurements.

**Calibration runs once per corpus; conversion runs per build.** Tagged buckets in
`*.tags.jsonl` let `reap_subset.py` re-target a different mix without re-calibrating —
so after the first run, each new variant is roughly an hour.

## Scoring

Seven domains, in `eval/probes.yaml`, scored 0-5 by hand and **never averaged**. The
spread between domains is the entire result — a mean is precisely what hides selective
pruning damage.

`python` is the control that should survive any prune. `croatian` is the probe aimed at
the gap in the reference corpus. If the first stays clean while the second collapses,
that is the thesis demonstrated on one model in one run.

## Two gotchas encoded in the tooling

- **Corpus order is significant.** Calibration reads only the first `seqs x seqlen`
  tokens. Concatenated sources mean the tail domains are never seen. `build_corpus.py`
  interleaves round-robin and reports the achieved composition of the first quarter, not
  just the whole file.
- **Pooled multilingual datasets lie.** C4's pooled multilingual config measured 97%
  Latin script over 200 documents. Every recipe here uses per-language named configs.

## Licence

Kimi K3 is released under an MIT-structured licence that explicitly permits derivative
works. Restrictions apply to Model-as-a-Service operators above $20M monthly revenue or
100M MAU, who need a separate agreement with Moonshot and must display "Kimi K3" in
their UI. Neither threshold applies here.

Note Moonshot consistently say "open weight", never "open source". Worth matching that
language in any writeup.
