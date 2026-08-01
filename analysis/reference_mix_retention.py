#!/usr/bin/env python3
"""Does a build calibrated WITHOUT Croatian still preserve Croatian's experts?

reap_overlap.py reports retention against "the ACTUAL mixed build" -- the
token-weighted sum of every source in the corpus that was calibrated. For the
survey corpus that baseline is 24.6% Croatian, so it cannot answer the question
this project exists to answer: the published REAP builds contain NO Croatian.

This script builds the counterfactual instead. It selects the top-N experts using
ONLY the sources present in the reference calibration mix (code, en, zh, de, ru,
fr -- deliberately excluding hr and sr), then measures how much of Croatian's own
saliency mass those experts capture.

Interpretation:
  retention near hr's own ceiling  -> Croatian survives on its neighbours'
                                      experts; a targeted build buys little
  retention far below              -> Croatian's experts get deleted; a targeted
                                      build is necessary

    uv run analysis/reference_mix_retention.py --saliency build/survey.saliency.npz
"""

from __future__ import annotations

import argparse

import numpy as np

# The reference mix used by the published REAP builds: 40% code, 30% English web,
# 15% Chinese, 15% split across ja/ru/ko/de/fr/es/ar. Of our eight tagged sources,
# these are the ones that mix contains. hr and sr are absent from it.
REFERENCE_SOURCES = ["code", "en", "zh", "de", "ru", "fr"]

# Summing per_source raw would weight each source by ITS OWN token count in the
# survey corpus, which is not the reference mix. That distinction is not cosmetic:
# our corpus is 12.4% Russian, whereas the reference mix gives Russian one seventh
# of its 15% "other languages" slice -- about 2.1%. Russian is Croatian's nearest
# neighbour (73.9% expert overlap), so summing raw over-weights precisely the
# language most likely to rescue Croatian, and overstates its survival.
#
# These are the reference mix's own proportions, renormalised over the subset of
# it we actually have.
REFERENCE_WEIGHTS = {
    "code": 0.40,
    "en": 0.30,
    "zh": 0.15,
    "ru": 0.15 / 7,   # one of ja/ru/ko/de/fr/es/ar
    "de": 0.15 / 7,
    "fr": 0.15 / 7,
}


def unit_norm(a: np.ndarray) -> np.ndarray:
    mass = a.sum(-1, keepdims=True)
    return np.divide(a, mass, out=np.zeros_like(a), where=mass > 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saliency", required=True)
    ap.add_argument("--keep", type=int, default=242)
    ap.add_argument("--target", default="hr")
    args = ap.parse_args()

    z = np.load(args.saliency)
    labels = [str(x) for x in z["source_labels"]]
    moe = [int(i) for i in z["moe_layers"]]
    per = z["per_source"][:, moe, :]
    N = args.keep
    NE = int(z["num_experts"])
    S, L, _ = per.shape

    idx = {l: i for i, l in enumerate(labels)}
    ref = [s for s in REFERENCE_SOURCES if s in idx]
    missing = [s for s in REFERENCE_SOURCES if s not in idx]

    print(f"top-{N} of {NE} experts, {L} MoE layers")
    print(f"reference-mix sources : {ref}")
    if missing:
        print(f"  (not in corpus, ignored: {missing})")
    print(f"excluded from selection: {[l for l in labels if l not in ref]}")
    print(f"chance retention        : {N/NE:.1%}\n")

    norm = unit_norm(per)

    # Plan A: chosen by the reference mix only (the published-build counterfactual),
    # REWEIGHTED to the reference proportions rather than our corpus's token counts.
    tokens = {l: float(t) for l, t in zip(labels, z["source_tokens"])}
    wsum = sum(REFERENCE_WEIGHTS[s] for s in ref)
    print(f"  {'source':<8}{'our share':>11}{'ref share':>11}{'scale':>9}")
    ref_sum = np.zeros_like(per[0])
    tot_tok = sum(tokens.values())
    for s in ref:
        want = REFERENCE_WEIGHTS[s] / wsum
        have = tokens[s] / tot_tok
        scale = want / have
        print(f"  {s:<8}{have:>10.1%}{want:>11.1%}{scale:>9.2f}x")
        ref_sum += per[idx[s]] * scale
    print()
    ref_top = np.argsort(-unit_norm(ref_sum), axis=-1)[:, :N]

    # Plan B: chosen by every source (what reap_overlap calls "mixed").
    all_top = np.argsort(-unit_norm(per.sum(0)), axis=-1)[:, :N]

    print(f"  {'source':<8}{'own':>8}{'ref-mix':>10}{'all-src':>10}{'ref gap':>10}")
    print("  " + "-" * 44)
    rows = []
    for si, lab in enumerate(labels):
        own = np.mean([norm[si, l, np.argsort(-norm[si, l])[:N]].sum() for l in range(L)])
        ref_r = np.mean([norm[si, l, ref_top[l]].sum() for l in range(L)])
        all_r = np.mean([norm[si, l, all_top[l]].sum() for l in range(L)])
        rows.append((lab, own, ref_r, all_r))
        mark = "  <-- TARGET" if lab == args.target else ("  (in ref mix)" if lab in ref else "")
        print(f"  {lab:<8}{own:>7.1%}{ref_r:>10.1%}{all_r:>10.1%}{100*(own-ref_r):>+9.1f}{mark}")

    tgt = next(r for r in rows if r[0] == args.target)
    _, own, ref_r, _ = tgt
    in_ref = [r for r in rows if r[0] in ref]
    in_ref_gap = np.mean([r[1] - r[2] for r in in_ref])
    tgt_gap = own - ref_r

    print(f"\n  {args.target} retention under a reference-mix plan : {ref_r:.1%}")
    print(f"  {args.target} ceiling (own-distribution plan)      : {own:.1%}")
    print(f"  gap                                        : {100*tgt_gap:+.1f} pts")
    print(f"  mean gap for sources IN the reference mix  : {100*in_ref_gap:+.1f} pts")

    print()
    if tgt_gap <= in_ref_gap + 0.02:
        print(f"  => {args.target.upper()} IS EFFECTIVELY PRESERVED. Its gap is no worse than that of")
        print("     languages actually present in the reference corpus, so its experts")
        print("     survive on shared structure. A targeted build buys little.")
    elif ref_r < N / NE * 1.5:
        print(f"  => {args.target.upper()} COLLAPSES. Retention approaches chance; its experts are")
        print("     largely deleted by a reference-mix plan. A targeted build is required.")
    else:
        print(f"  => {args.target.upper()} IS PARTIALLY DEGRADED. It retains real mass but measurably")
        print("     less than languages in the reference mix. A targeted build helps.")


if __name__ == "__main__":
    main()
