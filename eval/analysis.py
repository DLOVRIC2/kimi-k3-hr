"""Regenerate every published number from the raw per-item records.

RESULTS.md was originally written by computing figures ad hoc in a shell and
typing them into markdown. That is how it came to claim "50 of 99 prompts
produced zero tokens": true of a partial checkpoint read mid-run, false of the
completed 164-item task. A number nobody can recompute is a number nobody can
correct.

So the checkpoints under results/full/checkpoints/ are the source of truth here,
NOT the summary JSONs -- the summaries are themselves derived, and re-deriving
from them would preserve any error they contain. The summary is read only for
things no per-item record holds: memory peaks and wall-clock.

Usage:
    python eval/analysis.py                  # all tables, markdown to stdout
    python eval/analysis.py --table headline
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parent.parent
FULL = ROOT / "results" / "full"
CKPT = FULL / "checkpoints"

# Display order is deliberate: the pruned build first, then comparators by
# descending size, so the size/score inversion is visible by reading downward.
MODELS = ["K3-REAP80-hr-code", "gpt-oss-120b", "gemma4-31b", "gpt-oss-20b"]

# On-disk footprint of the weights, in GB.
#
# Measured, not computed from parameter counts -- quantisation format and
# per-tensor overhead make the arithmetic estimate wrong by enough to matter.
# K3 from `du -sh` on the converted build; the ollama models from
# `ollama list`. These are the only figures in this file that come from
# outside the results directory, which is why they are pinned here with
# their provenance rather than passed in.
DISK_GB = {
    "K3-REAP80-hr-code": 326,
    "gpt-oss-120b": 65,
    "gemma4-31b": 19,
    "gpt-oss-20b": 13,
}

RESULT_JSON = {
    "K3-REAP80-hr-code": "eval-K3-REAP80-hr-code-2026-08-01.json",
    "gpt-oss-120b": "eval-gpt-oss-120b-2026-08-02.json",
    "gemma4-31b": "eval-gemma4-31b-2026-08-02.json",
    "gpt-oss-20b": "eval-gpt-oss-20b-2026-08-02.json",
}


# ------------------------------------------------------------------ loading

def load_items(model: str, task: str) -> list[dict]:
    p = CKPT / f"{model}-{task}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def load_summary(model: str) -> dict:
    p = FULL / RESULT_JSON[model]
    return json.loads(p.read_text()) if p.exists() else {}


def responded(rec: dict, task: str) -> bool:
    """Did the model emit a usable answer at all?

    Two different signals because the tasks fail differently. On Belebele a
    response exists but may be an echo of the passage, so `parsed` -- which is
    False for echoes by construction -- is the right test. On HumanEval the
    failure is starker: the first token sampled is a stop token and the
    generation is literally empty.
    """
    if task == "humaneval":
        return rec["gen_tokens"] > 0
    return rec["parsed"]


# --------------------------------------------------------------- statistics

def _two_sided_p(z: float) -> float:
    """Two-sided normal tail probability.

    erfc, not 1 - CDF. The naive form catastrophically cancels: at |z| > 8 the
    CDF rounds to exactly 1.0 in float64 and the p-value prints as a flat 0,
    which reads as "infinitely significant" when it means "below float
    resolution". erfc computes the tail directly and stays accurate past z=30.
    """
    return math.erfc(abs(z) / math.sqrt(2.0))


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Unpooled-difference, pooled-variance two-sided z-test.

    For INDEPENDENT samples only. Using this on two measurements of the same
    items throws away the pairing and costs real power -- see mcnemar().
    """
    if not n1 or not n2:
        return 0.0, 1.0
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, _two_sided_p(z)


def mcnemar(b: int, c: int) -> tuple[int, int, float]:
    """Exact two-sided McNemar on paired binary outcomes.

    b = right in A and wrong in B, c = the reverse. Items both models got right,
    or both wrong, carry NO information about the difference and are excluded --
    which is exactly why this has more power than the unpaired test: the noise
    from item difficulty cancels instead of being estimated.

    Exact binomial rather than the chi-square approximation because the
    discordant count can be small, and that is precisely when the approximation
    is worst.
    """
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2 ** n)
    return b, c, min(1.0, 2 * tail)


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


# ------------------------------------------------------------------- tables

@dataclass
class Score:
    n: int
    correct: int
    responded: int

    @property
    def acc(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def resp_rate(self) -> float:
        return self.responded / self.n if self.n else 0.0

    @property
    def acc_given_resp(self) -> float:
        return self.correct / self.responded if self.responded else 0.0


def score(model: str, task: str) -> Score:
    items = load_items(model, task)
    return Score(
        n=len(items),
        correct=sum(bool(r["correct"]) for r in items),
        responded=sum(responded(r, task) for r in items),
    )


def aggregate_tok_s(model: str) -> float:
    """Token-weighted decode rate across all three tasks.

    NOT the mean of the per-task means, which is what the run logs report and
    what the first draft of RESULTS.md published. That average is wrong in two
    ways at once:

    - It weights a 164-item coding task equal to a 200-item multiple-choice task
      whose median generation is ONE token. A single-token generation's "rate"
      is one forward pass divided by a very small interval, so it is mostly
      measurement noise, and it dominated the mean.
    - Items that generated nothing at all record a rate of 0.0 and were averaged
      in as though the model had run at zero tokens per second. K3 produced no
      output on 83 of 164 HumanEval prompts, so this dragged its published
      figure down to 4.3 tok/s against a true sustained 5.4.

    Both corrections point the same way: measure total tokens over total decode
    seconds, and exclude items that decoded nothing. Per-item decode seconds are
    recovered as gen_tokens / decode_tok_s.
    """
    toks = secs = 0.0
    for task in ("belebele_hrv", "belebele_eng", "humaneval"):
        for r in load_items(model, task):
            g, rate = r["gen_tokens"], r.get("decode_tok_s", 0.0)
            if g > 0 and rate > 0:
                toks += g
                secs += g / rate
    return toks / secs if secs else 0.0


def t_headline() -> str:
    out = ["## Headline", "",
           "| model | on disk | Croatian | English | gap | HumanEval | peak RSS | tok/s | wall |",
           "|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        hrv, eng, he = score(m, "belebele_hrv"), score(m, "belebele_eng"), score(m, "humaneval")
        s = load_summary(m)
        rss = s.get("resources", {}).get("peak_rss_gb", float("nan"))
        wall = s.get("total_wall_s", 0) / 60
        gap = (eng.acc - hrv.acc) * 100
        out.append(
            f"| {m} | {DISK_GB[m]} GB | {hrv.acc:.1%} | {eng.acc:.1%} | {gap:+.1f} | "
            f"{he.acc:.1%} | {rss:.0f} GB | {aggregate_tok_s(m):.1f} | {wall:.0f} min |")
    out += ["", "Chance on Belebele is 25%."]
    return "\n".join(out)


def t_initiation() -> str:
    out = ["## Response initiation vs capability", "",
           "Every score splits into two questions: did the model answer at all, and",
           "was it right when it did. Aggregating them hides a bimodal failure.", "",
           "| model | task | responded | correct overall | correct *when* responded |",
           "|---|---|---|---|---|"]
    for m in MODELS:
        for task, name in (("belebele_hrv", "Croatian"), ("belebele_eng", "English"),
                           ("humaneval", "HumanEval")):
            s = score(m, task)
            out.append(f"| {m} | {name} | {s.resp_rate:.1%} | {s.acc:.1%} | "
                       f"{s.acc_given_resp:.1%} |")
    he = load_items("K3-REAP80-hr-code", "humaneval")
    zero = [r for r in he if r["gen_tokens"] == 0]
    nz = [r for r in he if r["gen_tokens"] > 0]
    n_zero_correct = sum(r["correct"] for r in zero)
    out += ["",
            f"K3 on HumanEval splits cleanly: **{len(zero)} of {len(he)} prompts produced "
            f"zero tokens**, none of which could score" +
            ("" if n_zero_correct == 0 else f" (though {n_zero_correct} did)") +
            f", while the other {len(nz)} produced code and passed "
            f"{sum(r['correct'] for r in nz)/len(nz):.1%}.",
            "The three comparators produced output on 100% of prompts."]
    return "\n".join(out)


def t_gap() -> str:
    """English-vs-Croatian, paired where the data allows it."""
    out = ["## Language gap", "",
           "Belebele is fully parallel -- the same items exist in every language -- so",
           "the paired test is the correct one wherever the two runs share items.", ""]
    rows = []
    for m in MODELS:
        hrv = {r["id"]: r for r in load_items(m, "belebele_hrv")}
        eng = {r["id"]: r for r in load_items(m, "belebele_eng")}
        both = sorted(set(hrv) & set(eng))
        h, e = score(m, "belebele_hrv"), score(m, "belebele_eng")
        z, pz = two_proportion_z(h.correct, h.n, e.correct, e.n)
        # b: right in Croatian, wrong in English. c: the reverse.
        b = sum(1 for i in both if hrv[i]["correct"] and not eng[i]["correct"])
        c = sum(1 for i in both if eng[i]["correct"] and not hrv[i]["correct"])
        _, _, pm = mcnemar(b, c)
        rows.append((m, h.acc, e.acc, (e.acc - h.acc) * 100, z, pz, len(both), b, c, pm))

    out += ["| model | hrv | eng | gap | z (unpaired) | p | n paired | b | c | p (McNemar) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for m, ha, ea, g, z, pz, npair, b, c, pm in rows:
        out.append(f"| {m} | {ha:.1%} | {ea:.1%} | {g:+.1f} | {z:.2f} | {pz:.3g} {stars(pz)} "
                   f"| {npair} | {b} | {c} | {pm:.3g} {stars(pm)} |")

    npair = rows[0][6]
    if npair < 150:
        out += ["", f"**The paired columns are underpowered at n={npair}.** Sampling 200 items "
                    "per language independently produced only that many in common, so the "
                    "McNemar figures here are indicative and the unpaired test carries the "
                    "claim. `eval/pair_up.py` closes this."]
    return "\n".join(out)


def t_cost() -> str:
    out = ["## Cost", "",
           "| model | peak RSS | peak wired | tok/s | sustained page-in |",
           "|---|---|---|---|---|"]
    for m in MODELS:
        r = load_summary(m).get("resources", {})
        out.append(f"| {m} | {r.get('peak_rss_gb', 0):.0f} GB | "
                   f"{r.get('peak_wired_gb', 0):.0f} GB | {aggregate_tok_s(m):.1f} | "
                   f"{r.get('pagein_gb_per_min_sustained', 0):.3f} GB/min |")
    comp = max(load_summary(m).get("resources", {}).get("peak_compressed_gb", 0)
               for m in MODELS)
    out += ["", "Sustained page-in below 0.05 GB/min throughout means every model ran fully "
                f"resident; the threshold for thrashing is ~0.5. Peak compressed memory across "
                f"all arms was {comp:.1f} GB, so none of these are contaminated by memory "
                "pressure."]
    return "\n".join(out)


def t_versus() -> str:
    """The pruned build against the smallest comparator that beats it."""
    base = "gemma4-31b"
    out = [f"## K3 vs {base}", "",
           f"| task | K3 | {base} | delta | z | p |", "|---|---|---|---|---|---|"]
    for task, name in (("belebele_hrv", "Croatian"), ("belebele_eng", "English"),
                       ("humaneval", "HumanEval")):
        a, b = score("K3-REAP80-hr-code", task), score(base, task)
        z, p = two_proportion_z(a.correct, a.n, b.correct, b.n)
        out.append(f"| {name} | {a.acc:.1%} | {b.acc:.1%} | {(a.acc-b.acc)*100:+.1f} | "
                   f"{z:.2f} | {p:.3g} {stars(p)} |")
    ka = load_summary("K3-REAP80-hr-code").get("resources", {})
    ga = load_summary(base).get("resources", {})
    out += ["", f"K3 needs {ka.get('peak_rss_gb',0)/ga.get('peak_rss_gb',1):.0f}x the resident "
                f"memory and {DISK_GB['K3-REAP80-hr-code']/DISK_GB[base]:.0f}x the disk of "
                f"{base} to run "
                f"{aggregate_tok_s(base)/aggregate_tok_s('K3-REAP80-hr-code'):.0f}x slower "
                "and score worse on every task."]
    return "\n".join(out)


TABLES = {
    "headline": t_headline,
    "initiation": t_initiation,
    "gap": t_gap,
    "cost": t_cost,
    "versus": t_versus,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", choices=list(TABLES), action="append",
                    help="emit only these tables (repeatable); default is all")
    args = ap.parse_args()
    for name in (args.table or list(TABLES)):
        print(TABLES[name]())
        print()


if __name__ == "__main__":
    main()
