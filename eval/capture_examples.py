#!/usr/bin/env python3
"""Re-run a hand-picked set of items and keep the FULL model output.

The benchmark checkpoints deliberately store almost no text: Belebele truncates
the response to 40 characters and HumanEval records none at all. That is right
for a 2,276-item run -- the scorer only needs a digit or a pass/fail -- but it
means the most interesting result in the project has no artifact behind it. We
can say that 83 of 164 coding prompts produced zero tokens; we cannot show
anyone what that looked like, or what the 81 that did answer produced.

So this re-runs a small stratified sample with everything retained: the response
channel, the thinking channel, token counts, and the extracted code. Items are
chosen from the completed run by outcome, so the sample covers each failure mode
rather than whatever the first N happen to be.

A comparator answers the SAME prompts, because "K3 emitted nothing here" is a
much weaker statement than "K3 emitted nothing here and a 19 GB model solved it."

    uv run eval/capture_examples.py                  # K3 + gemma4:31b
    uv run eval/capture_examples.py --skip-k3        # comparator only (fast)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import backends as B  # noqa: E402
import tasks as T  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CKPT = ROOT / "results" / "full" / "checkpoints"
K3_LABEL = "K3-REAP80-hr-code"

SRC = pathlib.Path.home() / "models/Kimi-K3-src"
K3_PATH = pathlib.Path.home() / "models/K3-REAP80-hr-code-q8"

# How many of each outcome to capture. Small on purpose: at 5.4 tok/s a coding
# item costs about a minute, and the point is illustration, not measurement.
QUOTA = {
    "humaneval_silent": 3,   # zero tokens generated -- the headline failure
    "humaneval_passed": 2,   # produced working code, to show capability survives
    "hrv_echoed": 3,         # restated the passage instead of answering
    "hrv_correct": 2,
    "hrv_wrong": 2,
    "eng_echoed": 2,
}


def load(name: str) -> list[dict]:
    p = CKPT / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def pick(rows: list[dict], predicate, n: int, seed: int = 20260802) -> list[dict]:
    """Deterministic sample of items matching an outcome."""
    matching = [r for r in rows if predicate(r)]
    rng = random.Random(seed)
    rng.shuffle(matching)
    return matching[:n]


def select() -> list[dict]:
    """Choose items by their outcome in the completed run."""
    he = load(f"{K3_LABEL}-humaneval")
    hrv = load(f"{K3_LABEL}-belebele_hrv")
    eng = load(f"{K3_LABEL}-belebele_eng")

    chosen = []
    for r in pick(he, lambda r: r["gen_tokens"] == 0, QUOTA["humaneval_silent"]):
        chosen.append({"task": "humaneval", "id": r["id"], "category": "silent"})
    for r in pick(he, lambda r: r["correct"], QUOTA["humaneval_passed"]):
        chosen.append({"task": "humaneval", "id": r["id"], "category": "passed"})
    for r in pick(hrv, lambda r: r["echoed"], QUOTA["hrv_echoed"]):
        chosen.append({"task": "belebele_hrv", "id": r["id"], "category": "echoed"})
    for r in pick(hrv, lambda r: r["correct"], QUOTA["hrv_correct"]):
        chosen.append({"task": "belebele_hrv", "id": r["id"], "category": "correct"})
    for r in pick(hrv, lambda r: r["parsed"] and not r["correct"], QUOTA["hrv_wrong"]):
        chosen.append({"task": "belebele_hrv", "id": r["id"], "category": "wrong"})
    for r in pick(eng, lambda r: r["echoed"], QUOTA["eng_echoed"]):
        chosen.append({"task": "belebele_eng", "id": r["id"], "category": "echoed"})
    return chosen


def build_prompts(chosen: list[dict]) -> list[dict]:
    """Attach the exact prompt text each selected item was scored with."""
    by_task: dict[str, list[dict]] = {}
    for c in chosen:
        by_task.setdefault(c["task"], []).append(c)

    out = []
    for task, group in by_task.items():
        ids = [c["id"] for c in group]
        if task == "humaneval":
            ds = T.load_humaneval(None, 0)
            rows = {r["task_id"]: r for r in ds if r["task_id"] in set(ids)}
            for c in group:
                row = rows[c["id"]]
                out.append({**c,
                            "prompt": T.HUMANEVAL_PROMPT.format(prompt=row["prompt"]),
                            "signature": row["prompt"],
                            "entry_point": row["entry_point"],
                            "test": row["test"]})
        else:
            lang = {"hrv": "hrv_Latn", "eng": "eng_Latn"}[task.split("_")[1]]
            ds = T.load_belebele(lang, None, 0, ids=ids)
            rows = {T.belebele_id(r): r for r in ds}
            for c in group:
                row = rows[c["id"]]
                out.append({**c,
                            "prompt": T.BELEBELE_PROMPT.format(
                                passage=row["flores_passage"], question=row["question"],
                                a1=row["mc_answer1"], a2=row["mc_answer2"],
                                a3=row["mc_answer3"], a4=row["mc_answer4"]),
                            "correct_answer": int(row["correct_answer_num"])})
    return out


def run_model(backend, items: list[dict], model_key: str) -> None:
    for i, it in enumerate(items, 1):
        budget = (backend.code_budget if it["task"] == "humaneval"
                  else backend.answer_budget)
        print(f"  [{i}/{len(items)}] {model_key} {it['task']} {it['category']} "
              f"{it['id'][:50]}", flush=True)
        g = backend.generate(it["prompt"], max_tokens=budget)
        rec = {
            "text": g.text,
            "gen_tokens": g.gen_tokens,
            "prompt_tokens": g.prompt_tokens,
            "thinking_chars": g.thinking_chars,
            "truncated": g.truncated,
            "decode_tok_s": round(g.decode_tok_s, 2),
        }
        if it["task"] == "humaneval":
            code = T.extract_code(g.text, it["signature"], it["entry_point"])
            ok, reason = T._run_one(code, it["test"], it["entry_point"])
            rec["extracted_code"] = code
            rec["passed"] = ok
            rec["reason"] = reason
        it.setdefault("outputs", {})[model_key] = rec


def to_markdown(items: list[dict]) -> str:
    """Readable transcript, for quoting directly in a writeup."""
    out = ["# Captured model outputs", "",
           "Full generations for a stratified sample of items from the completed run.",
           "Selected by outcome, then re-run with the same prompt and token budget.", ""]
    order = ["silent", "echoed", "wrong", "correct", "passed"]
    for cat in order:
        group = [i for i in items if i["category"] == cat]
        if not group:
            continue
        out += [f"## {cat}", ""]
        for it in group:
            out += [f"### `{it['id']}` ({it['task']})", ""]
            for key, rec in it.get("outputs", {}).items():
                body = rec["text"].strip()
                shown = body if body else "(no output -- first sampled token was a stop token)"
                out += [f"**{key}** — {rec['gen_tokens']} tokens generated"
                        + (f", passed={rec['passed']}" if "passed" in rec else ""),
                        "", "```", shown[:2000], "```", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-k3", action="store_true")
    ap.add_argument("--comparator", default="gemma4:31b")
    ap.add_argument("--out", default=str(ROOT / "results" / "examples"))
    a = ap.parse_args()

    outdir = pathlib.Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)

    items = build_prompts(select())
    print(f"[capture] {len(items)} items selected")

    if not a.skip_k3:
        k3 = B.MLXBackend(str(K3_PATH), src=str(SRC),
                          toolchain=str(pathlib.Path.home() / "kimi-k3-mlx"),
                          label=K3_LABEL)
        run_model(k3, items, K3_LABEL)

    if a.comparator:
        run_model(B.OllamaBackend(a.comparator), items, a.comparator)

    (outdir / "examples.json").write_text(json.dumps(items, indent=2, ensure_ascii=False))
    (outdir / "examples.md").write_text(to_markdown(items))
    print(f"[capture] wrote {outdir}/examples.json and examples.md")


if __name__ == "__main__":
    main()
