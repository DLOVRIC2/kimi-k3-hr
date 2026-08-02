#!/usr/bin/env python3
"""Re-apply the scoring rules to completed runs, without re-running any model.

The echo detector had no minimum length, so it matched any response short enough
to occur incidentally inside the passage -- and the expected response is a single
digit, which most passages contain by chance. A model that answered '2' correctly
was recorded as having echoed the passage, then scored wrong.

Nothing about the generations was affected, only their classification, so this is
recoverable from what is already on disk. That is possible only because the
checkpoint stores `raw` as exactly `text.strip()[:40]` and the echo test looks at
`head[:40]` -- the stored prefix is precisely the input the rule consumes, so the
decision reproduces exactly rather than approximately.

The digit search is the one place to be careful: the original searched the full
response, and only 40 characters were kept. So an item is rescored only when its
stored text is demonstrably complete (shorter than the truncation limit). Longer
responses keep their original outcome and are reported as skipped.

Writes corrected checkpoints to results/rescored/ and leaves the originals alone.

    uv run eval/rescore.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import tasks as T  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCES = {
    "hrv": ROOT / "results" / "full" / "checkpoints",
    "eng": ROOT / "results" / "paired" / "checkpoints",
    # The original, unmatched English run is rescored too. Without it the
    # "what did unpaired sampling cost" comparison would contrast old-rule
    # unpaired numbers against new-rule paired ones, conflating two separate
    # corrections into one apparent effect.
    "eng_unpaired": ROOT / "results" / "full" / "checkpoints",
}
OUT = ROOT / "results" / "rescored" / "checkpoints"
MODELS = ["K3-REAP80-hr-code", "gpt-oss-120b", "gemma4-31b", "gpt-oss-20b"]
RAW_LIMIT = 40  # ItemResult.raw = text.strip()[:40]


def prompts_for(lang: str, ids: list[str]) -> dict[str, tuple[str, int]]:
    """Rebuild each item's exact prompt and correct answer."""
    config = {"hrv": "hrv_Latn", "eng": "eng_Latn",
              "eng_unpaired": "eng_Latn"}[lang]
    ds = T.load_belebele(config, None, 0, ids=ids)
    out = {}
    for row in ds:
        out[T.belebele_id(row)] = (
            T.BELEBELE_PROMPT.format(
                passage=row["flores_passage"], question=row["question"],
                a1=row["mc_answer1"], a2=row["mc_answer2"],
                a3=row["mc_answer3"], a4=row["mc_answer4"]),
            int(row["correct_answer_num"]),
        )
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    totals = {"changed": 0, "skipped": 0, "recovered": 0}

    for lang, src in SOURCES.items():
        for model in MODELS:
            task = "belebele_eng" if lang == "eng_unpaired" else f"belebele_{lang}"
            path = src / f"{model}-{task}.jsonl"
            rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            meta = prompts_for(lang, [r["id"] for r in rows])

            changed = skipped = recovered = 0
            for r in rows:
                prompt, answer = meta[r["id"]]
                raw = r["raw"]
                was_echo, was_correct = r["echoed"], r["correct"]
                now_echo = T._is_echo(raw, prompt)
                if now_echo == was_echo:
                    continue
                # Only rescore when the stored text is provably the whole
                # response; otherwise the digit search would run on a prefix.
                if len(raw) >= RAW_LIMIT:
                    skipped += 1
                    continue
                m = None if now_echo else T._DIGIT.search(raw)
                r["echoed"] = now_echo
                r["parsed"] = m is not None
                r["correct"] = bool(m and int(m.group()) == answer)
                r["rescored"] = True
                changed += 1
                if r["correct"] and not was_correct:
                    recovered += 1

            (OUT / f"{model}-belebele_{lang}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            print(f"{model:22s} {lang}  reclassified {changed:3d}  "
                  f"recovered {recovered:3d} correct answers  skipped {skipped}")
            totals["changed"] += changed
            totals["skipped"] += skipped
            totals["recovered"] += recovered

    print(f"\ntotal: {totals['changed']} reclassified, "
          f"{totals['recovered']} correct answers recovered, "
          f"{totals['skipped']} left alone (response longer than the stored prefix)")


if __name__ == "__main__":
    main()
