#!/usr/bin/env python3
"""Build a tagged, interleaved calibration corpus from a recipe.

Two non-obvious constraints drive this script, both from the kimi-k3-mlx docs:

1. Calibration reads only the FIRST `seqs x seqlen` tokens of the corpus file.
   Concatenating sources means the tail domains are never seen at all. Everything
   must be interleaved round-robin, weighted by target ratio.

2. C4's pooled multilingual config measured 97% Latin script over 200 documents.
   Per-language NAMED configs are mandatory; a pooled multilingual set will
   silently give you almost no Croatian while looking correct.

Output is written twice:
  <out>.txt        plain corpus  -> reap_calibrate.py --calib-text
  <out>.tags.jsonl one line per chunk with its domain tag -> reap_subset.py

The tags are what let you calibrate ONCE and then re-target different domain
mixes without re-running calibration.

    uv run corpus/build_corpus.py recipes/hr-code.yaml --out build/hr-code
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import yaml


@dataclass
class Source:
    tag: str
    ratio: float
    kind: str  # "hf" | "local"
    dataset: str | None = None
    config: str | None = None
    split: str = "train"
    text_field: str = "text"
    path: str | None = None
    glob: str = "**/*.py"
    min_chars: int = 200
    chunks: list[str] = field(default_factory=list)


def load_recipe(path: pathlib.Path) -> tuple[str, list[Source], dict]:
    spec = yaml.safe_load(path.read_text())
    sources = [Source(**s) for s in spec["sources"]]
    total = sum(s.ratio for s in sources)
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"ratios must sum to 1.0, got {total:.4f}")
    return spec.get("name", path.stem), sources, spec.get("options", {})


def pull_hf(src: Source, want: int, chunk_chars: int) -> None:
    """Stream a HuggingFace dataset until we have `want` chunks."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("pip/uv add datasets — required for hf sources")

    print(f"  [{src.tag}] streaming {src.dataset}" + (f":{src.config}" if src.config else ""))
    ds = load_dataset(src.dataset, src.config, split=src.split, streaming=True)

    buf = ""
    for row in ds:
        text = (row.get(src.text_field) or "").strip()
        if len(text) < src.min_chars:
            continue
        buf += text + "\n\n"
        while len(buf) >= chunk_chars:
            src.chunks.append(buf[:chunk_chars])
            buf = buf[chunk_chars:]
            if len(src.chunks) >= want:
                return


def pull_local(src: Source, want: int, chunk_chars: int) -> None:
    """Read real files off disk — your own code is better calibration than scraped code."""
    root = pathlib.Path(src.path).expanduser()
    if not root.exists():
        raise SystemExit(f"local source path does not exist: {root}")

    print(f"  [{src.tag}] reading {root}/{src.glob}")
    buf = ""
    for f in sorted(root.glob(src.glob)):
        if not f.is_file():
            continue
        try:
            text = f.read_text(errors="ignore").strip()
        except Exception:
            continue
        if len(text) < src.min_chars:
            continue
        buf += text + "\n\n"
        while len(buf) >= chunk_chars:
            src.chunks.append(buf[:chunk_chars])
            buf = buf[chunk_chars:]
            if len(src.chunks) >= want:
                return


def interleave(sources: list[Source], total_chunks: int) -> list[tuple[str, str]]:
    """Round-robin weighted by ratio. Order matters — calibration reads the head."""
    quota = {s.tag: int(round(s.ratio * total_chunks)) for s in sources}
    cursor = {s.tag: 0 for s in sources}
    by_tag = {s.tag: s for s in sources}
    emitted: list[tuple[str, str]] = []

    # Deterministic weighted round-robin: each pass takes proportionally from each
    # source, so any prefix of the output approximates the target composition.
    while len(emitted) < total_chunks:
        progressed = False
        for s in sources:
            take = max(1, round(s.ratio * 10))
            for _ in range(take):
                if cursor[s.tag] >= min(quota[s.tag], len(s.chunks)):
                    break
                emitted.append((s.tag, by_tag[s.tag].chunks[cursor[s.tag]]))
                cursor[s.tag] += 1
                progressed = True
                if len(emitted) >= total_chunks:
                    return emitted
        if not progressed:
            break
    return emitted


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a tagged calibration corpus")
    ap.add_argument("recipe", type=pathlib.Path)
    ap.add_argument("--out", required=True, help="output prefix, e.g. build/hr-code")
    ap.add_argument("--seqs", type=int, default=128, help="calibration sequences")
    ap.add_argument("--seqlen", type=int, default=2048, help="tokens per sequence")
    ap.add_argument("--chars-per-token", type=float, default=3.6,
                    help="rough chars/token; 3.6 is conservative for mixed script")
    args = ap.parse_args()

    name, sources, _opts = load_recipe(args.recipe)

    # Overshoot by 30% so interleaving never runs a source dry at the head.
    target_chars = int(args.seqs * args.seqlen * args.chars_per_token * 1.3)
    chunk_chars = int(args.seqlen * args.chars_per_token)
    total_chunks = max(1, target_chars // chunk_chars)

    print(f"recipe        : {name}")
    print(f"budget        : {args.seqs} x {args.seqlen} tokens (~{target_chars:,} chars)")
    print(f"chunks        : {total_chunks} of {chunk_chars:,} chars\n")

    for s in sources:
        want = max(1, int(s.ratio * total_chunks) + 4)
        if s.kind == "hf":
            pull_hf(s, want, chunk_chars)
        elif s.kind == "local":
            pull_local(s, want, chunk_chars)
        else:
            raise SystemExit(f"unknown source kind: {s.kind}")
        print(f"  [{s.tag}] collected {len(s.chunks)} chunks (wanted {want})")

    emitted = interleave(sources, total_chunks)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(f"{out}.txt", "w") as ftxt, open(f"{out}.tags.jsonl", "w") as ftags:
        for i, (tag, chunk) in enumerate(emitted):
            ftxt.write(chunk + "\n")
            ftags.write(json.dumps({"i": i, "tag": tag, "chars": len(chunk)}) + "\n")

    # Report ACHIEVED composition — not the requested one. They differ whenever a
    # source ran dry, and silently shipping the difference is how you get a corpus
    # you think is 20% Croatian but isn't.
    got: dict[str, int] = defaultdict(int)
    for tag, chunk in emitted:
        got[tag] += len(chunk)
    total = sum(got.values()) or 1

    print("\nachieved composition")
    print(f"  {'tag':<14}{'target':>9}{'actual':>9}")
    for s in sources:
        print(f"  {s.tag:<14}{s.ratio*100:>8.1f}%{got[s.tag]/total*100:>8.1f}%")

    head = emitted[: total_chunks // 4]
    head_tags: dict[str, int] = defaultdict(int)
    for tag, chunk in head:
        head_tags[tag] += len(chunk)
    head_total = sum(head_tags.values()) or 1
    print("\nfirst-quarter composition (what a short calibration run actually sees)")
    for s in sources:
        print(f"  {s.tag:<14}{'':>9}{head_tags[s.tag]/head_total*100:>8.1f}%")

    print(f"\nwrote {out}.txt  ({total:,} chars)")
    print(f"wrote {out}.tags.jsonl  ({len(emitted)} chunks)")


if __name__ == "__main__":
    main()
