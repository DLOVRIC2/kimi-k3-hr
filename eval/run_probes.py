#!/usr/bin/env python3
"""Run the seven-domain probe set against a converted Kimi-K3 build.

Emits two artifacts per run:
  results/<build>-<date>.json   machine-readable: outputs + timings
  results/<build>-<date>.md     hand-scoring sheet: prompt, rubric, output, blank score

Scores are deliberately NOT automated. At eleven probes, a rubric applied
consistently by hand beats a metric you do not trust -- and the thing being
measured (does Croatian degrade while Python does not) is a judgement about
fluency and correctness that perplexity will not capture.

TWO THINGS THAT WILL SILENTLY RUIN YOUR NUMBERS
-----------------------------------------------
1. mlxmem.wire() MUST run before load_model(). An unwired multi-hundred-GB model
   faults its weights from SSD on every decode step: 0.20 tok/s instead of 5.42,
   a 27x understatement. Measured on this toolchain's own published artifact.

2. Prefill and decode must be reported separately. Prefill is compute-bound and
   touches each page once, so it looks healthy (~73 tok/s) even when decode is
   50x below the bandwidth ceiling. A single blended tok/s hides exactly the
   failure you most need to see.

Tokenization uses the bundled tiktoken BPE rather than transformers'
AutoTokenizer -- the shipped tokenization_kimi.py targets transformers 4.56 and
does not import under 5.x.

    uv run eval/run_probes.py \\
        --model ~/…/models/K3-REAP80-hr-code \\
        --toolchain $K3_MLX \\
        --src ~/…/models/Kimi-K3-src
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from datetime import date

import yaml

# Kimi-K3 emits these to end a turn; smoke.py treats both as stop.
STOP_TOKENS = {163584, 163586}


def load_probes(path: pathlib.Path) -> tuple[dict, list[dict]]:
    spec = yaml.safe_load(path.read_text())
    return spec.get("meta", {}), spec["domains"]


def build_prompt(probe: dict, probes_dir: pathlib.Path) -> str | None:
    """Inline prompt, or a fixture file plus suffix for the long-context probe."""
    if "prompt" in probe:
        return probe["prompt"]

    fixture = probes_dir / probe["prompt_file"]
    if not fixture.exists():
        return None
    return fixture.read_text() + probe.get("prompt_suffix", "")


def generate(model, enc, prompt: str, max_tokens: int, mx) -> dict:
    """Greedy decode, timing prefill and decode separately.

    Hand-rolled rather than mlx_lm.generate() because the tiktoken encoder does
    not satisfy TokenizerWrapper's interface. That is safe ONLY because wire()
    has already been called -- see the module docstring.
    """
    ids = enc.encode_ordinary(prompt)
    x = mx.array([ids])
    cache = model.make_cache()

    t0 = time.time()
    logits = model(x, cache=cache)
    mx.eval(logits)
    prefill_s = time.time() - t0

    out: list[int] = []
    tok = int(mx.argmax(logits[0, -1]))
    t0 = time.time()
    ttft = None
    for i in range(max_tokens):
        if tok in STOP_TOKENS:
            break
        out.append(tok)
        if i == 0:
            ttft = time.time() - t0
        logits = model(mx.array([[tok]]), cache=cache)
        mx.eval(logits)
        tok = int(mx.argmax(logits[0, -1]))
    decode_s = time.time() - t0

    return {
        "text": enc.decode(out),
        "prompt_tokens": len(ids),
        "gen_tokens": len(out),
        "prefill_s": round(prefill_s, 2),
        "prefill_tok_s": round(len(ids) / max(prefill_s, 1e-9), 1),
        "decode_s": round(decode_s, 2),
        "decode_tok_s": round(len(out) / max(decode_s, 1e-9), 2),
        "ttft_s": round(ttft, 2) if ttft else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run probe set against a K3 build")
    ap.add_argument("--model", required=True, help="converted build directory")
    ap.add_argument("--toolchain", default=str(pathlib.Path(
        os.environ.get("K3_MLX", pathlib.Path.home() / "kimi-k3-mlx"))),
        help="checkout of PipeNetwork/kimi-k3-mlx; overrides $K3_MLX")
    ap.add_argument("--src", default=None, help="Kimi-K3-src, for tiktoken.model")
    ap.add_argument("--probes", default=None, help="defaults to eval/probes.yaml beside this script")
    ap.add_argument("--out-dir", default=None, help="defaults to ../results")
    ap.add_argument("--label", default=None, help="build label for filenames")
    ap.add_argument("--domains", default=None, help="comma-separated subset, e.g. croatian,python")
    args = ap.parse_args()

    here = pathlib.Path(__file__).resolve().parent
    probes_path = pathlib.Path(args.probes) if args.probes else here / "probes.yaml"
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else here.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = pathlib.Path(args.model).expanduser()
    label = args.label or model_path.name

    # The toolchain owns build_tokenizer and mlxmem; import from it rather than
    # vendoring a second copy that can drift.
    toolchain = pathlib.Path(args.toolchain).expanduser()
    sys.path.insert(0, str(toolchain / "scripts"))
    sys.path.insert(0, str(toolchain))

    import mlx.core as mx
    import mlxmem
    from mlx_lm.utils import load_model
    from reap_calibrate import build_tokenizer

    meta, domains = load_probes(probes_path)
    if args.domains:
        want = {d.strip() for d in args.domains.split(",")}
        domains = [d for d in domains if d["id"] in want]

    max_tokens = int(meta.get("max_tokens", 512))
    seed = int(meta.get("seed", 0))
    mx.random.seed(seed)

    # MUST precede load_model. See module docstring -- 27x.
    wired = mlxmem.wire()
    print(f"[probe] wired limit {wired / 1e9:.0f} GB", flush=True)

    tok_src = model_path if (model_path / "tiktoken.model").exists() else pathlib.Path(args.src or model_path)
    enc = build_tokenizer(str(tok_src))
    print(f"[probe] tokenizer ok (n_vocab {enc.n_vocab})", flush=True)

    t0 = time.time()
    print(f"[probe] loading {model_path} ...", flush=True)
    model, _cfg = load_model(model_path, lazy=False)
    mx.eval(model.parameters())
    load_s = time.time() - t0
    peak = mx.get_peak_memory() / 1e9 if hasattr(mx, "get_peak_memory") else float("nan")
    print(f"[probe] loaded in {load_s:.0f}s | peak {peak:.0f} GB\n", flush=True)

    run = {
        "label": label,
        "model_path": str(model_path),
        "date": date.today().isoformat(),
        "seed": seed,
        "max_tokens": max_tokens,
        "load_s": round(load_s, 1),
        "peak_gb": round(peak, 1),
        "wired_gb": round(wired / 1e9, 1),
        "domains": {},
    }

    for dom in domains:
        did = dom["id"]
        run["domains"][did] = {"role": dom.get("role", "probe"), "probes": {}}
        for probe in dom["probes"]:
            pid = probe["id"]
            prompt = build_prompt(probe, here)
            if prompt is None:
                print(f"[probe] {did}/{pid}: SKIPPED (missing fixture {probe.get('prompt_file')})")
                run["domains"][did]["probes"][pid] = {"skipped": "missing fixture"}
                continue

            print(f"[probe] {did}/{pid} ...", end="", flush=True)
            result = generate(model, enc, prompt, max_tokens, mx)
            result["rubric"] = probe.get("rubric", "")
            result["prompt"] = prompt if len(prompt) < 4000 else prompt[:2000] + "\n…[truncated]…\n"
            run["domains"][did]["probes"][pid] = result
            print(f" {result['gen_tokens']} tok @ {result['decode_tok_s']} tok/s "
                  f"(prefill {result['prefill_tok_s']} tok/s)", flush=True)

    stamp = date.today().isoformat()
    json_path = out_dir / f"{label}-{stamp}.json"
    json_path.write_text(json.dumps(run, indent=2, ensure_ascii=False))

    # Hand-scoring sheet. Rubric sits directly above the output so you are not
    # scrolling between two files while judging Croatian grammar.
    md = [
        f"# Probe run — {label}",
        "",
        f"- date `{stamp}` · seed `{seed}` · max_tokens `{max_tokens}`",
        f"- load `{run['load_s']}s` · peak `{run['peak_gb']} GB` · wired `{run['wired_gb']} GB`",
        "",
        "Score each probe 0–5 against its rubric. **Do not average across domains** —",
        "the spread between them is the result.",
        "",
        "---",
        "",
    ]
    for did, dom in run["domains"].items():
        md += [f"## {did}  ·  _{dom['role']}_", ""]
        for pid, r in dom["probes"].items():
            md += [f"### {pid}", ""]
            if "skipped" in r:
                md += [f"> SKIPPED — {r['skipped']}", ""]
                continue
            md += [
                f"`{r['gen_tokens']} tok @ {r['decode_tok_s']} tok/s` · "
                f"`prefill {r['prompt_tokens']} tok @ {r['prefill_tok_s']} tok/s`",
                "",
                "**Rubric**", "", "```", r["rubric"].strip(), "```", "",
                "**Output**", "", "```", r["text"].strip(), "```", "",
                "**SCORE: _ / 5**", "", "---", "",
            ]
    (out_dir / f"{label}-{stamp}.md").write_text("\n".join(md))

    print(f"\n[probe] wrote {json_path}")
    print(f"[probe] wrote {out_dir / f'{label}-{stamp}.md'}  <- score this by hand")


if __name__ == "__main__":
    main()
