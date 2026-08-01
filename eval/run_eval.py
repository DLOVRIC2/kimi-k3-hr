#!/usr/bin/env python3
"""Run the benchmark suite against one backend and write results.

    # local K3 build (loads 350 GB, keep it resident across all tasks)
    uv run eval/run_eval.py --backend mlx:/path/to/build \\
        --src ~/…/Kimi-K3-src --limit-belebele 200

    # ollama comparators
    uv run eval/run_eval.py --backend ollama:gemma4:31b --limit-belebele 200

One backend per invocation, deliberately: a 350 GB MLX model and an ollama
model cannot both be resident on this machine, and interleaving them would make
timings meaningless.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from datetime import date

import backends as B
import tasks as T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, help="ollama:<model> or mlx:<path>")
    ap.add_argument("--src", default=None, help="Kimi-K3-src (mlx backend only)")
    ap.add_argument("--toolchain", default=str(pathlib.Path.home() / "kimi-k3-mlx"))
    ap.add_argument("--label", default=None)
    ap.add_argument("--tasks", default="belebele_hrv,belebele_eng,humaneval")
    ap.add_argument("--limit-belebele", type=int, default=200)
    ap.add_argument("--limit-humaneval", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    out_dir = pathlib.Path(a.out_dir) if a.out_dir else pathlib.Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    kw = {}
    if a.backend.startswith("mlx:"):
        if not a.src:
            raise SystemExit("--src is required for the mlx backend")
        kw = dict(src=a.src, toolchain=a.toolchain, label=a.label, thinking=a.thinking)
    backend = B.make_backend(a.backend, **kw)
    label = a.label or backend.name

    run = {
        "label": label,
        "backend": a.backend,
        "date": date.today().isoformat(),
        "seed": a.seed,
        "decoding": "greedy (temperature 0)",
        "tasks": {},
    }

    wanted = [t.strip() for t in a.tasks.split(",")]
    t_start = time.time()

    for task in wanted:
        print(f"\n[eval] {label} :: {task}", flush=True)
        t0 = time.time()

        if task.startswith("belebele_"):
            lang = {"hrv": "hrv_Latn", "eng": "eng_Latn"}[task.split("_")[1]]
            ds = T.load_belebele(lang, a.limit_belebele, a.seed)
            res = T.score_belebele(backend, ds)
            print(f"    -> accuracy {res['accuracy']:.1%}  parse {res['parse_rate']:.1%}  n={res['n']}")
        elif task == "humaneval":
            ds = T.load_humaneval(a.limit_humaneval, a.seed)
            res = T.score_humaneval(backend, ds)
            print(f"    -> pass@1 {res['pass@1']:.1%}  n={res['n']}")
        else:
            raise SystemExit(f"unknown task {task!r}")

        res["wall_s"] = round(time.time() - t0, 1)
        run["tasks"][task] = res

    # The headline metric. Parallel items mean this isolates Croatian ability
    # from general capability -- a big model beating a small one on raw hrv
    # score says nothing on its own.
    hrv = run["tasks"].get("belebele_hrv", {}).get("accuracy")
    eng = run["tasks"].get("belebele_eng", {}).get("accuracy")
    if hrv is not None and eng is not None:
        run["croatian_gap"] = round(eng - hrv, 4)
        print(f"\n[eval] English-Croatian gap: {100*(eng-hrv):+.1f} pts "
              f"(eng {eng:.1%} / hrv {hrv:.1%})")

    run["total_wall_s"] = round(time.time() - t_start, 1)
    safe = label.replace(":", "-").replace("/", "-")
    path = out_dir / f"eval-{safe}-{run['date']}.json"
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False))
    print(f"\n[eval] wrote {path}  ({run['total_wall_s']/60:.1f} min)")

    backend.close()


if __name__ == "__main__":
    main()
