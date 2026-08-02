#!/bin/bash
# Re-score English on exactly the items already scored in Croatian.
#
# The main sweep drew 200 items per language with the same seed and got only 48
# in common, because shuffle(seed) fixes the permutation and each language config
# stores its rows in a different order. That reduced a paired benchmark to an
# unpaired one and threw away most of its statistical power.
#
# This runs English over the Croatian item set. Costs 152 new items per model --
# the 48 already in common replay from the existing checkpoint for free.
#
# Results land in results/paired/ rather than overwriting results/full/, so the
# original unpaired run stays intact and reproducible. Both are real data; only
# one answers the question properly.

set -u
cd "$(dirname "$0")/.." || exit 1

SRC=$K3_SRC
K3=$K3_BUILD
FULL=results/full
OUT=results/paired
LOGS=$OUT/logs
IDS=$OUT/item-ids.txt
mkdir -p "$LOGS" "$OUT/checkpoints"

# --- prepare the target set and seed the checkpoints -----------------------
uv run python - "$FULL" "$OUT" "$IDS" <<'PY' || exit 1
import json, pathlib, sys
full, out, idfile = (pathlib.Path(p) for p in sys.argv[1:4])
MODELS = ["K3-REAP80-hr-code", "gemma4-31b", "gpt-oss-20b", "gpt-oss-120b"]

def load(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

# Every model saw the identical Croatian 200, so the target set is unambiguous.
# Assert it rather than assume it -- if the arms had diverged, pairing against
# one model's set would silently mis-scope the others.
sets = {m: [r["id"] for r in load(full / "checkpoints" / f"{m}-belebele_hrv.jsonl")]
        for m in MODELS}
ref = sets[MODELS[0]]
for m, s in sets.items():
    if set(s) != set(ref):
        raise SystemExit(f"{m} scored a different Croatian item set; refusing to pair")
idfile.write_text("\n".join(ref) + "\n")
print(f"[pair] target set: {len(ref)} items shared by all {len(MODELS)} arms")

want = set(ref)
for m in MODELS:
    # Croatian is already exactly the target set -- copy it across verbatim
    # rather than spending 200 items per model regenerating identical answers.
    src = full / "checkpoints" / f"{m}-belebele_hrv.jsonl"
    (out / "checkpoints" / f"{m}-belebele_hrv.jsonl").write_text(src.read_text())

    # English: carry over whichever of the target items the unpaired run
    # happened to cover, so only the remainder costs generation time.
    keep = [r for r in load(full / "checkpoints" / f"{m}-belebele_eng.jsonl")
            if r["id"] in want]
    dst = out / "checkpoints" / f"{m}-belebele_eng.jsonl"
    if not dst.exists():
        dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))
    print(f"[pair] {m}: {len(keep)}/{len(ref)} english items reusable, "
          f"{len(ref)-len(keep)} to run")
PY

run() {
  local name="$1"; shift
  echo "=== [$(date '+%H:%M:%S')] START $name" | tee -a "$LOGS/pair.log"
  uv run eval/run_eval.py --out-dir "$OUT" --tasks belebele_eng \
      --belebele-ids "$IDS" "$@" >>"$LOGS/$name.log" 2>&1
  local rc=$?
  echo "=== [$(date '+%H:%M:%S')] END   $name (exit $rc)" | tee -a "$LOGS/pair.log"
}

# Free whatever ollama is holding before K3 asks for 326 GB.
for m in gemma4:31b gpt-oss:120b gpt-oss:20b; do ollama stop "$m" >/dev/null 2>&1; done
sleep 5

run k3-eng-paired --backend "mlx:$K3" --src "$SRC" --label K3-REAP80-hr-code

for m in gpt-oss:20b gemma4:31b gpt-oss:120b; do
  for other in gemma4:31b gpt-oss:120b gpt-oss:20b; do
    [ "$other" != "$m" ] && ollama stop "$other" >/dev/null 2>&1
  done
  sleep 3
  run "${m//:/-}-eng-paired" --backend "ollama:$m"
done

echo "=== [$(date '+%H:%M:%S')] PAIRING COMPLETE" | tee -a "$LOGS/pair.log"
