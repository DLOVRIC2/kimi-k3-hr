#!/bin/bash
# Full benchmark sweep, serial, unattended.
#
# Ordering is deliberate:
#   1. K3 thinking probe first -- it is the cheapest thing that could invalidate
#      everything after it, so it runs before 4h is spent.
#   2. K3 full next, while the machine is verified clean. It needs 326 GB and is
#      the only arm that can fail for memory reasons.
#   3. ollama arms last, cheapest first to slowest, each preceded by `ollama stop`
#      on the others so peak-RSS is attributed to the right model.
#
# Every arm checkpoints per item, so re-running this script resumes rather than
# repeats. Killing it at any point costs at most one item.

set -u
cd "$(dirname "$0")/.." || exit 1

# Paths come from the environment so this runs on any machine. Set them in
# your shell or a .env; the defaults assume everything sits under ~/models.
SRC=${K3_SRC:-$HOME/models/Kimi-K3-src}
K3=${K3_BUILD:-$HOME/models/K3-REAP80-hr-code-q8}
for d in "$SRC" "$K3"; do
  [ -d "$d" ] || { echo "missing: $d (set K3_SRC / K3_BUILD)" >&2; exit 1; }
done
OUT=results/full
LOGS=results/full/logs
mkdir -p "$LOGS"

run() {
  local name="$1"; shift
  local logf="$LOGS/$name.log"
  echo "=== [$(date '+%H:%M:%S')] START $name" | tee -a "$LOGS/sweep.log"
  uv run eval/run_eval.py --out-dir "$OUT" "$@" >>"$logf" 2>&1
  local rc=$?
  echo "=== [$(date '+%H:%M:%S')] END   $name (exit $rc)" | tee -a "$LOGS/sweep.log"
  return $rc
}

# Free whatever ollama is holding before K3 asks for 326 GB.
for m in gemma4:31b gpt-oss:120b gpt-oss:20b; do ollama stop "$m" >/dev/null 2>&1; done
sleep 5

# 1. Confound probe: is K3's weak Croatian an artefact of running it with
#    thinking disabled? n=20 is enough to see a 30-point swing.
run k3-thinking-probe \
  --backend "mlx:$K3" --src "$SRC" --thinking \
  --label K3-thinking-probe --tasks belebele_hrv --limit-belebele 20

# 2. The main event.
run k3-full \
  --backend "mlx:$K3" --src "$SRC" \
  --label K3-REAP80-hr-code --limit-belebele 200

# 3. Comparators. Stop the others first so RSS is not cross-attributed.
for m in gpt-oss:20b gemma4:31b gpt-oss:120b; do
  for other in gemma4:31b gpt-oss:120b gpt-oss:20b; do
    [ "$other" != "$m" ] && ollama stop "$other" >/dev/null 2>&1
  done
  sleep 3
  run "${m//:/-}-full" --backend "ollama:$m" --limit-belebele 200
done

echo "=== [$(date '+%H:%M:%S')] SWEEP COMPLETE" | tee -a "$LOGS/sweep.log"
