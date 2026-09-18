#!/usr/bin/env bash
set -uo pipefail

python_bin=/data/chocheng/.venvs/coverage-fm/bin/python
runner=scripts/run_e12_graph_free_global_generation_v1.py
base=results/e12_graph_free_global_generation_v1/teacher_shards
scene_sets=(
  TR20,TR24,TR28,TR32,TR36
  TR21,TR25,TR29,TR33,TR37
  TR22,TR26,TR30,TR34,TR38
  TR23,TR27,TR31,TR35,TR39
)

pids=()
for index in 0 1 2 3; do
  shard=$((index + 4))
  "$python_bin" "$runner" --stage collect-train-val \
    --output "$base/shard$shard" --scene-ids "${scene_sets[$index]}" \
    >"$base/shard$shard/worker.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in 0 1 2 3; do
  if ! wait "${pids[$index]}"; then
    echo "shard$((index + 4)) failed" >&2
    status=1
  fi
done
exit "$status"
