#!/usr/bin/env bash
# Run eval.n5k_train jobs one at a time from a queue file; append lines while it runs.
#   scripts/n5k_queue.sh results/n5k/train/queue.txt
# Line format: <run name> <eval.n5k_train args...>, or <run name> <shell command> (run from pipeline/)
# when the rest does not start with --. Blank lines and # comments skipped.
# Finished names go to queue.done; output in results/n5k/train/<name>/ and <name>.log.
set -u
Q=$(realpath "$1"); DONE="$Q.done"; OUT=$(dirname "$Q"); touch "$DONE"
cd "$(dirname "$0")/../pipeline"
while true; do
  line=$(grep -v '^\s*#' "$Q" | grep -v '^\s*$' | while read -r n rest; do grep -qx "$n" "$DONE" || { echo "$n $rest"; break; }; done)
  [ -z "$line" ] && { sleep 30; continue; }
  name=${line%% *}; rest=${line#* }
  echo "$(date +%H:%M) start $name: $rest"
  if [[ "$rest" == --* ]]; then
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True conda run -n foodseg --no-capture-output python -m eval.n5k_train ../data/datasets/Nutrition5k \
        $rest --out "$OUT/$name" > "$OUT/$name.log" 2>&1
  else                                   # any other command, run from pipeline/
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash -c "$rest" > "$OUT/$name.log" 2>&1
  fi
  echo "$(date +%H:%M) done $name: $(grep -E '^TEST|Error' "$OUT/$name.log" | tail -1)"
  echo "$name" >> "$DONE"
done
