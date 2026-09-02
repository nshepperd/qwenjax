#!/bin/zsh
# Overnight anti-collapse experiment queue. Continues past individual failures.
cd /home/em/Dev/neural/qwenjax
D=runs/attnmse/overnight
S=scripts/attnmse
run() {
  echo "=== $1 start $(date +%H:%M:%S)" >> $D/driver.log
  uv run python "${@:2}" > $D/$1.log 2>&1 || echo "=== $1 FAILED" >> $D/driver.log
  echo "=== $1 end $(date +%H:%M:%S)" >> $D/driver.log
}
run gen-anchor $S/gen_anchor.py
run temp $S/run_variants.py temp
run noise $S/run_variants.py noise
run resets $S/run_variants.py resets
run rms1 $S/run_variants.py rms1
run anchor $S/run_variants.py anchor
run battery $S/battery.py
echo "=== all done $(date +%H:%M:%S)" >> $D/driver.log
