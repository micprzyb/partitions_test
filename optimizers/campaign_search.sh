#!/usr/bin/env bash
# campaign_search.sh -- autonomous, detached perfect-halver SEARCH engine.
#
# Searches n = 24,23,...,18 (biggest first; bigger n gets more time), running
# parallel ILS+CEGAR waves that reseed from the running frontier, and seeding each
# n from the next-bigger n's best net (transfer bound).  It only SEARCHES and
# persists results to optimizers/campaign_state/n<n>/ (uniq.txt, plus a
# SEARCH_DONE marker when n's time budget elapses).  It performs NO repo edits --
# finalisation (benchmark, header edit, verify, commit, docs) is done separately
# by optimizers/finalize.py, so a bug here can never corrupt the tree.
#
# Fully independent of any Claude session: launch detached and it survives API /
# token / disconnect events for the whole ~120 h.
#
#   nohup taskset -c 2-31 optimizers/campaign_search.sh > /tmp/campaign.out 2>&1 &
#
# Env overrides: TOTAL_HOURS (default 120), JOBS (default 30), WAVE_RESTARTS,
# WAVE_ITERS, FRONTIER_CAP, CORE0 (first core to pin to).
set -u
cd "$(dirname "$0")/.."
CXX=${CXX:-g++}
JOBS=${JOBS:-30}
CORE0=${CORE0:-2}
# Each search process runs a LONG budget (long_restarts x long_iters ILS) so it has
# time to make progress at large n; the event-driven orchestrator (run_n.py)
# resyncs the moment the global bound improves -- it restarts only the laggards
# from the new bound and lets the finders keep their momentum.  So per-process
# runtime is long, but a better bound propagates immediately, not on a timer.
LONG_RESTARTS=${LONG_RESTARTS:-15}
LONG_ITERS=${LONG_ITERS:-1500}
FRONTIER_CAP=${FRONTIER_CAP:-40}
TOTAL_HOURS=${TOTAL_HOURS:-120}
STATE=${STATE:-optimizers/campaign_state}
mkdir -p "$STATE"
LOG="$STATE/progress.log"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Per-n time weights (bigger n -> more time). Sum below scales to TOTAL_HOURS.
declare -A W=( [24]=40 [23]=28 [22]=20 [21]=14 [20]=8 [19]=6 [18]=4 )
WSUM=0; for n in "${!W[@]}"; do WSUM=$((WSUM + ${W[$n]})); done

log "=== campaign_search start: TOTAL_HOURS=$TOTAL_HOURS JOBS=$JOBS waves=${WAVE_RESTARTS}x${WAVE_ITERS} ==="
$CXX -O3 -march=native -std=c++23 optimizers/search_halve_cegar.cpp -o /tmp/sh_campaign 2>>"$LOG" \
  || { log "FATAL: build failed"; exit 1; }
[ -d build ] || cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >>"$LOG" 2>&1

for n in 24 23 22 21 20 19 18; do
  budget_s=$(python3 -c "print(int($TOTAL_HOURS*3600*${W[$n]}/$WSUM))")
  d="$STATE/n$n"; mkdir -p "$d"
  [ -f "$d/SEARCH_DONE" ] && { log "n=$n already SEARCH_DONE, skip"; continue; }

  # Base seeds: header hN + sorter nN, plus the transfer seed from n+1.
  python3 optimizers/make_seeds.py "$n" >>"$LOG" 2>&1
  if [ "$n" -lt 24 ]; then python3 optimizers/reduce_seed.py "$n" >>"$LOG" 2>&1; fi
  cp "/tmp/seeds_$n.txt" "/tmp/seeds_base_$n.txt"

  log "n=$n: budget $((budget_s/60)) min (weight ${W[$n]}/$WSUM); event-driven (long runs ${LONG_RESTARTS}x${LONG_ITERS}, restart-laggards-on-improvement)"
  python3 optimizers/run_n.py "$n" "$budget_s" "$JOBS" "$CORE0" "$LONG_RESTARTS" "$LONG_ITERS" 60 >>"$LOG" 2>>"$d/run_n.err"
  cp "/tmp/seeds_base_$n.txt" "$d/seeds_base.txt" 2>/dev/null || true
  touch "$d/SEARCH_DONE"
  log "n=$n SEARCH_DONE ($(python3 optimizers/merge_reseed.py "$n" "$FRONTIER_CAP" 2>/dev/null))"

  # Finalise THIS n before moving on: benchmark, install the fastest floor net,
  # verify exhaustively (fwd+rev) + full test suite, document + commit -- or
  # revert untouched on any failure (the verify gate).  Done here so each n is
  # added to the repo before the next, and so the campaign needs no live session.
  if [ "${NO_FINALIZE:-0}" != "1" ]; then
    log "n=$n finalizing (benchmark + verify + commit)..."
    python3 optimizers/finalize.py "$n" --commit >>"$LOG" 2>&1 \
      || log "n=$n finalize returned nonzero (reverted or no-improvement; see log)"
  fi
done

touch "$STATE/ALL_DONE"
log "=== campaign_search ALL_DONE ==="
