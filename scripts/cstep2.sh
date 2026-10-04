#!/bin/sh
# STEADY-STATE decode step time at concurrency, against a LIVE server (no restart).
#
# ignore_eos pins every sequence to the same length so none retires mid-window; we wait until
# exactly C are RUNNING, sample the counters across a window in the middle, and re-check the
# count at the end -- a window whose edges carry fewer than C sequences reports a step time that
# is not any concurrency's. Model-agnostic: P= picks the port.
P=${P:-8100}
U="http://127.0.0.1:$P"
LEVELS="${LEVELS:-1 2 3 4 5 6 8}"
MT=${MT:-900}
WIN=${WIN:-6}
Q="Write a complete, heavily commented Go implementation of a generic LRU cache with a fixed capacity, O(1) get and put, a doubly linked list, a mutex for concurrent use, table-driven tests for every method, and a long prose explanation of every design decision you made."

m() { curl -s -m 10 "$U/metrics" | grep -F "$1" | grep -v '^#' | awk '{print $NF}' | head -1; }
waitidle() { i=0; while [ $i -lt 240 ]; do [ "$(m 'vllm:num_requests_running')" = "0" ] && return 0; i=$((i+1)); sleep 1; done; return 1; }

printf '%-4s %9s %9s %10s %10s %9s\n' C steps tok tok/step ms/step accept%
for C in $LEVELS; do
  waitidle || { echo "C=$C busy"; continue; }
  i=0
  while [ $i -lt "$C" ]; do
    i=$((i+1))
    curl -s -m 600 -o /dev/null "$U/v1/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"q\",\"prompt\":\"$Q\",\"max_tokens\":$MT,\"temperature\":0,\"ignore_eos\":true}" &
  done
  # ramp: wait until exactly C are running
  j=0; while [ $j -lt 120 ]; do [ "$(m 'vllm:num_requests_running')" = "$C" ] && break; j=$((j+1)); sleep 0.5; done
  sleep 3                                     # let the ramp's ragged edge pass
  [ "$(m 'vllm:num_requests_running')" = "$C" ] || { echo "C=$C never reached"; wait; continue; }
  s0=$(m 'radiance:engine_steps_decode_total'); d0=$(m 'radiance:decode_tokens_total')
  r0=$(m 'radiance:draft_tokens_total');        a0=$(m 'radiance:draft_accepted_total')
  p0=$(m 'radiance:prefill_tokens_total');      T0=$(date +%s.%N)
  sleep "$WIN"
  s1=$(m 'radiance:engine_steps_decode_total'); d1=$(m 'radiance:decode_tokens_total')
  r1=$(m 'radiance:draft_tokens_total');        a1=$(m 'radiance:draft_accepted_total')
  p1=$(m 'radiance:prefill_tokens_total');      T1=$(date +%s.%N)
  run=$(m 'vllm:num_requests_running')
  awk -v C="$C" -v ds=$((s1-s0)) -v dd=$((d1-d0)) -v dr=$((r1-r0)) -v da=$((a1-a0)) \
      -v dp=$((p1-p0)) -v t0="$T0" -v t1="$T1" -v run="$run" 'BEGIN{
    tag = (run+0!=C? "  <- count moved to " run : "") (dp>0? "  <- prefill " dp " in window" : "")
    if (ds<=0) { printf "%-4s %9s\n", C, "NO-STEPS"; exit }
    printf "%-4s %9d %9d %10.3f %10.2f %9.1f%s\n", C, ds, dd, dd/ds, 1000*(t1-t0)/ds, (dr>0?100.0*da/dr:0), tag
  }'
  wait
done
