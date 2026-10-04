#!/bin/sh
# What happens when a client goes away mid-request, against a server that is ALREADY RUNNING.
#
# WHY THIS EXISTS. A disconnect calls the scheduler's reap at an arbitrary point in the step loop,
# so anything that holds a raw Request* across that point -- the step plan's entry, a drafter's
# snapshot -- is a use-after-free that takes the whole server down. Every other harness in this
# tree waits for its requests, so none of them can see that.
#
# SURVIVING IS NOT THE TEST. A server that stays up and then answers differently has a freed slot
# feeding a live sequence, which is the same defect wearing a quieter shirt. So a reference answer
# is taken first, the drops happen, and the same question is asked again: the hashes have to match.
#
# THE DROP POINTS ARE DERIVED, NOT WRITTEN DOWN. One full request is timed first and the drops are
# placed at fractions of it, because a drop point past the end of the request is not a drop at all
# -- it is curl waiting for a response that already arrived, and it reports PASS having exercised
# nothing.
#
# `--max-time` is the disconnect: curl closes the socket where it lands. A drop is confirmed by
# `running` falling back to zero well before the request would have finished on its own.
#
# EVERY OBSERVATION COUNTS, NOT THE LAST ONE. A transient divergence right after a burst is the
# interesting case and a final re-check hides it: ask again and the answer is right, so the run
# reports clean. The verdict below is over all of them.
#
# AND ONE DIVERGENCE IS NOT A FINDING HERE. A single completion hash can diverge on an unchanged
# binary -- concurrency decides batch composition and greedy decoding differs at a tie -- so a
# diverged row means RUN IT AGAIN, several times, before believing it. The reference question
# itself is stable against an idle server; it is the contended path that can tie-flip.
#
#   P=8100 scripts/fndrop.sh
P="${P:-8100}"
H="${H:-127.0.0.1}"
ROOT="${ROOT:-$(dirname "$0")/..}"
URL="http://$H:$P"
C=/dev/shm/fndrop-corpus.txt
BPT=3
LONG="${LONG:-20000}"                  # tokens of prompt, so prefill is a measurable slice
GEN="${GEN:-3000}"                     # tokens of generation, so decode is a long one too
want=$((LONG * BPT + 8192))
touch "$C"
[ "$(wc -c < "$C")" -ge "$want" ] || { : > "$C"
    while [ "$(wc -c < "$C")" -lt "$want" ]; do cat "$ROOT"/spec.md "$ROOT"/docs/*.md >> "$C" 2>/dev/null || exit 1; done; }

alive()   { pgrep -f "bin/radianc[e]" >/dev/null && echo ALIVE || echo DEAD; }
healthy() { curl -s -m 5 "$URL/health" >/dev/null 2>&1 && echo ok || echo unreachable; }
running() { curl -s -m 5 "$URL/stats" | jq -r '.running // "?"'; }

ref() {
    curl -s -m 300 -X POST "$URL/v1/completions" -H 'Content-Type: application/json' \
        -d '{"prompt":"List the first five prime numbers, comma separated.","max_tokens":48,
             "temperature":0}' | jq -r '.choices[0].text // "ERR"' | sha256sum | cut -c1-16
}

body() {
    { printf 'drop %s\n\n' "$(date +%s%N)"; head -c $((LONG * BPT)) "$C"; } > /dev/shm/fndrop-p.txt
    jq -Rs --argjson g "$GEN" '{prompt: ., max_tokens: $g, ignore_eos: true, temperature: 0}' \
        < /dev/shm/fndrop-p.txt > /dev/shm/fndrop-b.json
}
send() { body; curl -s -m "$1" -o /dev/null -X POST "$URL/v1/completions" \
                 -H 'Content-Type: application/json' --data-binary @/dev/shm/fndrop-b.json 2>/dev/null; return 0; }

echo "server $URL, prompt ~$LONG tokens + $GEN generated"
base=$(ref)
t0=$(date +%s.%N); send 900; t1=$(date +%s.%N)
FULL=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.0f", b-a}')
[ "$FULL" -lt 6 ] && { echo "a full request is only ${FULL}s -- raise LONG or GEN so the phases are separable"; exit 1; }
printf '  a full request takes %ss; dropping inside it\n' "$FULL"
printf '  %-40s %-7s %-10s %-9s %s\n' step alive health 'running+2s' answer
printf '  %-40s %-7s %-10s %-9s %s  <- reference\n' "before any drop" "$(alive)" "$(healthy)" "-" "$base"

BAD=0; SEEN=0
observe() {   # $1 = label; records the answer against the reference rather than only printing it
    a=$(ref); SEEN=$((SEEN + 1)); [ "$a" = "$base" ] || BAD=$((BAD + 1))
    printf "  %-40s %-7s %-10s %-9s %s%s\n" "$1" "$(alive)" "$(healthy)" "$(running)" "$a" \
           "$( [ "$a" = "$base" ] || echo "  <- DIVERGED" )"
}
drop() {   # $1 = seconds, $2 = label
    send "$1"; sleep 2; observe "$2"
}
drop $((FULL / 8  + 1)) "dropped at 1/8 of it (early prefill)"
drop $((FULL / 3  + 1)) "dropped at 1/3 of it"
drop $((FULL / 2  + 1)) "dropped at 1/2 of it"
drop $((FULL * 3 / 4))  "dropped at 3/4 of it (deep in decode)"

for s in $((FULL / 4 + 1)) $((FULL / 3 + 1)) $((FULL / 4 + 1)) $((FULL / 2 + 1)); do send "$s" & done
send 900 &
wait
sleep 2
observe "four concurrent drops + one survivor"

echo
if [ "$BAD" -eq 0 ]; then
    printf '  all %d answers matched the reference\n' "$SEEN"
else
    printf '  %d of %d answers DIVERGED from the reference -- run it again before believing it\n' \
           "$BAD" "$SEEN"
fi
