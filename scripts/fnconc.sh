#!/bin/sh
# What a long prefill costs the requests already running beside it, and what the prefill chunk
# does to both halves of that trade.
#
# WHY IT CANNOT BE MEASURED ALONE. A bigger prefill chunk holds the device for longer in one
# go, so every decode step queued behind it waits that much longer. Run by itself a long prompt
# only shows the good half -- the chunk is worth tens of percent of prefill -- and the bill
# arrives on other people's sequences. Both halves have to be on the same clock.
#
# WARM FIRST. `expert_tiered` faults its tier in on the first requests, so a cold first pass
# charges the whole cold tier to the alone arm and comes back with contention making the decoders
# FASTER. The first pass ("warm") is discarded; each later pass runs both arms back to back.
#
# THE DECODERS USE ignore_eos SO THEIR LENGTH IS FIXED. A decoder that stops early leaves the
# contended window measuring fewer streams than the alone window did, which reads as contention
# being free.
#
#   P=8100 C=4 DEC=300 PROMPT=24000 scripts/fnconc.sh
#
# Reports, per arm: the decoders' wall alone, their wall with the prefill landing beside them,
# and the prefill's own wall.
P="${P:-8100}"
H="${H:-127.0.0.1}"
C="${C:-4}"                     # co-running decoders
DEC="${DEC:-300}"               # tokens each decoder generates
PROMPT="${PROMPT:-24000}"       # the long request's token target
BPT="${BPT:-4}"
ROOT="${ROOT:-$(dirname "$0")/..}"
URL="http://$H:$P"
CORPUS=/dev/shm/fnconc-corpus.txt

# Grown when a later run wants more than the last one did: `head -c` on a corpus that is too
# short returns what there is rather than failing, so the arm would time a prompt it did not send.
want=$((PROMPT * BPT + 4096))
if [ "$(wc -c < "$CORPUS" 2>/dev/null || echo 0)" -lt "$want" ]; then
    : > "$CORPUS"
    while [ "$(wc -c < "$CORPUS")" -lt "$want" ]; do
        cat "$ROOT"/spec.md "$ROOT"/docs/*.md >> "$CORPUS" 2>/dev/null || exit 1
    done
fi

decoder() {
    curl -s --max-time 900 -X POST "$URL/v1/completions" -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"$1 count upward and do not stop:\",\"max_tokens\":$DEC,\"ignore_eos\":true,\"temperature\":0}" \
        > /dev/null
}

long_request() {
    {   printf 'long %s\n\n' "$(date +%s%N)"; head -c $((PROMPT * BPT)) "$CORPUS"; } > /dev/shm/fnconc-p.txt
    jq -Rs '{prompt: ., max_tokens: 1, temperature: 0}' < /dev/shm/fnconc-p.txt > /dev/shm/fnconc-b.json
    t0=$(date +%s.%N)
    curl -s --max-time 900 -X POST "$URL/v1/completions" -H 'Content-Type: application/json' \
        --data-binary @/dev/shm/fnconc-b.json > /dev/shm/fnconc-out.json
    t1=$(date +%s.%N)
    awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.2f", b - a }'
}

# C decoders, timed. `wait` would also wait on anything else this shell started, so the pids are
# collected and waited on by name.
decoders_wall() {
    t0=$(date +%s.%N); pids=
    i=0; while [ "$i" -lt "$C" ]; do decoder "d$1$i" & pids="$pids $!"; i=$((i + 1)); done
    for p in $pids; do wait "$p"; done
    t1=$(date +%s.%N)
    awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.2f", b - a }'
}

arm() {
    tag="$1"
    alone=$(decoders_wall "a$tag")
    t0=$(date +%s.%N); pids=
    i=0; while [ "$i" -lt "$C" ]; do decoder "c$tag$i" & pids="$pids $!"; i=$((i + 1)); done
    lw=$(long_request)
    for p in $pids; do wait "$p"; done
    t1=$(date +%s.%N)
    cont=$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.2f", b - a }')
    pt=$(jq -r '.usage.prompt_tokens // 0' /dev/shm/fnconc-out.json)
    awk -v al="$alone" -v co="$cont" -v lw="$lw" -v pt="$pt" -v t="$tag" 'BEGIN {
        printf "  %-10s decoders alone %6.2f s   beside a prefill %6.2f s  (%+5.1f%%)   prefill %6.2f s = %.0f tok/s\n",
               t, al, co, 100 * (co - al) / al, lw, pt / lw;
    }'
}

echo "$C decoders x $DEC tokens, long prompt ~$PROMPT tokens, server $URL"
arm "warm"
arm "pass1"
arm "pass2"
