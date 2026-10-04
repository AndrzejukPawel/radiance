#!/bin/sh
# Prefill throughput against a server that is ALREADY RUNNING, and nothing else.
#
# WHY IT IS NOT A SERVE HARNESS. Prefill speed is a property of the deployment's flags -- the
# chunk, the tensor-parallel width, the expert/cache split -- so a harness that brings its own
# server measures the flags it chose rather than the ones in use. This one attaches to whatever is
# on $P and reports what that server does; use scripts/fnserve.sh to put one there.
#
# THE PROMPT IS PROSE, NOT RANDOM WORDS. A generated word of random letters is three tokens rather
# than one, so a token target computed from a word count is off by a factor of three -- and, worse,
# a stream of tokens the model has never seen together routes differently through a mixture of
# experts than text does, which is the thing a prefill measurement is mostly measuring. The corpus
# is the repository's own markdown, repeated to length.
#
# THE PREFIX CACHE MAKES A SECOND REP FREE, so every rep opens with a different nonce. A cache
# matches from the START of the prompt, so a leading nonce invalidates the whole match while a
# trailing one would leave every block but the last a hit -- and the run would report a prefill
# rate for work that was never done. `cached_tokens` is printed so that this is visible rather
# than assumed.
#
# THE CARD'S CLOCK IS PART OF THE MEASUREMENT. Prefill drifts downward over a sustained run as the
# part heats, monotonically and across runs as well as within them, so a single number is not
# comparable across runs taken at different times. Temperature and sclk are printed beside every
# rep for the same reason a bench prints its build id: without them a thermostat reads as a
# regression.
#
#   P=8100 N=8000 REPS=3 scripts/fnpf.sh
#
# N is a token TARGET reached through an assumed bytes-per-token; the rate is computed from the
# `prompt_tokens` the server reports, so the approximation does not enter the answer.
P="${P:-8100}"
H="${H:-127.0.0.1}"
N="${N:-8000}"
REPS="${REPS:-3}"
BPT="${BPT:-4}"                 # bytes of this corpus per token, measured at about 4.0
ROOT="${ROOT:-$(dirname "$0")/..}"
URL="http://$H:$P"
CORPUS="${CORPUS:-/dev/shm/fnpf-corpus.txt}"

# One corpus, built once and reused, so successive runs prefill the same bytes. GROWN when a
# later run wants more than the last one did: `head -c` on a corpus that is too short silently
# returns what there is, so a sweep that raised N would report a rate for a prompt it did not
# send -- and the prompt_tokens column makes that visible only to a reader who was looking.
want=$((N * BPT + 4096))
if [ "$(wc -c < "$CORPUS" 2>/dev/null || echo 0)" -lt "$want" ]; then
    : > "$CORPUS"
    while [ "$(wc -c < "$CORPUS")" -lt "$want" ]; do
        cat "$ROOT"/spec.md "$ROOT"/docs/*.md >> "$CORPUS" 2>/dev/null || exit 1
    done
fi

card() {
    rocm-smi --showtemp --showclocks --csv 2>/dev/null |
        awk -F, 'NR>1 && $1 ~ /card/ { printf "%s ", $2 }' | head -c 48
}

stats() { curl -s --max-time 5 "$URL/stats"; }

echo "server $URL, target $N tokens, $REPS reps, corpus $(wc -c < "$CORPUS") bytes"
for r in $(seq 0 "$REPS"); do
    {   printf 'request %s %s\n\n' "$r" "$(date +%s%N)"
        head -c $((N * BPT)) "$CORPUS"
    } > /dev/shm/fnpf-prompt.txt
    jq -Rs '{prompt: ., max_tokens: 1, temperature: 0}' < /dev/shm/fnpf-prompt.txt \
        > /dev/shm/fnpf-body.json
    t0=$(date +%s.%N)
    out=$(curl -s --max-time 1800 -X POST "$URL/v1/completions" \
              -H 'Content-Type: application/json' --data-binary @/dev/shm/fnpf-body.json)
    t1=$(date +%s.%N)
    pt=$(printf '%s' "$out" | jq -r '.usage.prompt_tokens // "ERR"')
    ct=$(printf '%s' "$out" | jq -r '.usage.prompt_tokens_details.cached_tokens // 0')
    [ "$pt" = "ERR" ] && { echo "  failed: $(printf '%s' "$out" | head -c 300)"; exit 1; }
    awk -v t0="$t0" -v t1="$t1" -v pt="$pt" -v ct="$ct" -v r="$r" -v c="$(card)" 'BEGIN {
        d = t1 - t0;
        printf "  %-6s %7d tok  %6d cached  %8.3f s  %8.1f tok/s   %s\n",
               (r == 0 ? "warm" : "rep " r), pt, ct, d, pt / d, c;
    }'
done
