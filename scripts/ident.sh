#!/bin/sh
# ident.sh -- is this server reproducible? One hash per (temperature, question), so a run can be
# compared against another run of the SAME configuration.
#
# WHAT IT PROVES: run-to-run determinism at a fixed configuration, across restarts. On
# Qwen3.8-Flash-Next at tp2 with the MTP head at depth 3, two servers started from scratch give
# the same six hashes: temperature 0, 0.7 and 1.0, both questions.
#
# WHAT IT DOES NOT PROVE: that a SPECULATIVE run matches a NON-SPECULATIVE one byte for byte.
# `narrow_ksplit` chooses the split count of the fp8 GEMM's finisher from M, so the same row
# computed at M=1 (one token at a time) and at M=8 (a verify block) is not bit-identical, and a
# long answer can flip a near-tie somewhere past its first few hundred characters, both
# continuations equally good. `--deterministic` does not change that -- it pins placement and
# never reaches a kernel.
#
# BUT A SHORT COMMON PREFIX IS A DEFECT, NOT A TIE. On Qwen3.8-Flash-Next at tp2, depth 3 against
# --num-speculative-tokens 0 gives the same six hashes here, and over longer answers the two agree
# for hundreds of characters before any flip. A speculative run that leaves the non-speculative
# text within the first sentence or two is computing something different after a rejected draft --
# rolling state read from a rejected token is exactly that shape -- and is worth bisecting. If
# byte-identity across that boundary is ever required everywhere, the work is plumbing a pinned
# split count into the fp8 GEMM, and it costs throughput at one M or the other.
#
#   P=8100 scripts/ident.sh          # against a server already up
P=${P:-8100}
WORK=$(mktemp -d "${TMPDIR:-/tmp}/ident.XXXXXX"); trap 'rm -rf "$WORK"' EXIT INT TERM
ask() {
  jq -n --arg p "$2" --argjson t "$1" \
     '{model:"m",messages:[{role:"user",content:$p}],max_tokens:200,temperature:$t,seed:12345,
       chat_template_kwargs:{enable_thinking:false}}' > "$WORK/iq.json"
  curl -s -m 900 http://localhost:$P/v1/chat/completions -H "Content-Type: application/json" \
       -d @"$WORK/iq.json" | jq -r '.choices[0].message.content' | sha256sum | cut -c1-16
}
Q1="Implement an LRU cache in Go with a fixed capacity and a small test."
Q2="Explain why merge sort is O(n log n), briefly."
for T in 0 0.7 1.0; do
  printf "temp %-4s  %s  %s\n" "$T" "$(ask $T "$Q1")" "$(ask $T "$Q2")"
done
