#!/bin/sh
# Bring up Qwen3.8-Flash-Next as a long-lived server and LEAVE IT RUNNING: start it, wait for
# /health, print the endpoints and exit with the server still up. The default container is the
# production one, w4nl64-i8lm.rad, built by data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe
# (scripts/convfn.sh).
#
# 200K CONTEXT, AND WHY THE RATIO IS 0.82 RATHER THAN 0.75. --max-model-len reserves nothing per
# request; it sets the WORST CASE that --expert-vs-cache-ratio sizes the paged pool against. At
# 200000 x 8 sequences the default 0.75 binds: the cache takes its full quarter of the elastic pool,
# and the routed experts it displaces fit in neither VRAM nor the host pool, so without
# --weights-disk-tier the planner refuses at load ("DID NOT FIT").
#
# The pool is a CAPACITY, not a per-request cap. A token costs 7.31 KiB across the two paged
# groups (kv_attn 26.00 KiB and kv_qsa_bkey 3.25 KiB per 4-token block), so one 200K sequence is
# 1.43 GiB. 0.87 is enough for one-shot requests (a 200K and a 120K sequence at once) but not
# for multi-turn ones: with prefix caching a finished sequence's blocks stay HELD so the next turn
# can reuse them, so the demand is the live SESSIONS, and a context of T tokens occupies about
# 1.8 x T pool tokens. At 0.87 two ~184K sessions do not fit, the cache evicts exactly what the
# next turn asks for, and every turn re-prefills its whole context (0% prefix hit; minutes per
# turn instead of seconds). 0.82 holds about two concurrent 200K sessions (99% hit on the second
# turn) for about 1.4% of decode step time, since the experts lose 5% of the elastic pool; 0.84
# holds one. Watch kv_util: the hit rate falls off a cliff between 0.75 and 0.81 of the pool.
#
# Defaults are the production shape (gpu-headroom-mib 96, expert-vs-cache-ratio 0.82,
# kv-cache-dtype fp8, max-num-batched-tokens 2048, tp 2, speculation depth 3), with the prefix
# cache on and --checkpoint-slots at its default, because multi-turn chat wants both.
#
# THE CHUNK IS 2048. `--max-num-batched-tokens` is the prefill chunk when checkpoints are off, and
# it sets how many routed rows the MoE GEMM sees at once. Part of the expert weight is not in
# VRAM (it is pinned host memory, read over the link inside the kernel), so a smaller chunk
# re-reads every expert's weight for fewer tokens: 2048 prefills about 55% faster than 512.
#
#   - 4096 IS NOT A 4096 CHUNK: plan_chunk splits at --checkpoint-interval (2048) before the
#     budget applies, so it runs 2048 chunks while the activation arena, which grows with the
#     BUDGET and is charged before the expert/KV split, takes resident experts for a chunk it never
#     gets. Startup names which of the two bound the chunk. A real 4096 chunk (with
#     --checkpoint-interval 4096) is no faster either: the larger arena costs more resident experts
#     than the amortisation returns.
#   - It does not cost co-running decoders: queued behind a prefill that finishes sooner, they
#     finish sooner at 2048 than at 512 (scripts/fnconc.sh measures both halves).
#   - It does not cost the prefix cache: the KV pool loses about 3% to the larger arena and two
#     ~180K sessions still hit 99% with no evictions.
#   - It costs about 0.6% of decode step time at one stream (scripts/cstep2.sh), from the resident
#     experts the arena takes; at eight streams the difference is below what the harness resolves.
#   - It changes the text. A different chunk is a different batch shape, so a prefill sums in a
#     different order and a tie can land on a different token; each chunk size is stable on its
#     own. Pin the chunk before recording anything to compare against later.
#
# THE PREFIX CACHE IS TIERED, AND THE ENGINE LEAVES THE TIERS OFF BY DEFAULT. Both
# --prefix-cache-host-mib and --prefix-cache-disk-mib are zero unless asked for, so a server started
# without them drops a conversation's context the moment the pool needs the blocks. This deployment
# keeps finished turns in host memory and then on disk, where they cost the card nothing.
#
# The ELASTIC half needs no flag and cannot be turned off: the pool holds the blocks it is using
# plus a buffer derived from the chunk, and the expert slab takes the VRAM it releases. That leans
# on --gpu-headroom-mib, which is why the 96 above matters -- startup names any shortfall and holds
# it back on top of the reserve.
#
#   --prefix-cache-host-mib  finished conversations are copied here at once and their VRAM goes
#                            to the expert slab; a return restores instead of re-prefilling. Part
#                            of it is the linear-state snapshot store, sized in proportion to what
#                            a session holds in snapshots against blocks; startup says how many
#                            snapshots it holds. On this model a snapshot is 113 MiB and
#                            --checkpoint-slots counts snapshots for the whole server, not per
#                            sequence, so the snapshot store is the binding resource: without it
#                            only the conversations whose snapshots are still in a device slot
#                            resume without re-prefilling their whole transcript.
#   --prefix-cache-disk-mib  every host copy is also written to disk under --prefix-cache-dir, so
#                            a full host tier gives its oldest slots up for nothing. Costs no RAM
#                            and no VRAM. It holds conversation content UNENCRYPTED.
#
# THE HOST NUMBER IS DELIBERATELY SMALL BECAUSE OF THE N-GRAM TABLE. The tier's arena is PINNED,
# and the 47.68 GiB n-gram (PLE) table is read through the container's own mmap, so it lives in
# the page cache -- and pinned bytes come out of the same host RAM. Taking too much here turns a
# decode step's minor faults into major ones, a far worse trade than a smaller snapshot store.
# Watch /proc/<pid>/stat majflt across a prefill before raising it.
#
# THE CROSS-RANK WIRE IS THE ROTATED 6-BIT ONE (--tp-wire wht6). An all-reduce at or above
# --tp-wire-min-kb (128 KiB: every prefill chunk, and decode from seven streams up) carries a
# Walsh-Hadamard-rotated 6-bit payload instead of bf16 -- 2.6x fewer bytes over PCIe -- inside the
# gated residual's fused writes. Against the exact wire at a 2048-token chunk it prefills about 6%
# faster and decodes about 1% faster at eight streams; at one stream nothing changes, because its
# messages stay under the floor and go exact.
#
# IT IS LOSSY, by about as much as a trade this model already makes. Over greedy continuations of
# 160 code and prose prompts, compared at every position while the two prefixes still agree, wht6
# agrees with the exact wire at 91.4% of positions -- the same as E4M3 hyper-connection matrices
# against bf16 ones (91.3%), which the container already carries. WIRE=exact serves the exact wire.
#
# Binds 0.0.0.0 so a client on another host can reach it. Stop it with scripts/fnstop.sh.
P="${P:-8100}"
M="${M:-$HOME/models/rad/w4nl64-i8lm.rad}"
cd "$HOME/workspace/radiance/build" || exit 1
pkill -f "bin/radianc[e]"; while pgrep -f "bin/radianc[e]" >/dev/null; do sleep 1; done; sleep 2
env GPU_MAX_ALLOC_PERCENT=100 GPU_MAX_HEAP_SIZE=100 $EXTRA_ENV \
nohup ./bin/radiance --model "$M" \
    --radiance-home "$HOME/workspace/radiance/build/radiance_home" \
    --host 0.0.0.0 --port "$P" \
    --max-num-seqs ${SEQS:-8} --max-model-len ${MLEN:-200000} \
    --tp ${TP:-2} --placement expert_tiered --host-pool-mib ${HOSTPOOL:-12288} \
    --gpu-headroom-mib ${HR:-96} --expert-vs-cache-ratio ${RATIO:-0.82} --kv-cache-dtype ${KVD:-fp8} \
    --prefix-cache-host-mib ${KVHOST:-4096} \
    --prefix-cache-dir ${KVDIR:-$HOME/kvcache} --prefix-cache-disk-mib ${KVDISK:-131072} \
    --num-speculative-tokens ${NS:-3} \
    --max-num-batched-tokens ${MNBT:-2048} \
    --tp-wire ${WIRE:-wht6} \
    $EXTRA > "$HOME/fnserve.log" 2>&1 &
echo "starting, log: ~/fnserve.log"
for i in $(seq 1 150); do
  [ "$(curl -s -m 3 -o /dev/null -w %{http_code} http://localhost:$P/health 2>/dev/null)" = 200 ] && break
  pgrep -f "bin/radianc[e]" >/dev/null || { echo "DIED after ${i}x4s"; grep -iE "^E |error|refus" "$HOME/fnserve.log" | tail -20; exit 1; }
  sleep 4
done
[ "$(curl -s -m 3 -o /dev/null -w %{http_code} http://localhost:$P/health 2>/dev/null)" = 200 ] || {
  echo "DID NOT START in 10 min"; grep -iE "^E |error" "$HOME/fnserve.log" | tail -20; exit 1; }
IP=$(ip -4 -o addr show scope global | awk '{split($4,a,"/"); print a[1]; exit}')
echo "UP  pid $(pgrep -f "bin/radianc[e]" | tr "\n" " ")"
echo "  completions  http://$IP:$P/v1/completions"
echo "  chat         http://$IP:$P/v1/chat/completions"
echo "  dashboard    http://$IP:$P/"
echo "  metrics      http://$IP:$P/metrics"
grep -iE "serving on|listening on|KV cache|placement" "$HOME/fnserve.log" | tail -8
