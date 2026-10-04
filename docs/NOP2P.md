# Running radiance without PCIe P2P — findings (2026-10-04)

Machine: 2× AMD Radeon AI PRO R9700 (`gfx1201`, 31.86 GiB each), ROCm 7.14,
engine built from source at `/home/admin/repos/radiance` (this tree).

## 1. The measurement: there is no peer path

A minimal HIP probe (`hipDeviceCanAccessPeer`, no kernels, no allocations) reports:

```
devices=2
canAccessPeer(0->1) = 0
canAccessPeer(1->0) = 0
```

Supporting platform facts:

- `rocm-smi --showtopo`: GPU0↔GPU1 weight 40, **hops 2**, link type PCIE.
  Two hops means traffic is routed up through the CPU root complex, not GPU↔GPU.
- `lspci`: GPU0 is `0000:01:00.0` behind bridge `00:1c.0`, GPU1 is `0000:02:00.0`
  behind bridge `00:1c.1` — sibling downstream ports of different root ports.
- Kernel cmdline has no IOMMU/ACS overrides
  (`BOOT_IMAGE=... ro quiet splash pcie_aspm.policy=performance pci=realloc pci=nocrs`),
  and PCI extended capabilities (needed to even *read* the ACS flags) are not
  accessible from this user, so ACS state could not be confirmed.
- Realistic expectation: P2P between sibling root ports is not forwardable on
  most root complexes even with ACS tweaks. Treat the platform as no-P2P.

## 2. The exact failure chain on this machine (no guessing)

With `--tp 2`, the engine refuses at model-load time, not at first token:

1. Every architecture plugin declares `all_reduce` / `all_gather` (and fused
   collectives) with `world_size = tp`
   (`core/engine.cpp:142`, `core/engine_bringup.cpp:348`).
2. The selected libr4d row's `init` runs `build()` (`libr4d/r4d_ar_entry.hip:153`),
   which allocates per-rank IPC scratch, publishes pointers through a
   host-side barrier, then calls `r4d_ar_ipc_enable_peer()`
   (`libr4d/r4d_ar_entry.hip:261`).
3. That calls `hipDeviceEnablePeerAccess`, which fails without a peer path, and
   **throws** (`libr4d/r4d_ar_oneshot_2rank_exact.hip:483-488`).
   The throw becomes `RAD_E_DEVICE` (-15) (`r4d_ar_entry.hip:51-65`).
4. Init failing means the op cannot be served → the engine refuses to start.
   (The core-level `rad_dev_enable_peer` in `core/device/hip.cpp:254-274`
   reports the same condition as `RAD_E_UNSUPPORTED` with a message citing spec §9,
   but nothing in the engine calls it — libr4d does its own enabling at declare.)

Independent confirmation from the freshly built test suite (38/41 pass,
`kbench` and `kbench_moe` green — every device kernel is numerically good on
this machine): `r4d_selftest`'s **only** failures out of hundreds of checks are

```
FAIL  2-rank all-gather init (peer rendezvous) -- rc -15/-15
FAIL  2-rank all-reduce init (peer rendezvous) -- rc -15/-15
```

rc -15 is `RAD_E_DEVICE` (`abi/rad_types.h:37`). Everything that does not need
a peer rendezvous passes. (The other two failures are unrelated to P2P:
`ops_doc` needs `gawk` — the script uses `gensub` but the system awk is mawk —
and `sched_test` segfaults; both deserve their own look.)

## 3. Why P2P is load-bearing, not a flag

libr4d's 2-rank all-reduce (`libr4d/r4d_ar_oneshot_2rank_exact.hip:1-25`) is a
one-shot **push** design whose kernels read and write the *peer's* VRAM
directly (`ps[k] = v0` into peer scratch, `:69-93`), coordinated by
system-scope release/acquire flag handshakes (`:41-46, :106-115`). Correctness
rests on three P2P-era assumptions:

- `hipDeviceEnablePeerAccess` succeeds, so a peer pointer is directly
  dereferenceable inside a kernel (single process, one thread per rank, UVA —
  `libr4d/r4d_ar_entry.hip:8-15`).
- Scratch is `hipDeviceMallocFinegrained` (uncached) memory
  (`r4d_ar_oneshot_2rank_exact.hip:304-317`), which is what makes the tuned
  fences sound: default `drain=3` (`s_wait_storecnt`, no L2 writeback) and
  `acq=0`. On coarse-grained memory the peer's bytes can sit in an L2 the
  reader never invalidates — the code detects the fallback and warns that the
  symptom is *occasionally wrong sums*, i.e. silently different text
  (`r4d_ar_entry.hip:209-222`).
- Posted-write ordering over PCIe (data lands before the later release flag).

The author measured this path at 13.52 GB/s per direction against 9.30 for a
kernel doing peer loads and **10.1 for RCCL** (`:326-330`) — RCCL was evaluated
and rejected on performance, which is why "a custom all-reduce kernel" (not
"just use RCCL") is the advice.

## 4. What "a custom all-reduce kernel" concretely means here

The plugin boundary makes this a contained job. Per `docs/PLUGIN.md:352-353`,
"tensor parallelism is ops you supply… the engine inserts none and assumes no
link." Collectives are ordinary rows in a kernel library, and `--kernels`
resolves first-match-wins (`docs/GUIDE.md` §2), so a new library can override
just the collectives and leave every GEMM/attention/norm on libr4d.

### 4a. Minimal surface: only 2 ops, not 6

For `world_size == 2` the vocabulary is (`libr4d/r4d_rows.cpp`):

| op | libr4d rows |
|---|---|
| `all_reduce` | `ar_oneshot_2rank_exact`, `ar_oneshot_2rank_wht6` |
| `all_gather` | `ar_gather_2rank` |
| `ar_hc_write`, `ar_gather_hc_write` | fused gated-residual write (Flash-Next) |
| `ar_ln_had_quant_i8`, `ar_rmsnorm_quant_fp8` | fused norm+quant (both models) |

The fused forms do **not** need reimplementing: every architecture declares
them as a handle that may legitimately not resolve, and falls back to the
unfused pair. E.g. `arch/common/rad_block_hc.h:295-310`: "a handle that did
not resolve is 0 and not an error: a backend without the fused kernel keeps
the pair, and `takes_ar()` then answers no on every step"; same pattern in
`arch/common/rad_fp8.h:467`. So a staged plugin implementing only
`all_reduce` + `all_gather` for `world_size 2` yields a correct, slower,
unfused graph with no architecture changes. (Cost of the fallback: one extra
all-reduce-sized pass over the residual per block plus separate norm/write
launches — measurable, but correct.)

### 4b. Transport design for the staged kernel

- **Init** (per resolved instance, may allocate and rendezvous — that is what
  init is for, `r4d_ar_entry.hip:1-6`): allocate the message-sized slot in
  **pinned host memory** (`hipHostMalloc`, one slot per rank, or a single
  host-side double buffer shared by both ranks since ranks share an address
  space), plus a host-side flag word. Reuse the existing host `Rendezvous`
  barrier pattern (`r4d_ar_entry.hip:71-101`) for pointer publication; skip
  `hipDeviceEnablePeerAccess` entirely.
- **Launch** (must not allocate/synchronise): rankell's kernel (or
  `hipMemcpyAsync` D2H on the rank's stream) pushes its input to the host
  slot; a stream-ordered event or host flag hands it over; the peer's stream
  pulls (H2D) into its own scratch and reduces locally. All traffic is
  GPU↔host over PCIe — no peer mapping required. Ordering must be explicit
  (the free posted-write ordering of §3 does not exist here); the slot
  double-buffering-by-launch-parity scheme (`r4d_ar_seq.h`, `r4d_ar_entry.hip`
  §"ONE INSTANCE PER…") can be kept as-is.
- **Numerics contract**: `ref_all_reduce` (`libref/ref_registry.cpp:1075`) is
  the spec — f32 accumulation, ascending rank order. Match it and greedy
  decode is bit-stable vs `--tp 1`.
- **What it will cost**: decode-step messages are small (eight residual rows,
  ~80 KiB) and latency-bound — host staging adds ~2 PCIe traversals plus
  launch latency per all-reduce per layer (×64 layers on 27B, ×48 on
  Flash-Next); expect single-digit ms/step added at low concurrency.
  Prefill chunks are bandwidth-bound (2048 × 5120 × 2 B ≈ 20 MB per
  all-reduce on 27B): host memory bandwidth (~tens of GB/s) replaces the
  13.5 GB/s P2P path, so prefill slows by roughly 2–4× on the collective
  fraction only. The `wht6` lossy wire is orthogonal and can be skipped in v1
  (serve `exact` only).
- **Out of scope for v1**: 4/8-rank rows, two-shot rows, the `ti8` wire —
  this machine is exactly 2 ranks.

### 4c. How to test it (kbench cannot)

`docs/KERNELS.md:1800` and `tools/rad_kbench.cpp:710`: collective ops are
deliberately excluded from kbench — "a kernel waiting for a peer that never
arrives hangs rather than failing." Validation plan instead:

1. `radiance --model … --tp 2 --kernels staged,… --debug-graph` — every
   collective resolves to the staged row, nothing falls to `libref`
   (which would be a fatal refusal, by design).
2. Golden test: `--deterministic --tp 1` output vs `--deterministic --tp 2
   --kernels staged` on fixed prompts at fixed seeds — must be token-identical
   (the embed-vocab reduction is exact by construction; the block reductions
   match if the staged kernel follows ref's f32 ascending-rank order).
3. `/stats` counters (`prefix_hit_rate`, step times) for the perf delta.

### 4d. Paths not recommended

- **RCCL**: no integration exists in-tree; RCCL without P2P stages through
  the host internally anyway, so it buys no performance over §4b while adding
  a vendored dependency and a new plugin kind. The author's own numbers
  (§3) already rank it below the custom path on the working fabric.
- **Platform surgery** (`pcie_acs_override`, IOMMU tweaks, rebinding):
  unverifiable without root here, and sibling-root-port P2P is unlikely to
  appear regardless (hops=2 through the root complex). Not a plan; at most a
  side experiment.

## 5. The zero-code option: `--tp 1` — VERIFIED WORKING

Every collective declaration in every in-tree architecture is guarded by
`if (g.world > 1)` (e.g. `arch/qwen35_fp8/qwen35_fp8.cpp:942`,
`arch/qwen4exp_fp8/qwen4exp_fp8.cpp:867`, all of `arch/common/rad_block_*.h`,
`arch/common/rad_fp8.h:302`), with `g.world = --tp` (`arch/common/rad_arch.h:726`).
At `--tp 1` the graph contains **zero** collectives: no P2P, no plugin work.
Both models were served and answered inference on 2026-10-04 with the stock
engine built from this tree:

- **27B FP8** (`/home/admin/models/run-qwen27b.sh`): `--tp 1 --max-num-seqs 2
  --max-model-len 8192 --max-num-batched-tokens 2048 --kv-cache-dtype fp8
  --gpu-headroom-mib 256 --deterministic --num-speculative-tokens 0`.
  Fully resident (27.14 GiB weights + 1.36 GiB KV on one card), prefill+decode
  at interactive speed (`/completions` "Say BLUE." → `"\n\nBLUE.\n\nHow"` in
  0.24 s; chat `2+2` → `'4'`, `finish: stop`, with `reasoning_effort: none`).
- **Flash-Next** (`/home/admin/models/run-flashnext.sh`): `--tp 1
  --max-num-seqs 1 --max-model-len 2048 --max-num-batched-tokens 256
  --kv-cache-dtype fp8 --placement expert_tiered --weights-disk-tier
  --host-pool-mib 36864 --gpu-headroom-mib 96 --deterministic
  --num-speculative-tokens 0`. The 4-bit experts need a relayout the file tier
  cannot serve, so 24 GiB of experts live in VRAM + 36 GiB of pinned host
  memory while routed experts stream per layer from the container. Verified:
  chat `2+2` → `'4'`, `finish: stop`. Drive-paced steps; a reference
  deployment, not a production one.

Caveats measured along the way:

- `--tp 2` refuses as predicted: every collective row's `init()` answers
  `device error` (`--debug-selection` output), engine never starts.
- `layer_offload` + `--host-pool-mib` for 27B tp1 is NOT usable on this
  machine: requests stall (one `gemm_fp8a8` launch stops retiring, ~2 packets
  retired per minute at 79% GPU), one run ended in
  `rocr::core::Runtime::VMFaultHandler: "GPU memory access fault"`, and the
  HTTP front holds connections without admitting them (`in_flight: 1`,
  `total_requests: 0`). Identical under the published
  `stilldeadcode/radiance` image, so it is not the host ROCm 7.14 build.
  Suspect: the mover/staging path, not the kernels (kbench is green).
- `libavx` already implements `all_reduce`/`all_gather` for world size 1
  (`libavx/avx_registry.cpp:720-722`), and the sampler handles
  `world_size == 1` (`core/sample/sampler.h:90`) — tp1 is a supported shape,
  not a hack.

## 6. Recommendation — DONE (2026-10-04): libstage

1. **Now**: run 27B at `--tp 1` with a small `--max-model-len` ( §5) — real
   inference today, no code.
2. **Next**: implement the §4b staged-collective plugin (~one new directory in
   the style of `libavx/`, two ops, host-pinned slots, exact wire only), test
   per §4c, and both models then serve at `--tp 2` with full 200 K context.
3. **If max quality-per-step matters more than context**: keep tp1 for 27B
   permanently (no inter-rank lossy wire exists at world 1 by construction)
   and use the staged plugin only where tp2 capacity is needed.

## 7. What was built: `libstage/` (2026-10-04)

The §4b plugin exists and both models serve at `--tp 2` with it
(`--kernels libstage,libr4d`). Differences from the §4b sketch, all forced
by findings during implementation:

- **Six ops, not two.** The architectures use a fused collective wherever one
  resolves (`takes_ar()`/`pick()` are resolution-based, no user switch), so
  `ar_hc_write`, `ar_gather_hc_write`, `ar_rmsnorm_quant_fp8` and
  `ar_ln_had_quant_i8` are implemented too, each as staged exchange plus the
  unfused chain's math, bit for bit (verified below). Declarations asking for
  the `wht6` wire resolve here as well and are served exactly.
- **Mapped host memory, not `hipMemcpy`.** Slots are `hipHostMallocMapped`,
  read/written directly by kernels over PCIe. No events, no IPC handles, no
  allocations or synchronisation in launch.
- **Device-resident launch counter.** A first version counted launches on
  the host; the engine's record-and-replay refused the replay ("issued
  differently from its recording") because the counter argument changed. The
  ticket now comes from a device `atomicAdd`, so launch arguments are
  identical across recording and replay -- the same reason libr4d keeps
  sequence counters in device memory.
- **No `printf` (or anything needing a hostcall) in device code.** Leftover
  debug prints made every launch fail to dispatch on the engine's AQL queues
  ("takes the implicit argument hidden_hostcall_buffer"), which surface as
  errors while the step limps on with garbage -- the failure mode that
  produced the first empty tp2 outputs. Raw HIP streams provide a hostcall
  buffer, so a standalone harness passed while the engine failed; check
  engine logs, not just harness results.
- **One slot, not two**, with `ready`/`done` handshake words (proven
  deadlock-free by ordering: every pre-publish wait is on strictly past
  completions).
- Schemas transcribed from libr4d exactly (leading `--kernels` fixes them
  here); `RADIANCE_TUNE` axes copied so tuned values resolve.

Validation:

- Standalone 2-rank harness (raw HIP, both cards): `all_reduce` (1 M bf16),
  `all_gather` (plain bf16, row-interleaved f32), `ar_rmsnorm_quant_fp8`
  (with/without residual and bf16 out) -- all bit-exact vs CPU reference,
  both ranks.
- Engine `--kld-ref` against `--tp 1`: 27B KL mean **0.002**, top-1 **100%**;
  Flash-Next KL mean **0.035** (MoE-shard and chunk-size reorder noise),
  top-1 **100%**.
- Live: 27B tp2 (`BLUE.`, `4`, `Paris`), Flash tp2 at the full production
  shape (`--tp-wire wht6`, MTP depth 3, 200 K ctx): `4`, `Paris`, zero engine
  errors.
- `layer_offload` tp1 remains broken on this machine (stalled
  `gemm_fp8a8`, HSA VM fault; identical under the published image) and is
  unrelated to collectives -- avoid it; the capacity answer is tp2.

## 8. The padding trap (found 2026-10-05, fixed same day)

Short prompts band the MoE pass to `M` rows while only a few are real, and
the `sorted` rows past the scattered ones hold stale zeros -- thousands of
stale rows all claiming slot 0, which real tokens claim too. The first
inverse-table build (last-wins, single kernel without a cross-block barrier)
handed live slots to padding rows nondeterministically: NaN logits, 7%
top-1, empty generations. Fixed with two ordered kernels (clear, then
scatter) and deterministic first-wins via `atomicMin` -- scatter writes real
rows first, so the lowest index is always real. After the fix: the same
corpus at 85.7% (remaining two flips are mutual top-2 swaps, i.e. tie
noise), long documents at 98.4% top-1. Lesson for any future port: short
prompts are the hardest test here, not the easiest -- they maximize the
stale-to-real ratio.
