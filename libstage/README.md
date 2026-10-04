# libstage -- collectives staged through host memory

Tensor-parallel collectives for machines whose GPUs have no PCIe P2P path
between them: `all_reduce`, `all_gather` and the four fused forms
(`ar_hc_write`, `ar_gather_hc_write`, `ar_rmsnorm_quant_fp8`,
`ar_ln_had_quant_i8`), exact wire only, world size 2.

## Why

libr4d's collectives push directly into the peer's VRAM, which needs
`hipDeviceEnablePeerAccess`. On a machine the driver reports no peer path
for -- `hipDeviceCanAccessPeer` answering 0, two hops through the root
complex here -- every one of those rows' `init()` refuses and the engine will
not start a `--tp 2` model. This library stages through `hipHostMallocMapped`
memory instead, which every GPU in the process reads and writes without any
peer mapping: each rank publishes into its own mapped region and a kernel
pulls the peer's over PCIe. All traffic is GPU<->host. See `docs/NOP2P.md`.

## Protocol

One mapped region per rank per `(rank, world, dtype, message bound)` -- the
same sharing libr4d's instance cache does, for the same reason (a slot holds
a full message; a 64-layer model declares over a hundred collectives needing
a handful of them). Layout: `[slot][ready:u32][done:u32]`.

A launch is three dispatches on the rank's stream, all async, none
allocating, none synchronising:

1. `stage_xfer_kernel` (one block): draw the ticket from the instance's
   device-resident counter, spin until the peer finished the previous launch
   (slot reuse), push my bytes, system fence, publish `ready`, spin until the
   peer published. The counter lives in device memory so the launch arguments
   never change -- a recording and its replay issue identically, which a
   host-side counter broke (the engine refused the replay as "issued
   differently").
2. the math: `out[e] = narrow(in[e] + peer[e])` in fp32 ascending rank order,
   or the fused op's math over the same inputs.
3. `stage_mark_done_kernel`: publish completion.

The flags are monotonic, so there is no ABA; a handshake that cannot complete
is a lockstep violation and traps, as libr4d's does. No `hipEvent`, no IPC
handles, no `hipDeviceEnablePeerAccess` anywhere in this library.

## The padding trap (moe gather)

Short prompts band the MoE pass to `M` rows while only a few are real, and
the `sorted` rows past the scattered ones hold stale zeros -- thousands of
stale rows all claiming slot 0, which real tokens claim too. A last-wins
table hands live slots to padding rows nondeterministically (this produced
NaN logits and 7% top-1 agreement in testing). The table here is first-wins
via `atomicMin`, so the lowest index -- always a real row, since scatter
writes real rows first -- wins deterministically. Unclaimed slots keep
`INT_MAX` and are skipped.

## Numerics

The exchange is exact -- full bytes, never quantised -- so an `exact=0`
declaration served here gets the exact sum (the safe direction). Reductions
match the unfused chains they replace bit for bit: hardware-RNE narrow on
the all-reduce store, round-half-up in `hc_write`, RNE gather narrows, RTNE
fp8/i8 quantisers, sequential per-row reductions, scalar-order FWHT, no FMA
(`-ffp-contract=off` via `rad_hipify`). The one deliberate exception is the
wire: declarations asking for the rotated 6-bit payload resolve here too and
are served exactly -- more accurate, never less, at full message bytes.

Measured on dual R9700 without P2P, `qwen3.8-27b-fp8 --tp 2` against the same
model at `--tp 1`: KL mean 0.002, top-1 agreement 100%. `flash --tp 2`
against `--tp 1`: KL mean 0.035 (MoE sharding and chunk-size reorder noise),
top-1 100%.

## Cost

A decode-step all-reduce pays two PCIe traversals plus microsecond
handshakes; a 20 MB prefill message pays them at host-memory rates. Expect a
slower collective fraction than P2P, and no change anywhere else: compute
stays on libr4d. `describe` is null, so steps containing these run eager
rather than graphed -- correct and merely slower.

## Surface

Six rows, world size 2 only, dtypes `bf16`/`fp16`/`fp32` (gather moves bytes
for any of them). Wider worlds, the lossy wires as lossy, and fused forms
beyond these four fall through to whatever else is in `--kernels`. Schemas
are transcribed from libr4d exactly: leading `--kernels libstage,...` fixes
them here, and a disagreement would refuse both plugins at load.
