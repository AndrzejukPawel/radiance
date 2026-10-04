# Tensor parallelism without PCIe P2P: the problem and the fix

This document tells the whole story in one place: why `--tp 2` refused to
start on this machine, what was built to fix it, and how it was proven
correct. The plugin's own documentation lives in `libstage/README.md`;
the raw investigation notes live in `docs/NOP2P.md`.

## 1. The issue

Radiance splits a model across two GPUs with `--tp 2`. After each
parallelised block, the two cards must combine their partial results. That
combination is an **all-reduce** (sum both halves), sometimes fused with the
next operation (a norm, a quantiser, a residual write). Six operations in
total: `all_reduce`, `all_gather`, `ar_hc_write`, `ar_gather_hc_write`,
`ar_rmsnorm_quant_fp8`, `ar_ln_had_quant_i8`.

The bundled kernel library, `libr4d`, implements these by writing **directly
into the other card's memory**: each GPU pushes its bytes into a scratch
buffer that lives in the *peer's* VRAM, then both sides reduce. That only
works if the driver lets the cards map each other's memory
(`hipDeviceEnablePeerAccess`), which requires a PCIe peer-to-peer (P2P) path
between them.

This machine has none:

- `hipDeviceCanAccessPeer` returns 0 in both directions (measured with a
  20-line HIP probe, no kernels involved).
- `rocm-smi --showtopo` reports 2 hops over PCIe between the cards -- traffic
  goes up through the CPU root complex, not card to card. The two R9700s sit
  behind sibling root ports (`00:1c.0` and `00:1c.1`), a layout that rarely
  forwards P2P even with ACS/IOMMU tweaks.

The failure was total and early: at model load, every collective row's
`init()` threw inside `hipDeviceEnablePeerAccess`, the engine printed
`device error` once per collective and refused to start. The test suite told
the same story from the other side: `r4d_selftest` passed hundreds of checks
and failed on exactly two -- the 2-rank all-reduce and all-gather peer
rendezvous (`rc -15`, i.e. `RAD_E_DEVICE`).

So on this machine `--tp 2` was impossible, and every model that needs two
32 GB cards (both target models do at full context) could only run degraded
on one card, if at all.

## 2. Why a new kernel plugin (and not the alternatives)

- **`--tp 1`.** Works with zero code changes (every collective declaration is
  guarded by `g.world > 1`), and was in fact used to get both models running
  first. But a 113 GiB model cannot live on one 32 GiB card except streamed
  from disk, and the 27B only fits with a small context. Not the answer for
  production shapes.
- **RCCL.** The codebase already measured it (10.1 GB/s vs 13.5 GB/s for the
  custom P2P path) and carries no integration for it; without P2P it would
  stage through the host anyway -- all of the work, none of the control.
- **Platform surgery** (`pcie_acs_override`, IOMMU options, rebinding).
  Unverifiable without root here, and sibling-root-port P2P is unlikely to
  appear regardless. Not a plan.
- **Reusing libr4d's fused ops with a different transport.** Impossible from
  outside: the P2P accesses are inside libr4d's kernels, and the architecture
  picks a fused kernel wherever one resolves. The only seam the project
  offers is the one it was designed around: *collectives are ordinary rows
  in a kernel library, and `--kernels` picks first-match-wins*. A new
  plugin implementing just the six collective ops, listed ahead of libr4d,
  diverts exactly the P2P traffic and nothing else. Every GEMM, attention
  and norm stays on libr4d.

That plugin is `libstage/`.

## 3. How libstage works

One process, one thread per rank, shared address space (this is how the
engine runs tensor parallel). Instead of peer VRAM, each rank allocates its
scratch in **`hipHostMallocMapped` memory** -- page-locked host memory that
every GPU can read and write with no peer mapping at all. All traffic is
GPU<->host over PCIe; no peer pointer is ever dereferenced.

Per resolved instance (one per rank, world size, dtype and message bound --
shared across ops exactly like libr4d's cache, because a slot holds a full
message and a model declares over a hundred collectives needing a handful
of them) each rank owns one mapped region: `[message slot][ready flag][done
flag]`. A launch is three dispatches on the rank's stream, all asynchronous
-- nothing allocated, nothing synchronised, so sequences containing them
simply run eager instead of graphed:

1. **Transfer kernel** (one block): draw a ticket from the instance's
   *device-resident* counter, spin until the peer finished the previous
   launch (slot reuse), push my bytes, system fence, publish `ready`,
   spin until the peer published.
2. **Math kernel**: `out[e] = narrow(in[e] + peer[e])`, or the fused op's
   math over the same inputs.
3. **Completion marker**: publish `done`.

The flags are monotonic, so there is no ABA problem; a handshake that
cannot complete is a lockstep violation and traps, as libr4d's does. The
device-resident counter (rather than an obvious host-side counter) exists
because the engine records dispatch sequences and replays them: a host
counter changes launch arguments between recording and replay, which the
engine refuses as "issued differently". The ticket is drawn on-device, so
arguments are identical every issue.

Numerics are exact, always: full bytes cross the link, never a quantised
wire, so an `exact=0` declaration served here gets the exact sum (the safe
direction). Reductions accumulate in fp32 in ascending rank order and narrow
exactly the way each unfused chain they replace does -- hardware-RNE store
for the all-reduce, round-half-up in `hc_write`, RNE gather narrows, RTNE
fp8/i8 quantisers, sequential per-row reductions, scalar-order FWHT, no FMA
(`-ffp-contract=off`). Declarations asking for the lossy 6-bit wire resolve
here too and are served exactly: more accurate, never less. Fused ops reuse
the same exchange and add only their documented math.

## 4. The bugs found while building it

Three were in the new code, one in its assumptions:

1. **Replay refusal** (above): host-side counter broke record-and-replay.
   Fixed with the device counter.
2. **`printf` in device code.** A leftover debug print made every launch
   fail to dispatch on the engine's AQL queues, which provide no hostcall
   buffer -- while passing in a raw-HIP harness, whose streams do. The
   engine logged the refusal and limped on with garbage, which is how the
   first empty model outputs happened. No hostcalls in device code, ever.
3. **Single-kernel inverse table.** The MoE gather builds an inverse table
   (`slot -> sorted row`); building it with init and scatter phases in one
   kernel is a race across blocks. Split into two ordered kernels.
4. **Stale padding in the gather table (the big one).** Short prompts band
   the MoE pass to `M` rows while only a few are real, and the `sorted` rows
   past the scattered ones hold stale zeros -- thousands of stale rows all
   claiming slot 0, which real tokens claim too. Last-wins handed live slots
   to padding rows nondeterministically: NaN logits, 7% top-1 agreement,
   empty generations. Fixed with deterministic first-wins via `atomicMin`
   (scatter writes real rows first, so the lowest index is always real;
   unclaimed slots keep `INT_MAX` and are skipped). After the fix the same
   corpus measures 85.7%, long documents 98.4% top-1.

A useful rule that fell out of this: short prompts are the hardest test
here, not the easiest -- they maximize the stale-to-real ratio.

## 5. What changed in the tree

- `libstage/` (new): `stage.h` (transport, instance cache, rendezvous),
  `stage_transport.cpp` (init/fini, `all_reduce`, `all_gather`),
  `stage_fused.cpp` (the four fused launches), `stage_kernels.hip` (all
  device kernels), `stage_rows.cpp` (six op schemas transcribed from libr4d
  exactly, constraints, tunables, row table, plugin exports),
  `CMakeLists.txt` (standard `rad_add_plugin`, device code only -- skipped
  without HIP), `README.md`.
- `CMakeLists.txt` (root): build `libstage/` whenever the HIP backend is on.
- `tests/oot_plugins.sh`: build `libstage` out-of-tree like `libr4d` (same
  HIP compiler/targets handling), so the file-for-file and schema-identity
  checks cover it.
- Nothing else: no engine, architecture, tool or existing-library changes.
  With `--kernels libstage,libr4d`, libr4d serves everything except the six
  collective ops.

## 6. Validation and how to run it

- Standalone 2-rank harness (raw HIP, both cards): all six ops bit-exact
  against CPU references across shapes (M 1-16, widths to 5120, pitched
  and tight, f32/bf16, shared/no-shared MoE arms), both ranks, repeated.
- Engine `--kld-ref` against `--tp 1`: 27B KL mean 0.002, top-1 100%;
  Flash-Next 98.4% top-1 (PPL ratio 1.03). Remaining flips are mutual
  top-2 swaps -- sharding noise of the class already accepted for lossy
  wires.
- Live, via the repo-root scripts: `./run-qwen27b.sh` (`'4'`, `'Paris'`,
  `'\n\nBLUE.'`, plus a 60-token coherent generation) and
  `./run-flashnext.sh` at the full production shape (`--tp-wire wht6`,
  MTP depth 3, 200K context, expert tiering): `'4'`, `'Paris'`, zero engine
  errors.

Known non-regressions worth knowing: `layer_offload` tp1 wedges on this box
(identical under the published image -- unrelated to collectives);
`sched_test` segfaults and `ops_doc` needs gawk (both pre-date this work);
short-prompt edge cases and the full `ctest` suite (including the lengthened
out-of-tree test) remain good follow-ups.

---

# radiance

An LLM inference server for AMD GPUs, in C++ and HIP, with an OpenAI-compatible HTTP API. Model
architectures, kernel libraries and quantisers are `.so` plugins; the ones in this tree are built
and loaded the same way a third party's are ([`docs/PLUGIN.md`](docs/PLUGIN.md)).

The bundled device kernels (`libr4d`) target **AMD RDNA4** (`gfx1200`, `gfx1201`). The serve flags
below are for **two 32 GB cards** (Radeon AI PRO R9700). The engine itself starts on any AMD GPU
family; another card needs a kernel library for it. A machine with no GPU builds the host backend,
which runs the full test suite but does not serve these models at any useful speed.

If you wish to support the developement of the radiance engine you can do so [here](https://patreon.com/StillDeadCode?utm_medium=unknown&utm_source=join_link&utm_campaign=creatorshare_creator&utm_content=copyLink)

## Models

| model | container | get it | compose file |
|---|---|---|---|
| Qwen3.8-Flash-Next: 4-bit experts, int8 trunk, MTP, vision | `qwen3.8-next-flash-fp8-iq4r-moe.rad` (114 GiB) | [download](https://huggingface.co/StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe) | `flashnext.yaml` |
| Qwen3.8-27B FP8, DFlash2 drafter, vision | `qwen3.8-27b-fp8.rad` (29 GiB) | [download](https://huggingface.co/StillDeadcode/qwen3.8-27b-fp8) | `qwen3.8-27b-fp8.yaml` |
| Qwen3.6-35B-A3B FP8, MTP | `qwen3.6-35b-a3b-fp8.rad` (35 GiB) | [download](https://huggingface.co/StillDeadcode/qwen3.6-35b-a3b-fp8) | `qwen3.6-35b-a3b.yaml` |
| MiniCPM5-2B FP8, DSpark drafter | `minicpm5-2b-fp8.rad` (3 GiB) | [download](https://huggingface.co/StillDeadcode/minicpm5-2b-fp8) | `minicpm5-2b.yaml` |
| MiniCPM5-2B bf16, DSpark drafter | `minicpm5-2b-bf16.rad` (5 GiB) | [download](https://huggingface.co/StillDeadcode/minicpm5-2b-bf16) | `minicpm5-2b-bf16.yaml` |
| Qwen3.8-27B bf16 | `qwen3.8-27b-bf16.rad` (50 GiB) | [convert](#convert-a-checkpoint) | `qwen3.8-27b-bf16.yaml` |
| Qwen3.8-Flash-Next bf16, a disk-streamed reference | `qwen3.8-flash-next-bf16.rad` (335 GiB) | [convert](#convert-a-checkpoint) | `flashnext-bf16.yaml` |

A container holds everything the server needs: the weights, the tokeniser, the chat template and
any drafter. Each download link's repository has the recipe and the exact command that made the
file. Download one into your models directory (the transfer resumes if interrupted):

```sh
mkdir -p /srv/models/rad && cd /srv/models/rad
curl -LC - -O https://huggingface.co/StillDeadcode/minicpm5-2b-fp8/resolve/main/minicpm5-2b-fp8.rad
```

## Run with Docker

The host needs Docker and an `amdgpu` driver that supports ROCm 7.2; it does not need ROCm. The
image is [`stilldeadcode/radiance`](https://hub.docker.com/r/stilldeadcode/radiance), with its
kernels built for `gfx1201`. Pull it, then start a model from its compose file:

```sh
docker pull stilldeadcode/radiance
cp -r deploy/compose ~/radiance-compose && cd ~/radiance-compose
cp .env.example .env                         # set RADIANCE_MODELS, RADIANCE_STATE, RADIANCE_API_KEY, the GIDs
docker compose -f minicpm5-2b.yaml up -d     # or any file from the table
docker compose -f minicpm5-2b.yaml logs -f
docker compose -f minicpm5-2b.yaml down
```

Every model uses both cards and the same port, so run one at a time. Running without compose, and
the reason for each `docker run` option, is in [`docs/DOCKER.md`](docs/DOCKER.md).

To build the image yourself, for example for other cards, run `docker/build.sh -t 'gfx1200;gfx1201'`
(it needs `docker buildx` and tests the build as it goes). Then set `RADIANCE_IMAGE=radiance:latest`
in `.env`.

## Build from source

You need Linux, a C++20 compiler, CMake 3.21+ and Ninja. For the device kernels you also need ROCm
7.2 (`hipcc`; set `ROCM_PATH` if it is not `/opt/rocm`).

```sh
./build.sh                                   # configure, build for the cards present, run the tests
./build.sh --no-hip                          # host backend only
cmake -S . -B build -G Ninja -DRAD_GPU_TARGETS='gfx1201'   # build for cards this machine lacks
cmake --install build --prefix /opt/radiance # optional; bin/, include/, share/radiance/
```

## Run without Docker

```sh
export RADIANCE_HOME=$PWD/build/radiance_home   # the plugins; an installed prefix finds its own
build/bin/radiance --model /srv/models/rad/minicpm5-2b-fp8.rad --port 8000 <flags>
```

These are the flags each model is served with. They match its compose file, minus Flash-Next's
prefix-cache tiers:

| model | flags |
|---|---|
| Qwen3.8-Flash-Next | `--tp 2 --tp-wire wht6 --max-num-seqs 8 --max-model-len 200000 --placement expert_tiered --host-pool-mib 12288 --gpu-headroom-mib 96 --expert-vs-cache-ratio 0.82 --kv-cache-dtype fp8 --num-speculative-tokens 3 --max-num-batched-tokens 2048` |
| Qwen3.8-27B FP8, Qwen3.6-35B-A3B | `--tp 2 --max-num-seqs 32 --max-model-len 200000 --kv-cache-dtype fp8` |
| Qwen3.8-27B bf16 | `--tp 2 --max-num-seqs 8 --max-model-len 200000 --kv-cache-dtype fp8` |
| MiniCPM5-2B FP8 and bf16 | `--tp 2 --max-num-seqs 32 --max-model-len 131072 --kv-cache-dtype fp8 --gpu-headroom-mib 160 --max-num-batched-tokens 512` |
| Qwen3.8-Flash-Next bf16 | `--tp 2 --max-model-len 32768 --placement expert_tiered --weights-disk-tier --host-pool-mib 12288 --gpu-headroom-mib 768 --deterministic --max-num-seqs 32 --max-num-batched-tokens 32768 --num-speculative-tokens 0` |

Every model serves up to 32 sequences a step. Two rows keep 8, because on a hybrid model each
sequence holds its own recurrent state and 32 of them leave the attention cache short of one
200K-token request: Flash-Next at ratio 0.82 (`--max-num-seqs 32 --expert-vs-cache-ratio 0.78`
keeps the 200K request and costs single-stream decode about 15%) and the 27B in bf16. See
[`docs/GUIDE.md`](docs/GUIDE.md) §4.4.

Add `--api-key KEY` to require `Authorization: Bearer KEY`. Add `--debug-graph` to print every op
and the kernel it resolved to, then exit without serving. Then:

```sh
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":50}'
```

`http://localhost:8000/` is the dashboard. [`docs/GUIDE.md`](docs/GUIDE.md) covers the API, memory
budgets, long context, speculative decoding and operating a server.

## Convert a checkpoint

`rad-convert` turns a Hugging Face checkpoint into a container, quantising what a recipe in
`data/recipes/` names. Download checkpoints with `scripts/hfget.sh <org/repo> <dir>`.

```sh
build/bin/rad-convert ~/models/Qwen/Qwen3.8-27B -o qwen3.8-27b-bf16.rad \
    --recipe data/recipes/q38-27b-bf16.recipe
build/bin/rad-convert ~/models/Qwen/Qwen3.8-Flash-Next -o qwen3.8-flash-next-bf16.rad --max-ctx 4096
```

With Docker, use `docker run --rm -v ~/models:/models --entrypoint rad-convert stilldeadcode/radiance ...`. In
the image the recipes are in `/opt/radiance/share/radiance/recipes/`. A checkpoint served at its own
precision needs no container: `radiance --model ~/models/Qwen/Qwen3.8-27B-FP8`.

## Test

```sh
ctest --test-dir build -LE 'gpu|oot'   # everything that runs without a card (a few minutes)
ctest --test-dir build -L gpu          # device suites; they need the cards to themselves
ctest --test-dir build -L oot          # rebuilds every bundled plugin out of tree against an install
```

## Documentation

| | |
|---|---|
| [`docs/GUIDE.md`](docs/GUIDE.md) | installing, converting, serving, the HTTP API, operating a server |
| [`docs/TOOLS.md`](docs/TOOLS.md) | every flag of every tool, and the recipe syntax |
| [`docs/DOCKER.md`](docs/DOCKER.md) | the image and the compose files |
| [`docs/PLUGIN.md`](docs/PLUGIN.md) | writing a kernel library, architecture or quantiser plugin, including one for another AMD card |
| [`docs/ARCHITECTURES.md`](docs/ARCHITECTURES.md) | the supported model families and their geometry |
| [`docs/OPS.md`](docs/OPS.md), [`docs/KERNELS.md`](docs/KERNELS.md) | the op vocabulary, and what `rad-kbench` last measured |
| [`spec.md`](spec.md) | the design |

## License

[Apache-2.0](LICENSE), copyright Deadcode and the radiance contributors. Keep [`NOTICE`](NOTICE)
with any copy or derivative. It credits the authors and lists the MIT-licensed code the tree
includes. The model weights are not covered: each keeps the license of the model it was converted
from. The Docker image also holds third-party libraries under their own licenses, GPL and LGPL
among them; `/usr/share/doc/THIRD-PARTY` in the image lists each one with its version and source.
