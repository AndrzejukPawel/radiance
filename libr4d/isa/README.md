# libr4d/isa -- reading the machine code the kernels actually became

Seven scripts and four microbenchmarks. They exist because "this kernel is slow" and "this kernel
is at the wall" look identical from a wall-clock timer, while the disassembly plus one calibration
number tells them apart without a GPU run at all. The trace tools are the exception: they need the
GPU, and they are the only instruments here that report the geometry a kernel was ACTUALLY launched
with rather than the geometries it could have been.

    BUILD=/path/to/build ./res.sh | head -40        # every kernel: vgpr/lds/scratch
    BUILD=... ./dis.sh r4d_gemm_fp8a8 ILb1ELb0ELj8ELj1ELj8ELj64ELb0E
    BUILD=... ./waterfall.sh                        # divergent VGPR indices, per kernel
    ./loop.sh                                       # its K loop, classified
    ./trace.sh traces/dec -- ./bin/radiance ...     # then: ./tsum.sh traces/dec/k_kernel_trace.csv

`res.sh` prints the mangled name; the substring `dis.sh` takes is a template instantiation out of
it, and the parameter order is in the kernel's own declaration (`gemm_fp8a8_kernel` is
`<Full, AW, NT, MT, MW, FK, GATED>`, so `ILb1ELb0ELj8ELj1ELj8ELj64ELb0E` is the prefill tile).
`loop.sh` anchors on `v_wmma` by default and takes the SMALLEST backward branch containing every
anchor -- a plain largest-loop search finds the epilogue, which is a long unrolled store chain.

## The tools

| tool | what it answers |
|---|---|
| `co.sh` | extracts the gfx1201 code object out of one built `.hip.o`; everything else reads that |
| `dis.sh` | disassembles ONE template instantiation out of one translation unit |
| `loop.sh` | isolates the working loop of that disassembly and classifies its body by instruction class |
| `res.sh` | one row per device kernel in a build tree: VGPRs, SGPRs, LDS, scratch, spills, waves a SIMD, workgroup size |
| `waterfall.sh` | every kernel that indexes a VGPR array with an index the compiler could not prove uniform |
| `trace.sh` | runs a command under `rocprofv3 --kernel-trace` and leaves a CSV |
| `tsum.sh` | summarises that CSV over a steady-state window: where device time goes, per (kernel, grid) |
| `gaps.sh` | the other half -- where the device is IDLE, and which op it was waiting for |
| `rskew.sh` | per-kernel device time on each agent, and the skew between ranks |
| `isarate.hip` | what one instruction slot is worth on gfx1201 (below) |
| `graphrate.hip` | what a dispatch costs the HOST to submit, and what capturing a fixed sequence saves |
| `launchreg.hip` + `launchreg_lib.hip` | whether a launch costs more when the module registers more kernels, and what crossing into a `.so` costs |
| `blockcopy.hip` | contiguous versus scattered device-to-host copy, at KV-block granularity |

Each script's own header states its traps and how to read its output. The microbenchmarks are built
by hand; the command line is in each file's header, and `build_launchreg.sh` builds the two-object
pair the registration test needs.

## What an instruction slot is worth on gfx1201

`isarate.hip` runs each instruction back to back in eight independent chains, so nothing stalls but
issue. The ratio it reports is clock-independent, which is why it reads no clock.

**One `v_wmma_f32_16x16x16_fp8_fp8` costs about 12.9 plain VALU issue slots**, and that does not
change when 36 KiB of LDS holds occupancy down to what a real prefill tile gets (at that occupancy
it measures nearer 11.6, for a peak of about 434 TFLOP/s). So a kernel's ceiling is

    TFLOP/s = 434 * wmma_slots / (wmma_slots + other_instructions_issued)

with `wmma_slots = 12.9 * (wmma count)`, both counts per loop iteration out of `loop.sh`.

**It is a CEILING and not a prediction**, and the difference matters: a kernel cannot beat it, and a
kernel far under it is under it for some other reason that then has to be found. Issue is frequently
not the binding constraint. Cutting a kernel's non-WMMA instruction count by more than half can make
it SLOWER, because the removed work was covering memory and LDS latency and its removal raises
register pressure; and a memory-bound kernel moves by about a percent under the same cut. Use the
model to rule a lever out, not to predict a gain.

gfx1201 has **no dense 16x16x32 fp8 WMMA** -- `llvm-mc -mcpu=gfx1201` rejects it. The only 32-deep
fp8 form is `v_swmmac_f32_16x16x32_fp8_fp8`, which is the 2:4 sparse one. The GEMMs here are already
on the widest dense op that exists.

## The profiler: what works and what does not

`rocprofv3 --pmc` does not work on this platform. Any counter set, any kernel filter, on this tree
or on `isarate` alone: it collects, then hangs in `queue.cpp:939 Timeout while waiting for queue
sync: 1 kernels still active` and aborts on SIGABRT. **There are no hardware counters**, so the
dynamic instruction count is not directly observable -- `loop.sh` plus knowing which blocks are
predicated is how to get it.

`--kernel-trace` works, unmasked. The abort that names an unsupported agent

    rocprofiler_iterate_agent_supported_counters failed for agent 3 (gfx1036)
      :: Agent HW architecture is not supported, no counter metrics found.

is **counter** enumeration and appears only under `--pmc`. It names a display GPU integrated on the
CPU, which has no counter support; masking it does not make `--pmc` work, and `--kernel-trace` never
needed it masked. `trace.sh` sets `ROCR_VISIBLE_DEVICES=0,1` only to pin the agent set and keep that
line out of the log.

`rocprofv3` is not on `PATH`; it is `/opt/rocm/bin/rocprofv3`.

**A killed `rocprofv3` leaves a zombie that outlives its own `timeout`.** The hang is in
finalisation, after the alarm has fired, so the child ignores the timeout and sits in queue-sync
indefinitely. It does not hold CUs, so it does not poison a timing measurement -- but it does hold
VRAM, which makes the *next* tool report allocation failures that read like miscompiled kernels.
Always follow a `--pmc` attempt with `pkill -9 -f rocprofv3`; `trace.sh` does this itself.

### Reading a trace

* Tracing costs about 2 us per dispatch per RANK. The ranks are separate host threads and inflate in
  parallel, so per-rank dispatches are what count, not the total.
* Kernel **durations** are hardware timestamps and do not move under tracing. The **gaps** absorb the
  whole interception cost, so busy time is real and idle time is an upper bound. `tsum.sh` does that
  subtraction and prints both.
* Generated text does not change under tracing, so a traced run can be pointed at a real serving
  configuration rather than a proxy.
* **Busy is not utilisation, and this is the easiest number here to misread.** A dispatch counts as
  busy if it is resident at all, so a 5 us kernel holding 1 of 64 CUs reads exactly like a GEMM
  saturating the card. "96% busy" says the step is almost never *waiting* for a launch. It says
  nothing about occupancy -- for that, read the blocks column, where anything under ~64 cannot fill
  the device however long it runs.
* **A trace can go quiet and still look complete.** A run that idles before its workload records the
  startup and nothing else, and the CSV is well formed either way. Send the workload the moment
  health answers, keep the run short, and check `dispatches/step` in `tsum.sh`'s header against what
  the model should issue before believing any table built from it.

## What the instruments say in general

Two results recur often enough to be worth stating up front, because they decide which levers are
worth pulling at all.

**Every dispatch costs a few microseconds before it does anything.** Across the shapes a decode step
asks the narrow fp8 GEMM for, the time is not six different rates but one line,

    t = fixed + bytes / achievable_stream_rate

with a fixed term of roughly 4.5 us and a stream rate near the card's achievable bandwidth. The same
fixed cost shows on every small op -- rmsnorm, quant_act, add, gated_quant -- whatever it moves. So
a "slow" small shape is usually a fixed cost failing to amortise, not a kernel with bandwidth left
in it, and **fewer kernels** is the only lever that reaches it.

**Most of the weight-streaming kernels are at the memory wall.** A narrow GEMM whose K loop issues a
few hundred instructions per few kilobytes of weight per wave has roughly four times more issue
capacity than the card has bandwidth, and measures at the achievable stream rate. Assembly-level
work on such a kernel is not worth doing; confirm the reading with `loop.sh` before spending any.

**Read the kernel's own header before proposing anything about it.** Each hot kernel's source
carries its own ablations, sweeps and rejected arms with the reason each was rejected, and the
answer to most assembly-level ideas is already recorded there.
