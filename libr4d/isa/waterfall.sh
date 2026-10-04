#!/bin/sh
# Every kernel in libr4d that indexes a VGPR array with an index the compiler could not prove
# uniform -- and what that costs.
#
#   BUILD=/path/to/build ./waterfall.sh
#
# WHY THIS IS ITS OWN INSTRUMENT. A divergent VGPR index is not an addressing mode on gfx12. The
# compiler emits `v_movrels_b32` inside an exec-masked loop that re-runs once per distinct index in
# the wave -- a waterfall -- and nothing about the source says so. The shape that produces one is
#
#     for (e = lane; e < n_expert; e += kWave) v[n_own++] = ...;
#     ... later ... for (i = 0; i < n_own; ++i) ... v[i] ...
#
# where `n_own` is derived from `lane`, so every `v[i]` after it indexes divergently and becomes a
# waterfall costing tens of thousands of cycles over a wave. The fix is to unroll to the
# compile-time bound so the indices are constants. res.sh cannot see any of this -- such a kernel
# has no scratch, a modest VGPR count and an unremarkable instruction count. Only the disassembly
# says it.
#
# The count is STATIC, so read it as "look here", not as a cost. A `v_movrels` with a genuinely
# uniform index is cheap and appears here too; what makes it a waterfall is the `s_cbranch_execnz`
# back-edge around it, which the second column reports. And a waterfall in a cold kernel is worth
# nothing: most of what this finds is far below the noise of a step. Check the kernel against a
# trace before touching it.
: "${ROCM:=/opt/rocm}"
: "${BUILD:=build}"
: "${ISA_OUT:=/tmp/r4disa}"
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$ISA_OUT"
printf "%-22s %6s %6s  %s\n" "translation unit" "movrel" "loops" "kernel"
for o in $(find "$BUILD" -name '*.hip.o' | sort); do
  tu=$(basename "$o" .hip.o)
  "$here/co.sh" "$o" >/dev/null 2>&1 || continue
  "$ROCM/llvm/bin/llvm-objdump" -d --symbolize-operands --no-show-raw-insn \
      "$ISA_OUT/$tu.co" 2>/dev/null > "$ISA_OUT/$tu.scan.s" || continue
  awk -v tu="$tu" '
    function flush(  ) {
      if (n > 0) printf "%-22s %6d %6d  %s\n", tu, n, loops, k
      n = 0; loops = 0
    }
    /^[0-9a-f]+ <.*>:$/ { if ($2 !~ /^<L[0-9]+>:$/) { flush(); k = $2 } }
    /movrel/            { n++; seen = 1 }
    /s_cbranch_exec(nz|z)/ { if (seen) { loops++; seen = 0 } }
    END { flush() }
  ' "$ISA_OUT/$tu.scan.s"
done
