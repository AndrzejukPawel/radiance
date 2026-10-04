#!/bin/sh
# kbench-analysis.sh -- the half of a kernel library's health that is not a number rad-kbench can
# measure: what a static analyser says about the source, and what a sanitised build says about the
# run.
#
# It writes ONE markdown fragment, which rad-kbench splices into its report with --analysis. That
# is deliberate rather than having the tool shell out: the tool is what somebody runs on a card,
# this is what somebody runs in CI on a machine that may have no GPU at all, and a report that
# needs both present to render is a report that renders nowhere.
#
# Usage: tools/kbench-analysis.sh <build-dir> <out.md>
#
# What it does NOT do is fail the build. Every finding here is advisory -- clang-tidy on a HIP
# translation unit reports things that are true of the language and not of the code, and a
# sanitised replay is a different binary from the one that ships. The numbers belong in the report
# so somebody reads them; gating on them would mean somebody turns the gate off.
set -eu

BUILD=${1:-build}
OUT=${2:-analysis.md}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CC_JSON="$BUILD/compile_commands.json"

TIDY=$(command -v clang-tidy || echo /opt/rocm/llvm/bin/clang-tidy)

{
  echo "## Static and dynamic analysis"
  echo
  echo "Advisory, and generated separately from the measurements above:"
  echo "\`tools/kbench-analysis.sh\` runs on a machine that need not have a GPU."
  echo
} > "$OUT"

# ---------------------------------------------------------------- clang-tidy
# Only the checks that mean something for this code. `bugprone-*` and `clang-analyzer-*` are the
# ones that find real defects; the readability and naming packs would bury those under thousands
# of style opinions this tree has already decided against.
CHECKS='-*,bugprone-*,clang-analyzer-*,performance-*,-bugprone-easily-swappable-parameters,-bugprone-narrowing-conversions'

if [ ! -f "$CC_JSON" ]; then
  echo "clang-tidy: skipped, no $CC_JSON (configure with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)" >> "$OUT"
  echo >> "$OUT"
elif [ ! -x "$TIDY" ] && ! command -v clang-tidy > /dev/null 2>&1; then
  echo "clang-tidy: not installed on this machine" >> "$OUT"
  echo >> "$OUT"
else
  TMP=$(mktemp)
  # Host C++ only. A HIP translation unit needs the offload flags clang-tidy will not replay, and
  # it reports the __device__ attributes as errors rather than analysing the code.
  for f in $(find "$ROOT/libref" "$ROOT/core" "$ROOT/tools" -name '*.cpp' | sort); do
    "$TIDY" -p "$BUILD" --checks="$CHECKS" --quiet "$f" 2>/dev/null || true
  done > "$TMP"

  N=$(grep -c "warning:" "$TMP" || true)
  echo "### clang-tidy" >> "$OUT"
  echo >> "$OUT"
  echo "\`bugprone-*\`, \`clang-analyzer-*\` and \`performance-*\` over every host translation unit" >> "$OUT"
  echo "(\`libref/\`, \`core/\`, \`tools/\`). HIP units are excluded: clang-tidy cannot replay the" >> "$OUT"
  echo "offload flags and reports \`__device__\` as an error rather than analysing the body." >> "$OUT"
  echo >> "$OUT"
  echo "- $N warning(s)" >> "$OUT"
  if [ "$N" -gt 0 ]; then
    echo >> "$OUT"
    echo "| check | count |" >> "$OUT"
    echo "|---|---|" >> "$OUT"
    grep -o '\[[a-z][a-z-]*-[a-z][a-z.-]*\]$' "$TMP" | sort | uniq -c | sort -rn |
      while read -r n c; do echo "| \`${c}\` | $n |" >> "$OUT"; done
  fi
  echo >> "$OUT"
  rm -f "$TMP"
fi

# ---------------------------------------------------------------- sanitised replay
# A second build tree, because -fsanitize changes every object and mixing them is how you get a
# link that succeeds and a binary that reports nothing.
echo "### AddressSanitizer + UndefinedBehaviorSanitizer" >> "$OUT"
echo >> "$OUT"
SAN="$BUILD-asan"
if [ -x "$SAN/bin/rad-kbench" ]; then
  LOG=$(mktemp)
  ASAN_OPTIONS=detect_leaks=0:abort_on_error=0 \
  UBSAN_OPTIONS=print_stacktrace=1 \
    "$SAN/bin/rad-kbench" --home "$SAN/radiance_home" > "$LOG" 2>&1 || true
  E=$(grep -c "ERROR: AddressSanitizer\|runtime error:" "$LOG" || true)
  echo "A sanitised replay of the whole fixture. Host code only -- see the note in the" >> "$OUT"
  echo "**Memory safety** section for why the device side cannot be sanitised on this card." >> "$OUT"
  echo >> "$OUT"
  echo "- $E report(s)" >> "$OUT"
  if [ "$E" -gt 0 ]; then
    echo >> "$OUT"
    echo '```' >> "$OUT"
    grep -h "ERROR: AddressSanitizer\|runtime error:" "$LOG" | sort | uniq -c | head -40 >> "$OUT"
    echo '```' >> "$OUT"
  fi
  rm -f "$LOG"
else
  echo "Not run: no sanitised build at \`$SAN\`. Build one with" >> "$OUT"
  echo >> "$OUT"
  echo '```sh' >> "$OUT"
  echo "cmake -S . -B $SAN -DRAD_ASAN=ON && cmake --build $SAN -j8" >> "$OUT"
  echo '```' >> "$OUT"
fi
echo >> "$OUT"
echo "analysis written to $OUT" >&2
