#!/bin/sh
# oot_plugins.sh BUILD SOURCE WORK [HIP_COMPILER HIP_TARGETS]
#
# THE PLUGINS RADIANCE SHIPS, BUILT THE WAY A THIRD PARTY BUILDS THEIRS. The tree's build is
# installed into a prefix; then every plugin directory -- the kernel libraries, the quantisers,
# every architecture -- is configured ON ITS OWN against that prefix with find_package(radiance),
# built, and installed into a home of its own. Nothing in the tree's build is reused: what is
# tested is the installed package a stranger would have.
#
# What has to hold afterwards:
#   1. the out-of-tree home holds the same plugins, file for file, as the tree's own build put in
#      its home -- no plugin needs the tree to be built;
#   2. every kernel library declares exactly the vocabulary its in-tree twin declares;
#   3. the search path registers the same plugins, in the same order, with the same declarations
#      (rad-info --plugins), from the out-of-tree home alone;
#   4. the installed rad-kbench replays the recorded fixture through the host libraries from that
#      home, and every check passes;
#   5. with HIP_COMPILER given, libr4d is built out of tree too, for HIP_TARGETS, and compiles to
#      the tree's device code. A replay of a device library needs a card, and this test must not
#      touch one a server is using; the same instructions mean the tree's own GPU suite
#      (ctest -L gpu) has already checked what the out-of-tree build produced.
set -eu
B=$1; S=$2; W=$3
HIPC=${4:-}; HIPT=${5:-}
GEN=${CMAKE_GENERATOR:-Ninja}
J=${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || echo 4)}

rm -rf "$W"
mkdir -p "$W/b" "$W/logs"
P="$W/prefix"
H="$W/home"

say() { printf '%s\n' "$*"; }
fail() { say "FAIL: $*"; exit 1; }
run() {   # run LOG CMD... -- quiet unless it fails
  log=$1; shift
  if ! "$@" > "$log" 2>&1; then tail -40 "$log"; fail "$* (log: $log)"; fi
}

say "installing $B into $P"
run "$W/logs/install.log" cmake --install "$B" --prefix "$P"

dirs="libref libquant"
[ -d "$S/libavx" ] && [ -f "$B/radiance_home/kernels/libavx.so" ] && dirs="$dirs libavx"
for d in "$S"/arch/*/; do
  [ -f "$d/CMakeLists.txt" ] && dirs="$dirs arch/$(basename "$d")"
done
[ -n "$HIPC" ] && dirs="$dirs libr4d libstage"

for d in $dirs; do
  n=$(basename "$d")
  say "building $d against the installed package"
  extra=""
  [ "$n" = libr4d ] || [ "$n" = libstage ] && extra="-DCMAKE_HIP_COMPILER=$HIPC -DRAD_GPU_TARGETS=$HIPT"
  # shellcheck disable=SC2086
  run "$W/logs/$n.configure.log" cmake -S "$S/$d" -B "$W/b/$n" -G "$GEN" \
      -DCMAKE_PREFIX_PATH="$P" -DRAD_PLUGIN_INSTALL_HOME="$H" $extra
  run "$W/logs/$n.build.log" cmake --build "$W/b/$n" -j "$J"
  run "$W/logs/$n.install.log" cmake --install "$W/b/$n"
done

# 1. The same plugins, file for file.
sets() {
  for k in kernels architectures quantizers; do
    [ -d "$1/$k" ] && (cd "$1/$k" && ls *.so 2>/dev/null | sed "s|^|$k/|")
  done | sort
}
want=$(sets "$B/radiance_home")
[ -n "$HIPC" ] || want=$(printf '%s\n' "$want" | grep -v '^kernels/libr4d\.so$' || true)
have=$(sets "$H")
if [ "$want" != "$have" ]; then
  say "in-tree:"; say "$want"; say "out of tree:"; say "$have"
  fail "the out-of-tree home does not hold the plugins the tree's build does"
fi
say "1. same plugins: $(printf '%s\n' "$have" | wc -l | tr -d ' ') files"

# 2. The same vocabulary, op for op.
for so in "$H"/kernels/*.so; do
  n=$(basename "$so")
  "$B/bin/rad-schemas" "$B/radiance_home/kernels/$n" > "$W/schemas.tree.$n.txt"
  "$B/bin/rad-schemas" "$so" > "$W/schemas.oot.$n.txt"
  cmp -s "$W/schemas.tree.$n.txt" "$W/schemas.oot.$n.txt" ||
    fail "$n declares a different vocabulary out of tree (diff $W/schemas.*.$n.txt)"
done
say "2. same schemas"

# 3. The same registration, from each home alone. No card: the test must not depend on one, and
# must not touch one a server is using.
listing() {
  HIP_VISIBLE_DEVICES=-1 "$P/bin/rad-info" --plugins --home "$1" 2>/dev/null |
    grep -v '^plugins on' | grep -v '^     /'
}
tree_list=$(listing "$B/radiance_home")
[ -n "$HIPC" ] || tree_list=$(printf '%s\n' "$tree_list" | grep -v ' libr4d ' || true)
oot_list=$(listing "$H")
# The in-tree list numbers libr4d's position when it is left out above; compare without the order
# column and check the order separately by name.
strip() { printf '%s\n' "$1" | sed 's/^ *[0-9]* //'; }
if [ "$(strip "$tree_list")" != "$(strip "$oot_list")" ]; then
  say "in-tree:"; say "$tree_list"; say "out of tree:"; say "$oot_list"
  fail "the plugins register differently out of tree"
fi
say "3. same registration:"
say "$oot_list"

# 4. The installed tools serve from the out-of-tree home. The replay runs every library its home
# holds, so it is given the out-of-tree home less its device library, which has no card here.
HH="$W/hosthome"
mkdir -p "$HH/kernels"
for k in architectures quantizers; do [ -d "$H/$k" ] && ln -s "$H/$k" "$HH/$k"; done
for so in "$H"/kernels/*.so; do
  [ "$(basename "$so")" = libr4d.so ] || ln -s "$so" "$HH/kernels/"
done
run "$W/logs/kbench.log" env HIP_VISIBLE_DEVICES=-1 RADIANCE_HOME="$HH" \
    "$P/bin/rad-kbench" --fixture "$P/share/radiance/kernels.rkb" --max-cases 1
grep -E "passed|failed" "$W/logs/kbench.log" | tail -3
say "4. the installed rad-kbench replays the fixture from the out-of-tree home: $(ls "$HH/kernels" | tr '\n' ' ')"

# 5. The device library's machine code, against the tree's, object by object. A code object also
# carries its compilation unit's ID, which clang hashes from the command line and the output path,
# so two builds of one source never agree byte for byte; their instructions (.text) and kernel
# descriptors do, unless the flags that reach the device compiler differ.
#
# THE DESCRIPTORS ARE COMPARED BY NAME, NOT AS .rodata's BYTES. The compiler lays them out in an
# order that changes from one run of the same command to the next -- ROCm 7's clang does it to the
# eight instances of the template in r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip -- and each holds
# its kernel's entry as an offset from itself, so the section's bytes differ while every kernel and
# every descriptor is the same. Each descriptor is read by its symbol, its entry resolved to the
# address in .text it names, and the rest of .rodata is compared with the descriptors blanked.
if [ -n "$HIPC" ]; then
  L=$(dirname "$HIPC")
  for t in llvm-objcopy llvm-readelf clang-offload-bundler; do
    [ -x "$L/$t" ] || fail "no $t beside $HIPC"
  done
  td="$B/libr4d/CMakeFiles/libr4d.dir"
  od="$W/b/libr4d/CMakeFiles/libr4d.dir"
  mkdir -p "$W/co"
  fatbin() {   # fatbin OBJ -- its fat binary into $W/co/fb; false for a source with no kernels
    "$L/llvm-objcopy" --dump-section .hip_fatbin="$W/co/fb" "$1" /dev/null 2>/dev/null
  }
  descriptors() {   # descriptors CO RODATA -- each descriptor by name; blanks it in RODATA
    set -- "$1" "$2" $("$L/llvm-readelf" -SW "$1" |
                       awk '{ for (i = 1; i < NF; ++i) if ($i == ".rodata") print $(i + 2), $(i + 3) }')
    [ $# -eq 4 ] || return 0
    base=$((0x$3)); foff=$((0x$4))
    "$L/llvm-readelf" -sW "$1" | awk '$4 == "OBJECT" && $8 ~ /\.kd$/ { print $2, $3, $8 }' |
      sort -u -k3 |
      while read -r val size name; do
        a=$((0x$val)); o=$((foff + a - base))
        e=$(od -An -t d8 -j $((o + 16)) -N 8 "$1" | tr -d ' ')
        printf '%s %s entry=%x %s %s\n' "$name" "$size" $((a + e)) \
          "$(od -An -v -t x1 -j "$o" -N 16 "$1" | tr -d ' \n')" \
          "$(od -An -v -t x1 -j $((o + 24)) -N $((size - 24)) "$1" | tr -d ' \n')"
        dd if=/dev/zero of="$2" bs=1 seek=$((a - base)) count="$size" conv=notrunc 2>/dev/null
      done
  }
  code() {     # code OBJ TARGET -- checksums of that target's instructions, its descriptors by
               # name, and the rest of its read-only data
    fatbin "$1"
    "$L/clang-offload-bundler" --type=o --input="$W/co/fb" --unbundle --targets="$2" \
        --output="$W/co/co"
    for sec in .text .rodata; do
      "$L/llvm-objcopy" --dump-section "$sec=$W/co/$sec" "$W/co/co" /dev/null 2>/dev/null ||
        : > "$W/co/$sec"
    done
    cksum < "$W/co/.text"
    descriptors "$W/co/co" "$W/co/.rodata" | cksum
    cksum < "$W/co/.rodata"
  }
  [ "$(ls "$td"/*.hip.o | wc -l)" = "$(ls "$od"/*.hip.o | wc -l)" ] ||
    fail "libr4d compiles a different set of HIP sources out of tree"
  n=0
  for o in "$td"/*.hip.o; do
    f=$(basename "$o")
    [ -f "$od/$f" ] || fail "libr4d's $f has no out-of-tree twin"
    if ! fatbin "$o"; then
      if fatbin "$od/$f"; then fail "libr4d's $f has device code out of tree only"; fi
      continue
    fi
    for t in $("$L/clang-offload-bundler" --type=o --input="$W/co/fb" --list | grep -v '^host-'); do
      [ "$(code "$o" "$t")" = "$(code "$od/$f" "$t")" ] ||
        fail "libr4d's $f compiles to different device code for $t out of tree"
      n=$((n + 1))
    done
  done
  say "5. libr4d's device code is the tree's: $n code objects, the same instructions and descriptors"
fi
say "PASS"
