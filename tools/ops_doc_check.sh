#!/bin/sh
# ops_doc_check.sh -- hold docs/OPS.md to the schemas the plugins actually declare.
#
# OPS.md is the authority for the ops it defines, and a document that claims that and is not
# checked will drift: the ways it drifts are quiet ones -- a dropped optional operand,
# a missing `:inout` marker, a parameter that moved into an operand, a section describing a
# parameter no plugin declares. None of those look wrong on the page.
#
# The cost is that an architecture plugin gets written against the DOCUMENT. An op that lives in a
# kernel library and in no schema anyone checked is an op whose caller is guessing, and a document
# that is wrong is worse than one that is missing, because it is read instead of the source.
#
#     tools/ops_doc_check.sh docs/OPS.md build/bin/rad-schemas build/radiance_home/kernels/*.so
#
# A row with a FOURTH column is exempt and its marker is printed -- that is how a proposal stays in
# the document without being a lie. Exit 1 on any disagreement; exit 77 -- ctest's
# SKIP_RETURN_CODE -- when a plugin will not dump, so a box without a card reports honestly
# rather than green. The code and not a "  SKIP " line, because ctest evaluates
# SKIP_REGULAR_EXPRESSION BEFORE the return code and a skip line would then mask a real
# disagreement found later in the same run.
set -e
[ $# -ge 3 ] || { echo "usage: $0 <docs/OPS.md> <rad-schemas> <plugin.so>..." >&2; exit 2; }
doc=$1; schemas=$2; shift 2
[ -r "$doc" ] || { echo "  SKIP no $doc"; exit 77; }
[ -x "$schemas" ] || { echo "  SKIP no rad-schemas at $schemas"; exit 77; }

dump=$(mktemp); trap 'rm -f "$dump"' EXIT
for so in "$@"; do
    [ -r "$so" ] || { echo "  SKIP no plugin at $so"; exit 77; }
    "$schemas" "$so" >> "$dump" 2>/dev/null || { echo "  SKIP $so would not dump"; exit 77; }
done

awk -v doc="$doc" '
# ---- tool notation: name:type[/optional|/derived], name:role[?] ----------------------------
function t_param(t,   f,o) {
    if (sub(/\/optional$/, "", t)) o = "/opt"; else if (sub(/\/derived$/, "", t)) o = "/der"
    f = (t ~ /:f64$/) ? ":f64" : ""; sub(/:(int|str|f64)$/, "", t); return t f o }
function t_opd(t,   q,w,io) {
    if (sub(/\?$/, "", t)) q = "?"
    # A WEIGHT TABLE is a weight operand whose issue names a RUN of handles, so it is annotated
    # apart: `w`wt in the document. See docs/OPS.md and RAD_WTAB in abi/rad_runtime.h.
    if (sub(/:wtab$/, "", t)) w = "wt"
    if (sub(/:weight$/, "", t)) w = "w"
    if (sub(/:inout$/, "", t)) io = ":inout"
    sub(/:(in|out)$/, "", t); return t w io q }
# ---- doc notation: `name` followed by its annotations -------------------------------------
function d_name(t,   a) { a = index(substr(t, 2), "`"); return substr(t, 2, a - 1) }
function d_rest(t,   a) { a = index(substr(t, 2), "`"); return substr(t, a + 2) }
function d_param(t,   n,r,f,o) {
    if (substr(t, 1, 1) != "`") return ""
    n = d_name(t); r = d_rest(t); sub(/\(range\)/, "", r)
    f = (r ~ /:f64/) ? ":f64" : ""
    if (r ~ /\/opt/) o = "/opt"; else if (r ~ /\/der/) o = "/der"
    return n f o }
function d_opd(t,   n,r,w,io,q) {
    if (substr(t, 1, 1) != "`") return ""
    n = d_name(t); r = d_rest(t)
    if (substr(r, 1, 2) == "wt") w = "wt"; else if (substr(r, 1, 1) == "w") w = "w"
    if (r ~ /:inout/) io = ":inout"
    if (r ~ /\?/) q = "?"
    return n w io q }
# ---- the dump: FIRST declarer fixes the schema, which is the loader s rule (spec 2.3) -------
FILENAME != doc {
    if ($0 ~ /^  [a-z]/)         { op = $1; seen[op] = 1; next }
    if ($0 ~ /^      params /)   { s = ""; for (i = 2; i <= NF; i++) s = s (i > 2 ? " " : "") t_param($i)
                                   if (!(op in P)) P[op] = s; next }
    if ($0 ~ /^      operands /) { s = ""; for (i = 2; i <= NF; i++) s = s (i > 2 ? " " : "") t_opd($i)
                                   if (!(op in O)) O[op] = s; next }
    next }
# ---- the doc rows --------------------------------------------------------------------------
/^\| `/ {
    n = split($0, c, "|")
    op = d_name(gensub(/^ +| +$/, "", "g", c[2])); documented[op] = 1
    if (n > 5) { exempt++; printf "exempt  %-24s line %d -- %s\n", op, FNR,
                        gensub(/^ +| +$/, "", "g", c[5]); next }
    pc = c[3]; oc = c[4]; gsub(/[,;]/, " ", pc); gsub(/[,;]|→/, " ", oc)
    ps = ""; np = split(pc, pa, /[ \t]+/)
    for (i = 1; i <= np; i++) { v = d_param(pa[i]); if (v != "") ps = ps (ps == "" ? "" : " ") v }
    os = ""; no = split(oc, oa, /[ \t]+/)
    for (i = 1; i <= no; i++) { v = d_opd(oa[i]);   if (v != "") os = os (os == "" ? "" : " ") v }
    rows++
    # An op no loaded plugin declares is not a disagreement, it is a row this CONFIGURATION cannot
    # check. A host-only build loads the reference library and no device library, so every
    # device-only op in the vocabulary is undeclared here -- and failing on that would make the
    # suite red on a machine with no card, which is a supported configuration (spec 18). It is
    # counted and named, so the coverage is visible rather than assumed.
    if (!(op in P)) { unchecked++; u_list = u_list (u_list == "" ? "" : " ") op; next }
    ok = 1
    if (ps != P[op]) { ok = 0; printf "PARAMS  %-24s line %d\n          doc  %s\n          real %s\n", op, FNR, ps, P[op] }
    if (os != O[op]) { ok = 0; printf "OPERAND %-24s line %d\n          doc  %s\n          real %s\n", op, FNR, os, O[op] }
    if (!ok) bad++ }
END {
    for (op in seen) if (!(op in documented)) {
        bad++; printf "UNDOCUMENTED %-20s declared by a plugin, in no table\n", op }
    if (unchecked) printf "\n%d op(s) no loaded plugin declares, so unchecked here: %s\n",
                          unchecked, u_list
    printf "\n%s: %d rows, %d exempt, %d unchecked, %d disagree\n", doc, rows, exempt, unchecked, bad
    exit bad ? 1 : 0 }
' "$dump" "$doc"
