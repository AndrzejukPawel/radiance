#!/bin/sh
# tiercross.sh -- two conversations through the tiers at once. Does either get the other's state?
#
# WHY THIS IS NOT tierident.sh WITH A SECOND PROMPT. That script asks ONE prompt four times and
# compares the four answers, which proves a session survives the round trip. It cannot, by
# construction, see the failure this one is for: with a single conversation in the server there is
# no other conversation's bytes for it to be served. The cross-contamination failures a tiered
# cache can have -- an entry holding reassigned block ids, a checkpoint slot freed by one session
# and still named by another -- need two sessions in flight to show up at all.
#
# THE SCARCE RESOURCE IS THE CHECKPOINT SLOT, so run this with --checkpoint-slots low enough that
# the two sessions fight over it. That is what makes the retention policy and the victim scan run,
# which is what makes a snapshot get retired underneath a record that still names its slot:
#
#   --checkpoint-slots 2 --prefix-cache-host-mib 512 --prefix-cache-disk-mib 16384 \
#   --prefix-cache-dir <dir>
#
# EACH SESSION IS COMPARED ONLY TO ITSELF. A and B are different prompts and their answers differ
# for ordinary reasons; what must hold is that A's restored answer is A's cold answer. Corrupt
# state does not produce babble, it produces B's train of thought under A's question.
set -e

P=${P:-8100}
H=${H:-127.0.0.1}
TO_HOST=${TO_HOST:-10}
TO_DISK=${TO_DISK:-180}
FILL=${FILL:-1200}
N=${N:-48}
work=$(mktemp -d "${TMPDIR:-/tmp}/tiercross.XXXXXX")
trap 'rm -rf "$work"' EXIT INT TERM

command -v jq >/dev/null || { echo "tiercross: jq is required" >&2; exit 2; }

# TWO PROMPTS THAT SHARE NO PREFIX. A shared opening would put both sessions on the same chain for
# its length, and the blocks they then share are legitimately one another's -- which is the whole
# point of a prefix cache and would make a difference here unreadable. They diverge at token one.
#
# EACH HAS TO PRODUCE AN ANSWER. A wall of repeated sentences is a passage an instruction-tuned
# model can reasonably answer with a single EOS, and every arm of that session would then compare
# an empty string to an empty string. Both lengths are checked below, because a vacuous green here
# is worse than a red.
mk() {                        # mk <file> <subject> <opening> <question>
    : > "$1"
    i=0
    while [ $i -lt 300 ]; do
        printf '%s %d. %s ' "$2" "$i" "$3" >> "$1"
        i=$((i + 1))
    done
    printf '\n%s\n\nAnswer: ' "$4" >> "$1"
}
mk "$work/a.txt" "Chapter" "The paged pool is layer-major, so one logical block is many small fragments rather than one extent." \
   "Using only the passage above, write three sentences about it."
mk "$work/b.txt" "Entry"   "The linear layer keeps one state per sequence and updates it token by token." \
   "Using only the passage above, write three sentences about it."
# THE FILES ARE THE PROMPTS, READ VERBATIM, through jq's --rawfile. Flattening them into shell
# variables deletes the line breaks the prompts' SHAPE depends on, and a passage whose question
# ends up welded to the last sentence is one this model answers with an immediate EOS (the same
# reasoning is in tierident.sh).

ask() {                       # ask <who> <label> -> sha of the answer
    f="$work/$(printf '%s' "$1" | tr 'AB' 'ab').txt"
    jq -n --rawfile p "$f" --argjson n "$N" \
        '{model:"m", prompt:$p, max_tokens:$n, temperature:0, seed:1234, stream:false}' \
        > "$work/req.json"
    curl -sS --fail -X POST "http://$H:$P/v1/completions" \
         -H 'Content-Type: application/json' --data @"$work/req.json" > "$work/$2.json"
    jq -r '.choices[0].text' < "$work/$2.json" > "$work/$2.txt"
    sha1sum < "$work/$2.txt" | cut -d' ' -f1
}

# THE HIT LENGTH, which is what decides how the prefill is chunked and therefore the shape every
# GEMM in it runs at. Two restores that recovered different amounts of prefix are not the same
# computation, and comparing their output hashes would be comparing two different runs -- so this
# is printed beside every arm and checked before a mismatch is called corruption.
hit() { jq -r '.usage.prompt_tokens_details.cached_tokens' < "$work/$1.json"; }

fail=0
# WHAT A DIFFERING HASH MEANS, AND THE ONE THING THAT HAS TO BE RULED OUT FIRST.
#
# A hybrid session's reuse is clamped by the scheduler to the last LINEAR CHECKPOINT it can reach,
# not to the attention hit (core/sched/scheduler.cpp, admit_one: RadBatch carries one query length
# for every KV group, so the linear layers cannot be replayed over a longer range than the
# attention ones). So reuse is always a multiple of the checkpoint interval, and which multiple
# depends on which snapshots survived -- 0 on a cold run, 4096 or 6144 on a restored one.
#
# Reuse does not change the answer on Qwen3.8-Flash-Next: with no tiers running at all, the same
# prompt at reuse 0, 4096 and 6144 gives the same bytes. So a differing hash is a real difference
# and not an artefact of a shorter hit -- but it is the first thing to re-check if this fails on
# another model, because a near-tie that flips with the chunk boundary would look exactly like
# corruption and is not.
#
# THE COMPARISON IS AGAINST THE DEVICE-HIT ARM, not the cold one. A cold run reuses nothing by
# definition, so comparing a restore to it would always be comparing two different hit lengths;
# the device hit is the same query the restored arms run, served out of VRAM.
cmp_arm() {                   # cmp_arm <label> <sha> <ref sha> <reuse> <ref reuse>
    if [ "$2" = "$3" ]; then
        echo "  ok    $1 (reuse $4)"
        return
    fi
    fail=$((fail + 1))
    if [ "$4" = "$5" ]; then
        echo "  FAIL  $1: SAME reuse ($4), different answer. This is corrupt state."
    else
        echo "  FAIL  $1: different answer, and it reused $4 where the reference reused $5."
        echo "        Almost certainly corruption -- reuse does not change the answer on the model"
        echo "        this was written against -- but confirm that on THIS model with the tiers"
        echo "        off before reporting it as one."
    fi
}
# ---- pushing a session out of host memory --------------------------------------------------------
# NOTHING LEAVES HOST MEMORY ON A CLOCK. Every host copy is written on to disk as soon as it lands
# and keeps its host slot until another entry needs one, so the way to put a session on disk is to
# give the arena something else to hold: a prompt that shares no prefix with anything here and is
# bigger than the room the arena has left. The server must be started with a host tier FILL
# sections of it overflow; --prefix-cache-host-mib 512 holds about 27K tokens on
# Qwen3.8-Flash-Next, and the default FILL is about 30K.
fill_arena() {
    : > "$work/fill.txt"
    i=0
    while [ $i -lt "$FILL" ]; do
        printf 'Ledger line %d: warehouse north, shelf eleven, a part number and a count that nobody will ask about. ' "$i" >> "$work/fill.txt"
        i=$((i + 1))
    done
    printf '\nName the warehouse.\n\nAnswer: ' >> "$work/fill.txt"
    jq -n --rawfile p "$work/fill.txt" '{model:"m", prompt:$p, max_tokens:4, temperature:0, stream:false}' \
        > "$work/fill.json"
    curl -sS --fail -X POST "http://$H:$P/v1/completions" -H 'Content-Type: application/json' \
         --data @"$work/fill.json" > /dev/null
}
# Until some blocks are on the disk tier, or TO_DISK seconds. The filler's copies are what take the
# slots, and they land after its request finishes.
wait_disk() {
    t=0
    while [ $t -lt "$TO_DISK" ]; do
        curl -sS --fail "http://$H:$P/sessions" > "$work/w.json"
        case "$(jq -r .blocks_sum_overlapping.disk < "$work/w.json")" in 0|null|"") ;; *) return 0 ;; esac
        sleep 2; t=$((t + 2))
    done
    return 1
}
echo "tiercross: $H:$P, two sessions, ${TO_HOST}s for VRAM to go, $FILL filler sections"
CKS=$(curl -sS --fail "http://$H:$P/sessions" | jq -r .snapshots_moved.bytes_each)
[ "$CKS" = "0" ] && echo "  (no recurrent state in this model; the attention half is still tested)"

A0=$(ask A a0); B0=$(ask B b0)
echo "  first     A $A0  reuse $(hit a0)"
echo "            B $B0  reuse $(hit b0)"
# AND THEY MUST DIFFER. Two prompts that produce the same answer would make every comparison
# below pass whatever the tiers did with them.
if [ "$A0" = "$B0" ]; then
    echo "tiercross: both prompts produced the same answer, so no comparison here can tell the" >&2
    echo "           two sessions apart. Change the prompts." >&2
    exit 1
fi
for f in a0 b0; do
    CN=$(wc -c < "$work/$f.txt")
    [ "$CN" -lt 32 ] && { echo "tiercross: $f answered $CN bytes; every sha for it would match for that" >&2
                            echo "           reason alone. Give that prompt something to answer." >&2; exit 1; }
done

# ---- the control: both sessions again, straight out of VRAM -------------------------------------
# THE SAME QUERY THE RESTORED ARMS WILL RUN. Without it a differing hash below proves nothing: this
# engine could be irreproducible across requests for reasons that have nothing to do with tiering,
# and the two sessions could be contaminating each other while both are still resident.
AH=$(ask A ah); BH=$(ask B bh)
echo "  device    A $AH  reuse $(hit ah)"
echo "            B $BH  reuse $(hit bh)"
if [ "$AH" != "$A0" ] || [ "$BH" != "$B0" ]; then
    echo "  note: a device hit does not reproduce the cold answer. That is not necessarily wrong --"
    echo "        the hit reuses $(hit ah)/$(hit bh) tokens where the cold run reused nothing -- but every"
    echo "        comparison below is against the DEVICE HIT from here on, which is the like-for-like one."
fi

echo "  waiting $((TO_HOST + 8))s..."
sleep $((TO_HOST + 8))
A1=$(ask A a1); B1=$(ask B b1)
echo "  from RAM  A $A1  reuse $(hit a1)"
echo "            B $B1  reuse $(hit b1)"
cmp_arm "A restored from RAM is still A" "$A1" "$AH" "$(hit a1)" "$(hit ah)"
cmp_arm "B restored from RAM is still B" "$B1" "$BH" "$(hit b1)" "$(hit bh)"

echo "  filling host memory with $FILL sections of something else..."
fill_arena
wait_disk || echo "  note: nothing reached the disk tier within ${TO_DISK}s; the arm below may be a RAM restore"
sleep $((TO_HOST + 8))
A2=$(ask A a2); B2=$(ask B b2)
echo "  from disk A $A2  reuse $(hit a2)"
echo "            B $B2  reuse $(hit b2)"
cmp_arm "A restored from disk is still A" "$A2" "$AH" "$(hit a2)" "$(hit ah)"
cmp_arm "B restored from disk is still B" "$B2" "$BH" "$(hit b2)" "$(hit bh)"

curl -sS --fail "http://$H:$P/sessions" > "$work/s.json"
echo "  moved: blocks to_host=$(jq -r .moved.to_host < "$work/s.json") to_disk=$(jq -r .moved.to_disk < "$work/s.json") promoted=$(jq -r .moved.promoted < "$work/s.json")"
if [ "$CKS" != "0" ]; then
    echo "  state: to_host=$(jq -r .snapshots_moved.to_host < "$work/s.json") to_disk=$(jq -r .snapshots_moved.to_disk < "$work/s.json") promoted=$(jq -r .snapshots_moved.promoted < "$work/s.json") dropped=$(jq -r .snapshots_moved.drops < "$work/s.json")"
    # A RUN WITH NO CONTENTION TESTED THE EASY CASE. Two stores have to run out for the paths this
    # script exists for to execute, and they are sized by different flags: --checkpoint-slots is
    # the DEVICE pool the retention policy spends, and a share of --prefix-cache-host-mib (startup
    # says how many snapshots) is the snapshot store the tiers spend. `drops` counts the second.
    [ "$(jq -r .snapshots_moved.drops < "$work/s.json")" = "0" ] && \
        echo "  note: the snapshot store never overflowed -- lower --prefix-cache-host-mib to make the two sessions contend for it"
fi

if [ "$fail" -ne 0 ]; then
    echo
    echo "A session answering with the OTHER session's train of thought is what corruption looks"
    echo "like. A reuse mismatch above is a different finding and the diffs below do not speak to it."
    for pair in "ah a1" "ah a2" "bh b1" "bh b2"; do
        set -- $pair
        cmp -s "$work/$1.txt" "$work/$2.txt" || { echo "--- $1 vs $2"; diff "$work/$1.txt" "$work/$2.txt" | head -4; }
    done
fi

echo
[ "$fail" -eq 0 ] && echo "tiercross: PASS -- neither session was served the other's context or state"
[ "$fail" -ne 0 ] && { echo "tiercross: $fail FAILURE(S)"; exit 1; }
exit 0
