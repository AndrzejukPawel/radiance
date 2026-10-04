#!/bin/sh
# tierident.sh -- does a session that left VRAM come back BYTE FOR BYTE, both halves of it?
#
# This is the acceptance test for the idle session tiers (core/mem/kvtier.h). It is a byte
# comparison because CORRUPT KV DOES NOT READ AS BABBLE -- it reads as fluent, deterministic,
# coherent, plausible text that simply is not the text the model would have produced, and nothing
# that looks at the engine from outside notices it. Only a byte comparison against a known-good
# arm sees it.
#
# THE CONTROL IS THE WHOLE TEST. Arm 2 asks the same prompt again while the session is still in
# VRAM. If that does not reproduce arm 1 exactly, then this model at this temperature is not
# reproducible across requests at all and every later comparison is meaningless -- a differing sha
# would prove nothing, because a cache hit COULD have been legitimately tie-divergent. Run the
# control or do not run the test.
#
# ============================== AND ON A HYBRID MODEL, HALF THE TEST IS INVISIBLE ===============
#
# A session on a hybrid model is attention blocks AND a recurrent-state snapshot. The two are
# stored separately and can be lost separately -- and losing the snapshot IS NOT VISIBLE IN THE
# TEXT. The engine simply replays the linear layers from an older checkpoint, or from token zero,
# and produces exactly the same answer, slower. So byte identity proves the blocks came back and
# says nothing whatever about the state.
#
# That is why every snapshot assertion below is a counter and not a sha. A run where the blocks
# moved and the snapshots did not is a PASS by text and a FAILURE here, and it is the failure that
# matters: it is the difference between a restored 200K session costing a second and costing a
# minute of linear replay.
#
#   scripts/tierident.sh                  # against a server already up on $P (default 8100)
#
# The server must be started with the tiers on, with a host tier small enough for the filler below
# to overflow it (see fill_arena), and with few enough checkpoint slots that the filler's snapshots
# push the session's out of the device pool -- a snapshot still in its device slot needs no
# promotion, and the path that brings one back from disk then never runs:
#   --prefix-cache-host-mib 512 --prefix-cache-disk-mib 8192 --prefix-cache-dir <dir> \
#   --checkpoint-slots 2
set -e

P=${P:-8100}
H=${H:-127.0.0.1}
TO_HOST=${TO_HOST:-10}
TO_DISK=${TO_DISK:-180}
FILL=${FILL:-1200}
N=${N:-48}
work=$(mktemp -d "${TMPDIR:-/tmp}/tierident.XXXXXX")
trap 'rm -rf "$work"' EXIT INT TERM

command -v jq >/dev/null || { echo "tierident: jq is required" >&2; exit 2; }

# A PROMPT LONG ENOUGH TO BE WORTH TIERING, AND LONGER THAN A CHECKPOINT INTERVAL. The unit of the
# attention half is the block, so a prompt of a few tokens occupies one and exercises nothing. The
# unit of the linear half is the checkpoint interval -- 2048 tokens by default -- and a prompt
# shorter than that produces NO snapshot at all, so every snapshot assertion below would be
# vacuously satisfied by a server where snapshot tiering is entirely broken.
#
# AND IT HAS TO ASK FOR SOMETHING. A wall of repeated sentences with no question at the end is a
# passage an instruction-tuned model can answer with a single EOS, and every arm would then compare
# an empty string to an empty string and report the control green. That is the same failure this
# whole script is built around -- a comparison that passes because neither side has any content --
# so the answer length is checked below as well.
: > "$work/p.txt"
i=0
while [ $i -lt 300 ]; do
    printf 'Section %d. The cache stores conversation prefixes as content-hashed blocks, and the identity of a block includes every token before it. ' "$i" >> "$work/p.txt"
    i=$((i + 1))
done
printf '\nUsing only the passage above, write three sentences about what a block identity includes and why it matters.\n\nAnswer: ' >> "$work/p.txt"
# THE FILE IS THE PROMPT, READ VERBATIM, through jq's --rawfile. Escaping it by hand into a shell
# variable doubles the backslash jq adds before every quote, and flattening its newlines welds the
# question onto the passage ("...why it matters.Answer:" on one line), which this model answers
# with an immediate EOS -- the empty-completion failure described above.

ask() {                       # ask <label> -> writes $work/<label>.txt, echoes its sha
    jq -n --rawfile p "$work/p.txt" --argjson n "$N" \
        '{model:"m", prompt:$p, max_tokens:$n, temperature:0, seed:1234, stream:false}' \
        > "$work/req.json"
    curl -sS --fail -X POST "http://$H:$P/v1/completions" \
         -H 'Content-Type: application/json' --data @"$work/req.json" > "$work/$1.json"
    jq -r '.choices[0].text' < "$work/$1.json" > "$work/$1.txt"
    sha1sum < "$work/$1.txt" | cut -d' ' -f1
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
snap() { curl -sS --fail "http://$H:$P/sessions" > "$work/s.json"; }
q()    { jq -r "$1" < "$work/s.json"; }
# PROMPT TOKENS A LOOKUP COVERED WITH RECURRENT STATE. Moving a snapshot and moving it back proves
# the bytes travelled and nothing more; this is the number that says a hit then SERVED one. They
# are not the same claim -- a restore that puts the state back somewhere the chain cannot reach
# scores perfectly on the first and zero on this.
lin()  { jq -r ".linear.cached_tokens" < "$work/s.json"; }

where() {                     # where -> "vram=N ram=N disk=N [+ snapshots]" for the whole server
    snap
    w="vram=$(q .blocks_sum_overlapping.device) ram=$(q .blocks_sum_overlapping.host) disk=$(q .blocks_sum_overlapping.disk)"
    if [ "$(q .snapshots_moved.bytes_each)" != "0" ]; then
        w="$w  state: vram=$(q .snapshots.device) ram=$(q .snapshots.host) disk=$(q .snapshots.disk)"
    fi
    echo "$w"
}

fail=0
check() {                     # check <label> <sha> <expected>
    if [ "$2" = "$3" ]; then echo "  ok    $1"
    else echo "  FAIL  $1: $2 != $3"; fail=$((fail + 1)); fi
}
note() { echo "  FAIL  $1"; fail=$((fail + 1)); }

echo "tierident: $H:$P, ${TO_HOST}s for VRAM to go, $FILL filler sections to push host memory out"

# IS THERE A LINEAR HALF AT ALL. A pure-attention model has no snapshots and must not be failed
# for having none; a hybrid one with a snapshot store of zero slots is a real misconfiguration and
# is named as one rather than skipped.
snap
HYBRID=0
[ "$(q .snapshots_moved.bytes_each)" != "0" ] && HYBRID=1
if [ "$HYBRID" = 1 ]; then
    echo "  hybrid: snapshots are $(q .snapshots_moved.bytes_each) bytes, $(q .snapshots_moved.host_slots) ram slot(s), $(q .snapshots_moved.disk_slots) disk slot(s)"
    [ "$(q .snapshots_moved.host_slots)" = "0" ] && \
        note "the snapshot store has no room; an idle session's linear state cannot leave VRAM"
else
    echo "  this model has no recurrent state; the linear-state arms are not applicable"
fi

COLD=$(ask cold)
echo "  cold                       $COLD   [$(where)]"

# AN EMPTY ANSWER MAKES EVERY COMPARISON BELOW VACUOUS, and it is the easiest way for this script
# to report a confident green having tested nothing: three identical shas of the empty string.
CN=$(wc -c < "$work/cold.txt")
if [ "$CN" -lt 32 ]; then
    echo "tierident: the model answered $CN bytes. Every sha below would match for that reason" >&2
    echo "           alone. Give the prompt something to answer and run again." >&2
    exit 1
fi

# ---- the control -------------------------------------------------------------------------------
HOT=$(ask hot)
echo "  device hit                 $HOT   [$(where)]"
check "the same prompt twice from VRAM is byte-identical  (THE CONTROL)" "$HOT" "$COLD"
if [ "$HOT" != "$COLD" ]; then
    echo "tierident: the control failed, so nothing below would mean anything. Stopping." >&2
    echo "           This server is not reproducible across requests; fix that first." >&2
    exit 1
fi

# ---- host --------------------------------------------------------------------------------------
echo "  waiting $((TO_HOST + 8))s for the session to leave VRAM..."
sleep $((TO_HOST + 8))
W=$(where)
echo "  after the host threshold   [$W]"
# A RUN THAT MOVED NOTHING IS A FAILURE, NOT A PASS. This is the trap the whole script exists to
# avoid, turned on itself: if the session never left VRAM then every arm below is an ordinary
# device hit, the shas match for a reason that has nothing to do with tiering, and reporting PASS
# would be exactly the false confidence a byte comparison is supposed to replace.
snap
# A MISSING FIELD IS THE SAME FAILURE AS A ZERO: if the /sessions layout changes, this guard reads
# `null`, and treating that as a pass would pass vacuously.
case "$(q .blocks_sum_overlapping.host)" in 0|null|"") false ;; *) true ;; esac || note "the session never left VRAM -- nothing below would test anything
        (--prefix-cache-host-mib set? is the maintenance pass running?)"
if [ "$HYBRID" = 1 ]; then
    snap
    [ "$(q .snapshots_moved.to_host)" = "0" ] && \
        note "no linear-state snapshot reached RAM, so the restore below tests the blocks only
        (starved: $(q .snapshots_moved.starved), dropped: $(q .snapshots_moved.drops))"
fi
snap
PROM0=$([ "$HYBRID" = 1 ] && q .snapshots_moved.promoted || echo 0)
LIN0=$([ "$HYBRID" = 1 ] && lin || echo 0)
DEV0=$([ "$HYBRID" = 1 ] && q .snapshots.device || echo 0)
FROMHOST=$(ask fromhost)
echo "  restored from RAM          $FROMHOST   [$(where)]"
check "a session restored from RAM reproduces the cold answer" "$FROMHOST" "$COLD"
if [ "$HYBRID" = 1 ]; then
    snap
    # A SNAPSHOT STILL IN ITS DEVICE SLOT NEEDS NO PROMOTION: it stays there when the blocks go,
    # and only the checkpoint pool's own pressure detaches it. Whether it was SERVED is the check
    # below either way; this one only says which path served it.
    if [ "$(q .snapshots_moved.promoted)" -gt "$PROM0" ]; then
        echo "  ok    its linear state came back with it  ($(( $(q .snapshots_moved.promoted) - PROM0 )) snapshot(s))"
    elif [ "$DEV0" -gt 0 ]; then
        echo "  --    its linear state never left its device slot ($DEV0 there), so nothing was promoted"
    else
        note "the blocks came back and the linear state did not -- the answer is right and the
        session will replay every recurrent layer from an older checkpoint to produce it"
    fi
    if [ "$(lin)" -gt "$LIN0" ]; then
        echo "  ok    and the restored state was SERVED  ($(( $(lin) - LIN0 )) prompt tokens covered by it)"
    else
        note "the snapshot came back and no lookup used it -- restored somewhere the chain
        cannot reach, which costs the replay it was moved to avoid"
    fi
fi

# ---- disk --------------------------------------------------------------------------------------
echo "  filling host memory with $FILL sections of something else..."
fill_arena
wait_disk || true
echo "  and waiting $((TO_HOST + 8))s for VRAM to go..."
sleep $((TO_HOST + 8))
W=$(where)
echo "  after the filler           [$W]"
snap
case "$(q .blocks_sum_overlapping.disk)" in 0|null|"") false ;; *) true ;; esac || note "the session never reached disk -- the arm below tests nothing
        (--prefix-cache-disk-mib and --prefix-cache-dir set? a host tier FILL=$FILL sections overflow?)"
if [ "$HYBRID" = 1 ]; then
    snap
    [ "$(q .snapshots_moved.to_disk)" = "0" ] && \
        note "no linear-state snapshot reached disk
        (disk snapshot slots: $(q .snapshots_moved.disk_slots))"
fi
snap
PROM1=$([ "$HYBRID" = 1 ] && q .snapshots_moved.promoted || echo 0)
LIN1=$([ "$HYBRID" = 1 ] && lin || echo 0)
FROMDISK=$(ask fromdisk)
echo "  restored from disk         $FROMDISK   [$(where)]"
check "a session restored from disk reproduces the cold answer" "$FROMDISK" "$COLD"
if [ "$HYBRID" = 1 ]; then
    snap
    if [ "$(q .snapshots_moved.promoted)" -gt "$PROM1" ]; then
        echo "  ok    its linear state came back from disk too"
    else
        note "nothing promoted a snapshot out of disk -- the filler's snapshots did not push the
        session's out of the device pool (--checkpoint-slots 2?), so the disk path never ran"
    fi
    if [ "$(lin)" -gt "$LIN1" ]; then
        echo "  ok    and the state restored from disk was SERVED  ($(( $(lin) - LIN1 )) prompt tokens)"
    else
        note "a snapshot came back from disk and no lookup used it"
    fi
    echo "  snapshots: to_host=$(q .snapshots_moved.to_host) to_disk=$(q .snapshots_moved.to_disk) promoted=$(q .snapshots_moved.promoted) dropped=$(q .snapshots_moved.drops) starved=$(q .snapshots_moved.starved)"
fi

# ---- and the text itself, so a failure says WHERE ----------------------------------------------
if [ "$fail" -ne 0 ]; then
    echo
    echo "The first differing line, cold vs the failing arm:"
    for arm in fromhost fromdisk; do
        [ -f "$work/$arm.txt" ] || continue
        if ! cmp -s "$work/cold.txt" "$work/$arm.txt"; then
            echo "--- $arm"
            diff "$work/cold.txt" "$work/$arm.txt" | head -6
        fi
    done
    echo
    echo "A DIFFERENCE IN THE TEXT IS CORRUPT KV, not a sampling artefact: the control above proved"
    echo "this server reproduces itself from VRAM, so the only thing that changed is where the"
    echo "bytes came back from. A snapshot failure with matching text is not corruption -- it is"
    echo "the restore silently costing a linear replay it was built to avoid."
fi

echo
[ "$fail" -eq 0 ] && echo "tierident: PASS -- a session survives VRAM -> RAM -> disk byte for byte, state included"
[ "$fail" -ne 0 ] && { echo "tierident: $fail FAILURE(S)"; exit 1; }
exit 0
