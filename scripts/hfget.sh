#!/bin/sh
# hfget.sh -- fetch a Hugging Face repo with curl and jq, and nothing else.
#
# The engine and its tooling do not depend on Python, so `huggingface-cli` is not assumed. The API
# gives a file listing as JSON and every file has a stable resolve/ URL, which is all a downloader
# needs.
#
#   scripts/hfget.sh <repo-id> [dest-dir] [revision]
#
# Set HF_TOKEN for a gated or private repo. Re-running resumes: a file whose local size already
# matches the listing is skipped and a short one is continued with `curl -C -`, so an interrupted
# transfer of a 300 GB checkpoint costs only what it had not finished.
#
# WEIGHT DUPLICATES ARE SKIPPED. Many repos carry the same tensors several times over -- safetensors
# beside the older pickle .bin, an ONNX export, a GGUF conversion -- and on a large checkpoint each
# copy is hundreds of gigabytes of the same numbers. When the listing holds any .safetensors, every
# other weight format is left behind.
#
# HFGET_JOBS concurrent transfers, default 4. These are long sequential reads: a few streams
# saturate the link and more only makes the destination seek.
set -e

# ---- child mode: one file, named by "<size><tab><path>" in $1, everything else inherited -------
# sh cannot export a function, so parallelism is one re-entry of this script per file under
# `xargs -P` rather than a worker pool.
if [ -n "$HFGET_CHILD" ]; then
    size=${1%%	*}; path=${1#*	}
    out=$HFGET_DEST/$path
    if [ -f "$out" ]; then
        cur=$(wc -c < "$out")
        if [ "$cur" = "$size" ]; then echo "  = $path"; exit 0; fi
        echo "  + $path (resuming at $cur of $size)"
    else
        echo "  > $path"
    fi
    mkdir -p "$(dirname "$out")"
    if [ -n "$HF_TOKEN" ]; then
        curl -sSL --fail --retry 5 --retry-delay 5 --retry-all-errors -C - \
             -H "Authorization: Bearer $HF_TOKEN" \
             -o "$out" "https://huggingface.co/$HFGET_REPO/resolve/$HFGET_REV/$path"
    else
        curl -sSL --fail --retry 5 --retry-delay 5 --retry-all-errors -C - \
             -o "$out" "https://huggingface.co/$HFGET_REPO/resolve/$HFGET_REV/$path"
    fi
    got=$(wc -c < "$out")
    [ "$got" = "$size" ] || { echo "hfget: $path is $got bytes, listing says $size" >&2; exit 1; }
    exit 0
fi

# ---- parent ------------------------------------------------------------------------------------
repo=$1
[ -n "$repo" ] || { echo "usage: hfget.sh <repo-id> [dest-dir] [revision]" >&2; exit 2; }
dest=${2:-$HOME/models/$repo}
rev=${3:-main}
jobs=${HFGET_JOBS:-4}
command -v jq >/dev/null || { echo "hfget: jq is required" >&2; exit 2; }

echo "hfget: listing $repo@$rev"
# THE LISTING IS PAGINATED AND THE PAGE SIZE IS SMALL. A large checkpoint runs to a hundred-odd
# shards and the API returns fifty entries with a `Link: ...; rel="next"` header. Taking only the
# first page is the worst possible failure here: every file it did fetch is the right length, so
# the size check passes and the converter is handed a checkpoint missing two thirds of its
# tensors. Follow the cursor until there is no next link.
api="https://huggingface.co/api/models/$repo/tree/$rev?recursive=true&expand=true"
listing='[]'
page=0
while [ -n "$api" ]; do
    hdr=$(mktemp "${TMPDIR:-/tmp}/hfhdr.XXXXXX")
    if [ -n "$HF_TOKEN" ]; then
        body=$(curl -sSL --fail-with-body -D "$hdr" -H "Authorization: Bearer $HF_TOKEN" "$api")
    else
        body=$(curl -sSL --fail-with-body -D "$hdr" "$api")
    fi
    [ "$(printf '%s' "$body" | jq -r 'type')" = "array" ] || {
        rm -f "$hdr"
        echo "hfget: $(printf '%s' "$body" | jq -r '.error // .message // .')" >&2
        echo "hfget: a gated or private repo needs HF_TOKEN in the environment" >&2; exit 1; }
    listing=$(printf '%s\n%s' "$listing" "$body" | jq -s 'add')
    api=$(tr -d '\r' < "$hdr" | sed -n 's/^[Ll]ink: *<\([^>]*\)>; *rel="next".*/\1/p' | head -1)
    rm -f "$hdr"
    page=$((page + 1))
    [ "$page" -gt 200 ] && { echo "hfget: listing did not terminate after 200 pages" >&2; exit 1; }
done
echo "hfget: $page page(s) of listing"
have_st=$(printf '%s' "$listing" | jq '[.[] | select(.path | endswith(".safetensors"))] | length')
sel='.[] | select(.type == "file")'
[ "$have_st" -gt 0 ] && sel="$sel"' | select((.path | test("\\.(bin|pth|msgpack|h5|onnx|onnx_data|gguf)$")) | not)'

list=$(mktemp "${TMPDIR:-/tmp}/hfget.XXXXXX")
trap 'rm -f "$list"' EXIT INT TERM
printf '%s' "$listing" | jq -r "$sel"' | "\(.size)\t\(.path)"' > "$list"

n=$(wc -l < "$list")
[ "$n" -gt 0 ] || { echo "hfget: the listing has no files" >&2; exit 1; }
need=$(awk -F'\t' '{s += $1} END {printf "%.1f", s / 1073741824}' "$list")
mkdir -p "$dest"
free=$(df -BG --output=avail "$dest" | tail -1 | tr -dc '0-9')
echo "hfget: $n file(s), $need GiB -> $dest"
[ "$have_st" -gt 0 ] && echo "hfget: safetensors present, pickle weight formats skipped"
echo "hfget: $free GiB free on the destination filesystem"
awk -v n="$need" -v f="$free" 'BEGIN { if (n > f) { print "hfget: THE DOWNLOAD DOES NOT FIT"; exit 1 } }'

export HFGET_CHILD=1 HFGET_DEST="$dest" HFGET_REPO="$repo" HFGET_REV="$rev"
self=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
tr '\n' '\0' < "$list" | xargs -0 -P "$jobs" -I{} sh "$self" "{}" \
    || { echo "hfget: incomplete -- re-run to resume" >&2; exit 1; }
echo "hfget: complete -- $(du -sh "$dest" | cut -f1) in $dest"
