#!/bin/sh
# Rebuild the DSpark container for MiniCPM5-2B that scripts/mcserve.sh serves
# (minicpm5-2b-dspark.v2.rad): the bf16 checkpoint with the drafter merged in, every linear made
# block fp8 by data/recipes/minicpm5-2b-dspark.recipe, served by arch/llama_dense_fp8. It writes
# a .new.rad beside it, because the output is removed before the conversion starts.
cd "$HOME/workspace/radiance/build" || exit 1
RECIPE=${RECIPE:-$HOME/workspace/radiance/data/recipes/minicpm5-2b-dspark.recipe}
SRC=$HOME/models/openbmb/MiniCPM5-2B
DRAFT=$HOME/models/openbmb/MiniCPM5-2B-DSpark
OUT=$HOME/models/rad/minicpm5-2b-dspark.new.rad
[ -d "$DRAFT" ] || { echo "no drafter checkpoint at $DRAFT"; exit 1; }
rm -f "$OUT"
./bin/rad-convert "$SRC" -o "$OUT" \
    --home "$HOME/workspace/radiance/build/radiance_home" \
    --tokenizer "$SRC/tokenizer.json" \
    --draft-model "$DRAFT" \
    --recipe "$RECIPE" \
    -v 2>&1 | tail -25
[ -s "$OUT" ] || { echo CONVERT_FAILED; exit 1; }
echo CONVERT_OK "$OUT"
