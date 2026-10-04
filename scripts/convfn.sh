#!/bin/sh
# Rebuild the Qwen3.8-Flash-Next container `fnserve.sh` serves. THE CONVERSION OF RECORD.
#
# The recipe is the whole format decision: it names what each weight is quantised to, and the
# container records it (rad-info --recipe), so the engine serves the file in the format it was
# packed in and nothing else. The default, data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe, builds the
# production container w4nl64-i8lm.rad; its header says what each weight becomes and what that
# measures.
#
# CALIB=<dir> IS REQUIRED BY THAT RECIPE. Its routed-expert rule runs GPTQ against `calib=$CALIB`,
# and the converter refuses a recipe that names an unset variable. The directory holds each
# layer's pooled Grams and each expert's own down Gram, recorded from the bf16 model over THIS
# checkpoint (docs/MOE-W4.md, "Calibration"): Grams taken from another checkpoint are not
# transferable, and a layer with no file is packed without error feedback.
#
# A recipe whose rules do not name $CALIB (RECIPE=data/recipes/qwen4exp-w4.recipe, say) rounds its
# experts to nearest with absmax scales; CALIB=<dir> then puts them through int4 GPTQ instead, by
# the --quant rule that recipe's header names, which goes ahead of the recipe's own.
#
# OUT is removed before the conversion starts, so its default is a .new.rad beside the served
# container rather than the served container itself.
#
#   CALIB=<dir> [SRC=<checkpoint dir>] [OUT=<container>] [RECIPE=<file>] scripts/convfn.sh
set -e
cd "$HOME/workspace/radiance/build" || exit 1
SRC=${SRC:-$HOME/models/Qwen/Qwen3.8-Flash-Next}
OUT=${OUT:-$HOME/models/rad/w4nl64-i8lm.new.rad}
RECIPE=${RECIPE:-$HOME/workspace/radiance/data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe}
[ -d "$SRC" ] || { echo "no checkpoint at $SRC"; exit 1; }
[ -f "$SRC/tokenizer.json" ] || { echo "no tokenizer.json in $SRC"; exit 1; }
[ -f "$RECIPE" ] || { echo "no recipe at $RECIPE"; exit 1; }
[ -z "$CALIB" ] || [ -d "$CALIB" ] || { echo "no calibration directory at $CALIB"; exit 1; }
if grep -q '^[^#]*\$CALIB' "$RECIPE"; then
    [ -n "$CALIB" ] || { echo "$RECIPE quantises against \$CALIB: set CALIB=<calibration dir>"; exit 1; }
    export CALIB
    set --
elif [ -n "$CALIB" ]; then
    export CALIB
    set -- --quant 'blk.*.ffn_*_exps.*.weight=gptq:codes=i4,group=128,scale=bf16,transform=fwht128,calib=$CALIB'
else
    set --
fi
rm -f "$OUT"
./bin/rad-convert "$SRC" -o "$OUT" \
    --home "$HOME/workspace/radiance/build/radiance_home" \
    --tokenizer "$SRC/tokenizer.json" \
    --recipe "$RECIPE" "$@" \
    -v 2>&1 | tail -30
[ -s "$OUT" ] || { echo CONVERT_FAILED; exit 1; }
echo CONVERT_OK "$OUT"
