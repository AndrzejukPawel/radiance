#!/bin/sh
# HOW OFTEN DOES THE DRAFTER ACTUALLY PROPOSE? draft_tokens_total/depth is the number of draft
# BLOCKS; over decode steps that is the fraction of steps that speculated at all. accept% is
# conditional on having drafted, so a high accept% with tok/step ~1.2 can only mean most steps
# never drafted. The server log says the drafter may DECLINE ("prompt lookup fills in").
P=8112; U="http://127.0.0.1:$P"
pad() { i=0; W=""; while [ $i -lt "$1" ]; do
  W="$W static int eval_piece_$1_$i(const Board *b, int sq) { return b->squares[sq] * 100 + $i; }"
  i=$((i+1)); done; printf '%s' "$W"; }
P=8112 TP=2 MLEN=131072 SEQS=8 sh "$(dirname "$0")/mcserve.sh" >/dev/null 2>&1 || { echo DIED; exit 1; }
g() { curl -s -m 10 "$U/metrics"|grep -F "$1"|grep -v '^#'|awk '{print $NF}'|head -1; }
printf '%8s %9s %9s %10s %10s %9s\n' ctx steps blocks blocks/step tok/step accept%
for N in 4 60 260 1040; do
  jq -n --arg c "$(pad $N)" '{model:"m",max_tokens:400,temperature:0,
    messages:[{role:"user",content:("Here is a file:\n\n"+$c+"\n\nSummarise what this code does in detail.")}]}' > /tmp/dc.json
  s0=$(g engine_steps_decode_total); d0=$(g decode_tokens_total); r0=$(g draft_tokens_total); a0=$(g draft_accepted_total)
  curl -s -m 900 -o /tmp/dc.out "$U/v1/chat/completions" -H 'Content-Type: application/json' -d @/tmp/dc.json
  s1=$(g engine_steps_decode_total); d1=$(g decode_tokens_total); r1=$(g draft_tokens_total); a1=$(g draft_accepted_total)
  awk -v ctx="$(jq -r '.usage.prompt_tokens // 0' /tmp/dc.out)" -v ds=$((s1-s0)) -v dd=$((d1-d0)) \
      -v dr=$((r1-r0)) -v da=$((a1-a0)) 'BEGIN{
    if(ds<=0){print "  NO STEPS"; exit}
    b=dr/7
    printf "%8d %9d %9d %10.3f %10.3f %9.1f\n", ctx, ds, b, b/ds, dd/ds, (dr>0?100.0*da/dr:0) }'
done
pkill -f "bin/radianc[e]"
