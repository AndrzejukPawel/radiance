#!/bin/sh
# Does this server actually do what it says about request keys, against one that is ALREADY UP?
#
# The rule the whole API rests on is that A REQUEST THAT RETURNS 200 MEANS EVERY FIELD OF IT WAS
# HONOURED: a parameter this engine does not implement is REFUSED, not quietly dropped. That rule
# is only worth anything if it is true of every key on the list, and the list lives in
# core/server/oai.cpp in three parts -- accepted, inert, rejected-with-a-reason. This checks the
# running server against it.
#
# THE TWO FAILURES THAT MATTER POINT IN OPPOSITE DIRECTIONS:
#
#   a REJECTED key that returns 200 is the rule broken -- the caller asked for a sampler the
#   device does not have and got a plausible answer computed without it;
#
#   an INERT key that returns 400 breaks the official SDKs, which send `user`, `store` and
#   friends unprompted and cannot be told not to.
#
# A rejected key is refused on its NAME, so the value it carries does not matter and every case
# below sends 1. An accepted key is not checked here: it would need a valid value per key, and a
# 400 would then be ambiguous between "refused the key" and "disliked the value".
#
#   P=8100 scripts/apikeys.sh
P="${P:-8100}"
H="${H:-127.0.0.1}"
U="http://$H:$P"
pass=0; fail=0

# code and blamed parameter for a chat request carrying one extra key
probe() {   # $1 = key, $2 = json value
    jq -n --arg k "$1" --argjson v "$2" \
        '{messages:[{role:"user",content:"hi"}],max_tokens:4,temperature:0} + {($k): $v}' \
        > "$WORK/q.json"
    code=$(curl -s -o "$WORK/r.json" -w '%{http_code}' -m 120 -X POST "$U/v1/chat/completions" \
           -H 'Content-Type: application/json' --data @"$WORK/q.json")
    param=$(jq -r '.error.param // ""' < "$WORK/r.json" 2>/dev/null)
    printf '%s %s' "$code" "$param"
}

want_refused() {  # $1 = key
    set -- "$1" $(probe "$1" 1)
    if [ "$2" = "400" ] && [ "$3" = "$1" ]; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1)); printf '  FAIL %-24s expected 400 naming it, got %s %s\n' "$1" "$2" "$3"
    fi
}

want_accepted() { # $1 = key, $2 = json value
    set -- "$1" $(probe "$1" "$2") 
    if [ "$2" = "200" ]; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1)); printf '  FAIL %-24s expected 200, got %s (param %s)\n' "$1" "$2" "$3"
    fi
}

WORK=$(mktemp -d "${TMPDIR:-/tmp}/apikeys.XXXXXX"); trap 'rm -rf "$WORK"' EXIT INT TERM
echo "checking $U against the key lists in core/server/oai.cpp"

echo "  rejected keys must be refused, by name:"
for k in logit_bias functions function_call best_of suffix \
         mirostat mirostat_tau mirostat_eta top_n_sigma \
         dynatemp_range dynatemp_exponent lora modalities audio prediction \
         continue_final_message cache_prompt; do
    want_refused "$k"
done

echo "  inert keys must be accepted and ignored:"
for k in user metadata store service_tier safety_identifier prompt_cache_key; do
    want_accepted "$k" '"x"'
done

# AN ACCEPTED KEY NEEDS A VALID VALUE, or a 400 is ambiguous between "refused the key" and
# "disliked the value" -- so each of these carries one the schema actually allows, and the ones
# that only make sense together are sent together.
merge() {   # $1 = label, $2 = json object to merge into a minimal chat request
    jq -n --argjson o "$2" \
        '{messages:[{role:"user",content:"hi"}],max_tokens:4,temperature:0} + $o' > "$WORK/q.json"
    code=$(curl -s -o "$WORK/r.json" -w '%{http_code}' -m 120 -X POST "$U/v1/chat/completions" \
           -H 'Content-Type: application/json' --data @"$WORK/q.json")
    if [ "$code" = "200" ]; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
        printf '  FAIL %-24s expected 200, got %s: %s\n' "$1" "$code" \
               "$(jq -r '.error.message // ""' < "$WORK/r.json" | head -c 90)"
    fi
}

echo "  accepted keys must be honoured, not refused:"
merge model                 '{"model":"m"}'
merge n                     '{"n":1}'
merge seed                  '{"seed":1}'
merge stop                  '{"stop":["\n\n"]}'
merge temperature           '{"temperature":0.7}'
merge top_p                 '{"top_p":0.9}'
merge top_k                 '{"top_k":20}'
merge min_p                 '{"min_p":0.05}'
merge typical_p             '{"typical_p":0.95}'
merge presence_penalty      '{"presence_penalty":0.1}'
merge frequency_penalty     '{"frequency_penalty":0.1}'
merge repetition_penalty    '{"repetition_penalty":1.05}'
merge repeat_last_n         '{"repeat_last_n":64}'
merge xtc                   '{"xtc_probability":0.1,"xtc_threshold":0.15}'
merge dry                   '{"dry_multiplier":0.5,"dry_base":1.75,"dry_allowed_length":2,"dry_penalty_last_n":64,"dry_sequence_breakers":["\n"]}'
merge priority              '{"priority":0}'
merge ignore_eos            '{"ignore_eos":false}'

merge response_format       '{"response_format":{"type":"text"}}'
merge tools                 '{"tools":[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object","properties":{}}}}],"tool_choice":"auto","parallel_tool_calls":true}'
merge add_generation_prompt '{"add_generation_prompt":true}'
merge max_completion_tokens '{"max_completion_tokens":4}'
merge chat_template_kwargs  '{"chat_template_kwargs":{"enable_thinking":false}}'
merge reasoning_effort      '{"reasoning_effort":"low"}'
merge enable_thinking       '{"enable_thinking":false}'
merge stream                '{"stream":true,"stream_options":{"include_usage":true}}'

# AND TWO ACCEPTED KEYS ARE REFUSED BY CAPABILITY RATHER THAN BY VETTING, which is the rule
# working and not a hole in it: being on the accepted list means the engine UNDERSTANDS the key,
# not that every deployment can satisfy it. What matters is that it refuses instead of answering
# unconstrained, so what is asserted here is the refusal, with the key named.
#
#   logprobs -- `supports_logprobs` is initialised false and set true nowhere in the tree, so this
#   engine never returns per-token probabilities. /server_info reports `logprobs: false`, which is
#   where a client should look before asking.
#
#   grammar and response_format ON CHAT -- both pass through the chat template, which may rewrite
#   or drop them. A template that returns no grammar for a request that asked for one leaves the
#   caller parsing free prose against a schema they were told was enforced, so it is a refusal.
#   On /v1/completions there is no template in the way and the same grammar works.
want_capability_refusal() {   # $1 = label, $2 = object, $3 = key that must be blamed
    jq -n --argjson o "$2" \
        '{messages:[{role:"user",content:"hi"}],max_tokens:4,temperature:0} + $o' > "$WORK/q.json"
    code=$(curl -s -o "$WORK/r.json" -w '%{http_code}' -m 120 -X POST "$U/v1/chat/completions" \
           -H 'Content-Type: application/json' --data @"$WORK/q.json")
    param=$(jq -r '.error.param // ""' < "$WORK/r.json" 2>/dev/null)
    if [ "$code" != "200" ] && [ "$param" = "$3" ]; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
        printf '  FAIL %-24s expected a refusal naming %s, got %s %s\n' "$1" "$3" "$code" "$param"
    fi
}

echo "  and two accepted keys this deployment cannot satisfy must REFUSE, not answer anyway:"
want_capability_refusal logprobs '{"logprobs":true,"top_logprobs":1}' logprobs
want_capability_refusal "grammar (chat)" '{"grammar":"root ::= \"ok\""}' grammar

echo "  and a key on no list at all must be refused, naming it:"
want_refused "no_such_parameter_at_all"

printf '\n  %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ] || exit 1
