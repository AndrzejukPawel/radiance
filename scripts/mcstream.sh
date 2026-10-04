#!/bin/sh
# Tool-call replies STREAMED, which is how an agent client talks to this server: content is sent
# as it grows and cannot be retracted, so a tool call repaired only at the end would leave the
# client showing the raw XML AND the call. Asserts on the RAW SSE bytes, not on the assembled
# message.
# The port is an override (P=, default 8100): this harness attaches to a server that is already
# up rather than starting one.
P=${P:-8100}
WORK=$(mktemp -d "${TMPDIR:-/tmp}/mcstream.XXXXXX"); trap 'rm -rf "$WORK"' EXIT INT TERM
T='[{"type":"function","function":{"name":"bash","description":"run a shell command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}},{"type":"function","function":{"name":"write","description":"write a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]'
one() { # $1 label  $2 prompt
  curl -s -N -m 240 http://127.0.0.1:$P/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":$2}],\"tools\":$T,\"stream\":true,\"max_tokens\":2500,\"temperature\":0}" > "$WORK/sse.txt"
  # every content fragment the client would render, concatenated
  jq -rj '.choices[0].delta.content // ""' < /dev/null 2>/dev/null
  CONTENT=$(grep '^data: {' "$WORK/sse.txt" | sed 's/^data: //' | jq -rj '.choices[0].delta.content // ""' 2>/dev/null)
  NCALLS=$(grep '^data: {' "$WORK/sse.txt" | sed 's/^data: //' | jq -r '[.choices[0].delta.tool_calls[]?.function.name//empty]|.[]' 2>/dev/null | grep -c . )
  FIN=$(grep '^data: {' "$WORK/sse.txt" | sed 's/^data: //' | jq -r '.choices[0].finish_reason//empty' 2>/dev/null | grep -v '^$' | tail -1)
  LEAK=no
  case "$CONTENT" in *"<function"*|*"<param"*|*"CDATA"*|*"<tool_call>"*) LEAK=YES ;; esac
  printf '%-16s finish=%-10s streamed-calls=%s  LEAK-IN-STREAM=%s\n' "$1" "$FIN" "$NCALLS" "$LEAK"
  [ "$LEAK" = YES ] && { echo "    content was: $(printf '%s' "$CONTENT" | head -c 160)"; }
  return 0
}
one "write-a-file"  '"Write the file /work/hello.c containing a hello world program in C. Use the write tool."'
one "run-a-command" '"List the files in /work using the bash tool."'
one "plain-chat"    '"In one sentence, what is a red-black tree?"'
