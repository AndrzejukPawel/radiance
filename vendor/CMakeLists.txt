# vendor/ -- llama.cpp lifts and third-party headers, tracked to an upstream commit and patched
# rather than edited (spec §12). See UPSTREAM for what came from where and patches/ for every
# local change. Nothing here is written by us; anything that had to be rewritten lives in core/.
#
# The public target is `rad_vendor`, an INTERFACE library. core/CMakeLists.txt links it if it
# exists, so this file is the only place that decides what the rest of the tree can include.

# ---------------------------------------------------------------- compiled lifts
# Vendored sources are compiled with warnings off. Not because the warnings are wrong, but
# because a warning we will never fix is noise in front of the warnings we will, and editing
# upstream to silence one is exactly what the patch-series rule exists to prevent.
add_library(rad_vendor_llama STATIC
  llama/src/unicode.cpp
  llama/src/unicode-data.cpp
  llama/common/unicode.cpp
  llama/common/jinja/lexer.cpp
  llama/common/jinja/parser.cpp
  llama/common/jinja/runtime.cpp
  llama/common/jinja/string.cpp
  llama/common/jinja/value.cpp
  llama/common/jinja/caps.cpp
  llama/common/chat.cpp
  llama/common/chat-peg-parser.cpp
  llama/common/chat-auto-parser-generator.cpp
  llama/common/chat-auto-parser-helpers.cpp
  llama/common/chat-diff-analyzer.cpp
  llama/common/peg-parser.cpp
  llama/common/json-schema-to-grammar.cpp
  llama/common/json-partial.cpp
  llama/common/regex-partial.cpp
  shim/llama_shim.cpp)
target_compile_options(rad_vendor_llama PRIVATE -w)
# SYSTEM, so a warning in a vendored header is not a warning in OUR build. `-w` above covers the
# vendor's own translation units; these headers are included by ours too, and two -Wunused-function
# from jinja were the only things left in an otherwise clean build. A warning nobody intends to act
# on is a warning that stops the rest from being read.
target_include_directories(rad_vendor_llama SYSTEM PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  # shim/ first: the lifted files say #include "common.h" / "log.h" / "ggml.h" / "llama.h" and
  # every one of those is ours (see vendor/shim/*.h). Putting them on the path rather than
  # editing the include line is what keeps the lifts byte-identical to upstream.
  ${CMAKE_CURRENT_SOURCE_DIR}/shim
  # And llama/common, because the lifted files include each other and the jinja engine by their
  # own basenames. The cost of this one, named: `#include "unicode.h"` anywhere in core/ now
  # resolves to llama/common/unicode.h (the streaming UTF-8 helper) and NOT to
  # llama/src/unicode.h (the codepoint tables). core/ spells both in full from vendor/ for
  # exactly that reason, and the tokeniser's include is "llama/src/unicode.h".
  ${CMAKE_CURRENT_SOURCE_DIR}/llama/common)

# cpp-httplib 0.46.1 is a split header/source library, not header-only, so it is compiled.
# TLS is deliberately off: the server serves plain HTTP and terminating TLS is the job of
# whatever sits in front of it. Turning it on is a target_compile_definitions line here plus a
# link against OpenSSL, and is the server component's call, not this file's.
add_library(rad_vendor_httplib STATIC cpp-httplib/httplib.cpp)
target_compile_options(rad_vendor_httplib PRIVATE -w)
target_include_directories(rad_vendor_httplib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp-httplib)
target_link_libraries(rad_vendor_httplib PRIVATE Threads::Threads)
target_compile_definitions(rad_vendor_httplib PUBLIC
  # llama.cpp's operational settings, carried because the reasons are ours too: a long prompt in
  # a form body, many concurrent connections, a long prompt in a query string, and no Nagle delay
  # on a streamed token.
  CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=1048576
  CPPHTTPLIB_LISTEN_BACKLOG=512
  CPPHTTPLIB_REQUEST_URI_MAX_LENGTH=32768
  CPPHTTPLIB_TCP_NODELAY=1)

# ---------------------------------------------------------------- the public face
# Header-only vendors need no compilation, only a path:
#   <nlohmann/json.hpp>          vendor/nlohmann/json.hpp
#   <cpp-httplib/httplib.h>      vendor/cpp-httplib/httplib.h   (or plain "httplib.h")
#   "llama/src/unicode.h"        the tokeniser's codepoint tables
#   "llama/common/jinja/*.h"     the jinja engine
add_library(rad_vendor INTERFACE)
target_include_directories(rad_vendor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rad_vendor INTERFACE rad_vendor_llama rad_vendor_httplib)
