/* tokenizer_nfc.h -- NFC for the tokeniser's normaliser chain, applied only where it can change
 * something. Byte for byte what unorm_utf8(text, false, true) returns; see tokenizer_nfc.cpp for
 * why normalising the stretches around the codepoints that are not NFC boundaries is the same as
 * normalising the whole text. */
#pragma once
#include <string>

namespace rad {

/* Normalise `text` in place. False when it was already NFC and has not been touched. */
bool tok_nfc(std::string& text);

}  /* namespace rad */
