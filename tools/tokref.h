/* tokref.h -- a reference encoder for checking core/text's.
 *
 * `encode` runs the declared chain the way llama.cpp does: its structural splitters for the
 * patterns it recognises, std::regex over a category-collapsed copy of the text for the rest, and
 * a string-keyed bigram queue for the merge. It shares no code with core/text/tokenizer_regex.cpp
 * or tokenizer_bpe.cpp, which is the point: tokenizer_test and rad-tokcheck hold core/text's
 * encoder to it id for id.
 *
 * WHERE IT IS KNOWN TO DIFFER FROM HUGGINGFACE TOKENIZERS, which is the reference both are
 * ultimately judged by. core/text's encoder follows HuggingFace in each case, so these are the only
 * places the two encoders may disagree, and the tests that compare them name the case rather
 * than avoiding the input:
 *   - `(?i:'s|'t|'re|'ve|'m|'ll|'d)` is rewritten to `'[sS]|...` before matching, which drops
 *     Oniguruma's case orbit of 's': U+017F LATIN SMALL LETTER LONG S. "'ſ" is a contraction to
 *     HuggingFace and not here.
 *   - llama.cpp's category tables are Unicode 15.1; the Oniguruma in tokenizers is 16.0. The 5185
 *     codepoints assigned in 16.0 (Garay, Todhri, Tulu-Tigalari, ..., new CJK strokes and Latin
 *     letters) are unassigned here and categorised there.
 *   - `\p{Han}+` is llama.cpp's name for its Kimi-K2 splitter, which does not implement K2's
 *     upper/lower-case letter classes and uses its own Han ranges.
 *   - `\d{1,3}(?=(?:\d{3})*\b)` is split by llama.cpp's AFMoE digit rule, which also splits digit
 *     runs followed by letters; the pattern does not.
 *   - Split behaviours other than Isolated are applied as Isolated.
 *   - A regex Replace normaliser runs std::regex (ECMAScript, bytes) and expands `$` in its
 *     replacement; HuggingFace inserts the replacement verbatim.
 */
#pragma once
#include "text/tokenizer.h"

#include <string>
#include <string_view>
#include <vector>

namespace rad {
namespace tokref {

/* Same contract as Tokenizer::encode. */
int encode(const Vocab& v, std::string_view text, std::vector<int32_t>& out,
           bool add_special, bool parse_special);

/* One word through one Split pattern, Isolated, as the reference splits it. */
std::vector<std::string> split(const std::string& pattern, const std::string& word);

/* True when llama.cpp has a hand-written splitter for `pattern` (after the (?i) rewrite). */
bool structural(const std::string& pattern);

}  /* namespace tokref */
}  /* namespace rad */
