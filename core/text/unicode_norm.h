/* unicode_norm.h -- NFC / NFD / NFKC / NFKD over codepoint vectors.
 *
 * The four forms are declared steps in a HuggingFace normaliser chain, so the tokeniser has to
 * be able to run them or a model that declares one gets silently different token boundaries --
 * the exact failure class spec §12 exists to remove. llama.cpp has no equivalent, so this is
 * ours; the tables it reads live in unicode_norm_data.cpp and say there how they were made.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rad {

/* One codepoint's decomposition. `off_*`/`len_*` index k_decomp_pool; a zero `len` means the
 * codepoint has no decomposition of that kind and stands for itself. */
struct DecompRow { uint32_t cpt, off_c, len_c, off_k, len_k; };
struct ComposeRow { uint32_t a, b, c; };
struct CccRow { uint32_t first, last; uint32_t ccc; };

extern const uint32_t   k_decomp_pool[];
extern const size_t     k_decomp_pool_n;
extern const DecompRow  k_decomp_rows[];
extern const size_t     k_decomp_rows_n;
extern const ComposeRow k_compose[];
extern const size_t     k_compose_n;
extern const CccRow     k_ccc[];
extern const size_t     k_ccc_n;

uint32_t unorm_ccc(uint32_t cpt);

/* In place, on codepoints. `compat` selects the K forms, `compose` selects the C forms, so the
 * four public names below are the four corners of that square. */
void unorm_decompose(std::vector<uint32_t>& cpts, bool compat);
void unorm_compose(std::vector<uint32_t>& cpts);

std::vector<uint32_t> unorm_nfd (const std::vector<uint32_t>& in);
std::vector<uint32_t> unorm_nfkd(const std::vector<uint32_t>& in);
std::vector<uint32_t> unorm_nfc (const std::vector<uint32_t>& in);
std::vector<uint32_t> unorm_nfkc(const std::vector<uint32_t>& in);

/* UTF-8 in, UTF-8 out. The tokeniser's step chain works on strings, so this is the form it
 * calls; the codepoint forms above are what the tests pin. */
std::string unorm_utf8(const std::string& in, bool compat, bool compose);

}  /* namespace rad */
