/* word_scan.h -- find the first of a set of literal words in a byte stream that arrives in pieces.
 *
 * A lazy grammar waits for its trigger word -- `<tool_call>\n` on the Qwen templates -- and until
 * the word appears every generated token passes through here. The stream is the whole reply, so
 * anything that looks back over what has already arrived turns a long reply into quadratic host
 * work on the step path. This looks back at nothing: the scan state is one integer, each byte is
 * one table load, and what a match needs to be replayed is bounded by the longest word.
 *
 * THE AUTOMATON is Aho-Corasick compiled to a full DFA over byte classes. The trie of the words
 * gets failure links in breadth-first order, and every missing edge is resolved through them once,
 * at build time, so the scan never follows a failure chain. Bytes that occur in no word share one
 * class, which keeps the table at (states x distinct word bytes + 1) entries -- a few hundred for
 * any real trigger set, so it stays in L1 however long the reply runs.
 *
 * THE MATCH REPORTED is the leftmost START among the occurrences that complete inside one fed
 * piece. That is the answer a search of the whole stream so far gives after each piece: nothing
 * completed in an earlier piece (the caller would have stopped), so every complete occurrence ends
 * in this one, and of those the leftmost is the one that begins first. Each state carries the
 * length of the LONGEST word ending there, because the longest word ending at a byte is the one
 * that starts earliest.
 */
#pragma once
#include "../rad_internal.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rad {

class WordScanner {
public:
    static constexpr uint64_t npos = ~(uint64_t)0;

    /* Where a stream is: the automaton state and how many bytes have been fed. Plain data, so a
     * speculative window can copy it and put it back. */
    struct State {
        uint32_t s   = 0;
        uint64_t off = 0;
    };

    /* Compile `words`. Every word must be non-empty; RAD_E_INVAL names the first that is not.
     * An empty set builds a scanner that never matches. */
    int build(const std::vector<std::string>& words, std::string* err);

    bool     empty()   const { return max_len_ == 0; }
    uint32_t max_len() const { return max_len_; }

    /* Advance over `n` bytes. Returns the stream offset at which the leftmost occurrence that
     * completes within these bytes begins, or npos when none completes. */
    uint64_t feed(State& st, const char* p, size_t n) const {
        uint64_t best = npos;
        uint32_t s    = st.s;
        uint64_t off  = st.off;
        for (size_t i = 0; i < n; ++i) {
            s = next_[(size_t)s * ncls_ + cls_[(uint8_t)p[i]]];
            ++off;
            const uint32_t len = out_len_[s];
            if (len && off - len < best) best = off - len;
        }
        st.s   = s;
        st.off = off;
        return best;
    }

private:
    uint8_t               cls_[256] = {};
    uint32_t              ncls_     = 1;
    uint32_t              max_len_  = 0;
    std::vector<uint32_t> next_{ 0u };      /* [state * ncls_ + class] -> state */
    std::vector<uint32_t> out_len_{ 0u };   /* longest word ending in the state, 0 for none */
};

}  /* namespace rad */
