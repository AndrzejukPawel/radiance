/* imatrix.h -- llama.cpp's importance matrix, both of the formats it has written.
 *
 * An imatrix is a per-column activation second moment, collected over a calibration set. It does
 * two jobs here and they are unrelated:
 *
 *   - the QUANTISER weights its per-block error by it, so bits go where the activations are
 *     (spec §4.3);
 *   - the CONVERTER derives the expert-popularity profile from its activation counts, which is
 *     what the placement planner warm-starts from (spec §5.3). That is a warm start and not a
 *     policy: imatrix popularity is not decode popularity, and the heat engine re-ranks from real
 *     routing within a few dispatches regardless.
 *
 * Two formats, because llama.cpp has written two and both are in the wild:
 *
 *   LEGACY  a flat little-endian binary: int32 n_entries, then per entry
 *           int32 name_len, char[name_len], int32 ncall, int32 nval, float[nval].
 *           The values are SUMS OF SQUARES ALREADY DIVIDED BY ncall by the writer in some
 *           versions and not in others, which is exactly why the newer format exists.
 *   GGUF    a GGUF v3 container whose tensors are `<name>.in_sum2` (f32 [cols] or [cols,n_expert])
 *           and `<name>.counts` (f32 [1] or [1,n_expert]), plus `imatrix.chunk_count`,
 *           `imatrix.chunk_size` and `imatrix.datasets` in the KV block. Per-expert data is a
 *           second dimension, which the legacy format could not express at all.
 *
 * The GGUF walker below is local rather than core/text's. That is deliberate: the imatrix is a
 * file format and core/format owns file formats, the walker is a hundred lines, and coupling the
 * quantiser's input to the tokeniser component's reader would join two things with no other
 * relationship. It also has to sit beside the legacy reader, which is not GGUF at all.
 */
#pragma once
#include "../rad_internal.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace rad {

/* `sum2` and `counts` point into the mapped file for a GGUF imatrix, at whatever offset its
 * general.alignment put them -- which may be 1 -- so they are read with memcpy, never in place.
 * importance() and count() do that. */
struct ImatrixEntry {
    std::string  name;              /* the checkpoint tensor name, suffix stripped */
    const float* sum2   = nullptr;  /* [n_expert][cols], expert-major and contiguous */
    const float* counts = nullptr;  /* [n_expert], or null in the legacy format */
    int64_t      cols     = 0;
    int64_t      n_expert = 1;      /* 1 for a dense tensor */
    int64_t      ncall    = 0;      /* legacy only: calls folded into sum2 */
};

class ImatrixFile {
public:
    ImatrixFile() = default;
    ~ImatrixFile();
    ImatrixFile(const ImatrixFile&) = delete;
    ImatrixFile& operator=(const ImatrixFile&) = delete;

    /* Sniffs the magic and picks a reader. RAD_E_FORMAT if it is neither. */
    int  open(const std::string& path);
    void close();

    const ImatrixEntry*              find(const std::string& name) const;
    const std::vector<ImatrixEntry>& entries() const { return entries_; }

    const std::string& dataset()     const { return dataset_; }
    int64_t            chunk_count() const { return chunk_count_; }
    int64_t            chunk_size()  const { return chunk_size_; }
    bool               is_gguf()     const { return gguf_; }
    const std::string& path()        const { return path_; }

    /* Column importance for one expert, NORMALISED TO MEAN 1. The normalisation is what makes a
     * weighted error comparable with an unweighted one, and therefore what makes a quantiser's
     * error report mean the same thing with and without an imatrix. Returns false if the tensor
     * is absent, the expert is out of range, or the entry carries no values. */
    bool importance(const std::string& name, int64_t expert, std::vector<float>* out) const;

    /* Raw activation count for one expert -- the popularity signal, unnormalised. 0 if absent. */
    double count(const std::string& name, int64_t expert) const;

private:
    int read_legacy(const uint8_t* p, uint64_t n);
    int read_gguf(const uint8_t* p, uint64_t n);

    int         fd_ = -1;
    uint8_t*    map_ = nullptr;
    uint64_t    map_bytes_ = 0;
    bool        gguf_ = false;
    std::string path_, dataset_;
    int64_t     chunk_count_ = 0, chunk_size_ = 0;

    /* The legacy format's floats are not aligned in the file and are followed by no counts, so
     * they are copied out rather than pointed at. GGUF payloads are pointed at. */
    std::vector<std::vector<float>>         owned_;
    std::vector<ImatrixEntry>               entries_;
    std::unordered_map<std::string, size_t> index_;
};

}  /* namespace rad */
