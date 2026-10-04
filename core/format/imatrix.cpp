/* imatrix.cpp -- the two llama.cpp imatrix formats and the small GGUF walker the newer one needs.
 *
 * Little-endian only, like everything else here. Every read is bounds-checked against the mapping
 * before it happens: an imatrix is a file a user was handed by somebody else, and the quantiser
 * that consumes it runs unattended for an hour.
 */
#include "imatrix.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace rad {
namespace {

/* GGUF's value type tags. Only what a walker needs to know how far to step. */
enum : uint32_t {
    GT_U8 = 0, GT_I8 = 1, GT_U16 = 2, GT_I16 = 3, GT_U32 = 4, GT_I32 = 5, GT_F32 = 6,
    GT_BOOL = 7, GT_STR = 8, GT_ARR = 9, GT_U64 = 10, GT_I64 = 11, GT_F64 = 12
};
constexpr uint32_t GGML_TYPE_F32 = 0;

struct Cursor {
    const uint8_t* p;
    const uint8_t* end;
    bool ok = true;

    bool take(uint64_t n, const void** out = nullptr) {
        if (!ok || (uint64_t)(end - p) < n) { ok = false; return false; }
        if (out) *out = p;
        p += n;
        return true;
    }
    template <class T> bool get(T* out) {
        const void* q = nullptr;
        if (!take(sizeof(T), &q)) return false;
        std::memcpy(out, q, sizeof(T));
        return true;
    }
    bool str(std::string* out) {
        uint64_t n = 0;
        if (!get(&n)) return false;
        const void* q = nullptr;
        if (!take(n, &q)) return false;
        if (out) out->assign((const char*)q, (size_t)n);
        return true;
    }
    /* Step over a value of any type without interpreting it. Arrays recurse, which is how
     * imatrix.datasets (an array of strings) is skipped when we do not want it. */
    bool skip_value(uint32_t type) {
        switch (type) {
            case GT_U8: case GT_I8: case GT_BOOL:            return take(1);
            case GT_U16: case GT_I16:                        return take(2);
            case GT_U32: case GT_I32: case GT_F32:           return take(4);
            case GT_U64: case GT_I64: case GT_F64:           return take(8);
            case GT_STR:                                     return str(nullptr);
            case GT_ARR: {
                uint32_t et = 0; uint64_t n = 0;
                if (!get(&et) || !get(&n)) return false;
                for (uint64_t i = 0; i < n && ok; ++i) if (!skip_value(et)) return false;
                return ok;
            }
            default: ok = false; return false;
        }
    }
};

}  /* namespace */

ImatrixFile::~ImatrixFile() { close(); }

void ImatrixFile::close() {
    if (map_) ::munmap(map_, (size_t)map_bytes_);
    if (fd_ >= 0) ::close(fd_);
    map_ = nullptr; map_bytes_ = 0; fd_ = -1;
    gguf_ = false;
    entries_.clear(); index_.clear(); owned_.clear();
    dataset_.clear(); path_.clear();
    chunk_count_ = chunk_size_ = 0;
}

int ImatrixFile::open(const std::string& path) {
    close();
    path_ = path;

    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) { RAD_ERR("%s: %s", path.c_str(), std::strerror(errno)); return RAD_E_IO; }

    struct stat st{};
    if (::fstat(fd_, &st) != 0) { RAD_ERR("%s: %s", path.c_str(), std::strerror(errno));
                                  close(); return RAD_E_IO; }
    map_bytes_ = (uint64_t)st.st_size;
    if (map_bytes_ < 8) {
        RAD_ERR("%s: %llu bytes -- too short to be an imatrix", path.c_str(),
                (unsigned long long)map_bytes_);
        close();
        return RAD_E_FORMAT;
    }

    void* m = ::mmap(nullptr, (size_t)map_bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (m == MAP_FAILED) { RAD_ERR("%s: mmap: %s", path.c_str(), std::strerror(errno));
                           close(); return RAD_E_IO; }
    map_ = (uint8_t*)m;

    int s;
    if (std::memcmp(map_, "GGUF", 4) == 0) { gguf_ = true;  s = read_gguf(map_, map_bytes_); }
    else                                   { gguf_ = false; s = read_legacy(map_, map_bytes_); }
    if (s < 0) { close(); return s; }

    for (size_t i = 0; i < entries_.size(); ++i) index_.emplace(entries_[i].name, i);
    RAD_INFO("imatrix %s: %s, %zu tensor(s)%s", path.c_str(), gguf_ ? "gguf" : "legacy",
             entries_.size(),
             chunk_count_ ? fmt(", %lld chunks of %lld", (long long)chunk_count_,
                                (long long)chunk_size_).c_str() : "");
    return RAD_OK;
}

/* ------------------------------------------------------------------ legacy */
int ImatrixFile::read_legacy(const uint8_t* p, uint64_t n) {
    Cursor c{ p, p + n };

    int32_t n_entries = 0;
    if (!c.get(&n_entries) || n_entries <= 0 || n_entries > (1 << 20)) {
        RAD_ERR("%s: not a legacy imatrix (entry count %d)", path_.c_str(), n_entries);
        return RAD_E_FORMAT;
    }

    owned_.reserve((size_t)n_entries);
    entries_.reserve((size_t)n_entries);

    for (int32_t i = 0; i < n_entries; ++i) {
        int32_t name_len = 0;
        if (!c.get(&name_len) || name_len < 0 || (uint64_t)name_len > (uint64_t)(c.end - c.p)) {
            RAD_ERR("%s: entry %d has a %d-byte name, which does not fit", path_.c_str(), i,
                    name_len);
            return RAD_E_FORMAT;
        }
        const void* np = nullptr;
        if (!c.take((uint64_t)name_len, &np)) break;
        std::string name((const char*)np, (size_t)name_len);

        int32_t ncall = 0, nval = 0;
        if (!c.get(&ncall) || !c.get(&nval) || nval < 0) {
            RAD_ERR("%s: entry '%s' is truncated", path_.c_str(), name.c_str());
            return RAD_E_FORMAT;
        }
        const void* vp = nullptr;
        if (!c.take((uint64_t)nval * sizeof(float), &vp)) {
            RAD_ERR("%s: entry '%s' claims %d values, which run past the end of the file",
                    path_.c_str(), name.c_str(), nval);
            return RAD_E_FORMAT;
        }

        /* Copied, not pointed at: the legacy layout puts floats at an offset that depends on the
         * name length, so nothing in the file is guaranteed 4-byte aligned. */
        owned_.emplace_back((size_t)nval);
        std::memcpy(owned_.back().data(), vp, (size_t)nval * sizeof(float));

        ImatrixEntry e;
        e.name     = std::move(name);
        e.sum2     = owned_.back().data();
        e.counts   = nullptr;
        e.cols     = nval;
        e.n_expert = 1;
        e.ncall    = ncall;
        entries_.push_back(std::move(e));
    }

    if (!c.ok && entries_.empty()) {
        RAD_ERR("%s: truncated before the first entry", path_.c_str());
        return RAD_E_FORMAT;
    }
    /* The legacy format cannot express per-expert data at all, and a MoE quantised against it
     * weights every expert by the layer's aggregate. Say so once rather than let the quality
     * difference show up as an unexplained regression. */
    RAD_WARN("%s is the legacy imatrix format: it has no per-expert dimension, so every expert in "
             "a layer will be weighted by the layer's aggregate activations. Re-collect with a "
             "current llama.cpp for per-expert importance.", path_.c_str());
    return RAD_OK;
}

/* ------------------------------------------------------------------ gguf */
int ImatrixFile::read_gguf(const uint8_t* p, uint64_t n) {
    Cursor c{ p + 4, p + n };

    uint32_t version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (!c.get(&version) || !c.get(&n_tensors) || !c.get(&n_kv)) {
        RAD_ERR("%s: truncated GGUF header", path_.c_str());
        return RAD_E_FORMAT;
    }
    if (version != 3) {
        RAD_ERR("%s: GGUF version %u; this reader handles 3", path_.c_str(), version);
        return RAD_E_FORMAT;
    }

    uint64_t alignment = 32;   /* GGUF's default when general.alignment is absent */

    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!c.str(&key) || !c.get(&type)) {
            RAD_ERR("%s: truncated at KV %llu", path_.c_str(), (unsigned long long)i);
            return RAD_E_FORMAT;
        }
        if (key == "general.alignment" && type == GT_U32) {
            uint32_t a = 0;
            if (!c.get(&a)) break;
            if (a == 0 || (a & (a - 1)) != 0) {
                RAD_ERR("%s: general.alignment is %u, which is not a power of two", path_.c_str(), a);
                return RAD_E_FORMAT;
            }
            alignment = a;
        } else if (key == "imatrix.chunk_count" && type == GT_U32) {
            uint32_t v = 0; if (!c.get(&v)) break; chunk_count_ = v;
        } else if (key == "imatrix.chunk_size" && type == GT_U32) {
            uint32_t v = 0; if (!c.get(&v)) break; chunk_size_ = v;
        } else if (key == "imatrix.datasets" && type == GT_ARR) {
            uint32_t et = 0; uint64_t cnt = 0;
            if (!c.get(&et) || !c.get(&cnt)) break;
            for (uint64_t k = 0; k < cnt && c.ok; ++k) {
                if (et == GT_STR) { std::string s; if (!c.str(&s)) break;
                                    if (dataset_.empty()) dataset_ = s; }
                else if (!c.skip_value(et)) break;
            }
        } else if (!c.skip_value(type)) {
            RAD_ERR("%s: KV '%s' has type %u, which this walker does not know how to step over",
                    path_.c_str(), key.c_str(), type);
            return RAD_E_FORMAT;
        }
    }
    if (!c.ok) { RAD_ERR("%s: truncated in the KV block", path_.c_str()); return RAD_E_FORMAT; }

    /* THE TENSOR COUNT IS A NUMBER FROM THE FILE, and it sizes an allocation. Every tensor info
     * is at least a name length, a dimension count, a type and an offset -- 24 bytes -- so a count
     * the rest of the file cannot hold is refused here rather than handed to reserve(), where it
     * is a multi-exabyte request that ends the process. */
    constexpr uint64_t kMinInfoBytes = 8 + 4 + 4 + 8;
    if (n_tensors > (uint64_t)(c.end - c.p) / kMinInfoBytes) {
        RAD_ERR("%s: the header declares %llu tensors, and the %llu bytes after the KV block "
                "cannot hold that many tensor infos", path_.c_str(),
                (unsigned long long)n_tensors, (unsigned long long)(c.end - c.p));
        return RAD_E_FORMAT;
    }

    struct Info { std::string name; std::vector<uint64_t> dims; uint32_t type; uint64_t off; };
    std::vector<Info> infos;
    infos.reserve((size_t)n_tensors);

    for (uint64_t i = 0; i < n_tensors; ++i) {
        Info t{};
        uint32_t nd = 0;
        if (!c.str(&t.name) || !c.get(&nd) || nd > 4) {
            RAD_ERR("%s: truncated or malformed tensor info %llu", path_.c_str(),
                    (unsigned long long)i);
            return RAD_E_FORMAT;
        }
        t.dims.resize(nd);
        for (uint32_t d = 0; d < nd; ++d) if (!c.get(&t.dims[d])) break;
        if (!c.get(&t.type) || !c.get(&t.off)) {
            RAD_ERR("%s: truncated tensor info for '%s'", path_.c_str(), t.name.c_str());
            return RAD_E_FORMAT;
        }
        infos.push_back(std::move(t));
    }

    const uint64_t here = (uint64_t)(c.p - p);
    const uint64_t data_start = (here + alignment - 1) / alignment * alignment;
    if (data_start > n) {
        RAD_ERR("%s: the data section starts past the end of the file", path_.c_str());
        return RAD_E_FORMAT;
    }

    /* Join `<base>.in_sum2` with `<base>.counts`. Insertion order defines entries() order, so a
     * report over the file reads in the order the collector wrote it. `held` is how many floats
     * each half actually spans in the file, parallel to entries_, for the extent checks below. */
    std::unordered_map<std::string, size_t> by_base;
    struct Held { uint64_t sum2 = 0, counts = 0; };
    std::vector<Held> held;
    const uint64_t data_floats = (n - data_start) / sizeof(float);

    for (const Info& t : infos) {
        const bool is_sum2   = t.name.size() > 8 && t.name.compare(t.name.size() - 8, 8, ".in_sum2") == 0;
        const bool is_counts = t.name.size() > 7 && t.name.compare(t.name.size() - 7, 7, ".counts") == 0;
        if (!is_sum2 && !is_counts) continue;
        if (t.type != GGML_TYPE_F32) {
            RAD_WARN("%s: '%s' is ggml type %u, not f32 -- ignored", path_.c_str(),
                     t.name.c_str(), t.type);
            continue;
        }

        /* The element count is a product of four numbers from the file, so it is bounded by the
         * data section BEFORE each multiply: a product that wraps can come out small, pass the
         * bounds test below and be read at its declared extent. */
        uint64_t numel = 1;
        bool     fits  = true;
        for (uint64_t d : t.dims) if (d == 0) numel = 0;
        for (uint64_t d : t.dims) {
            if (numel == 0) break;
            if (numel > data_floats / d) { fits = false; break; }
            numel *= d;
        }
        const uint64_t bytes = numel * sizeof(float);
        if (!fits || t.off > n - data_start || bytes > n - data_start - t.off) {
            RAD_ERR("%s: '%s' runs past the end of the file", path_.c_str(), t.name.c_str());
            return RAD_E_FORMAT;
        }
        const float* fp = (const float*)(p + data_start + t.off);

        std::string base = t.name.substr(0, t.name.size() - (is_sum2 ? 8 : 7));
        auto it = by_base.find(base);
        if (it == by_base.end()) {
            ImatrixEntry e;
            e.name = base;
            by_base.emplace(base, entries_.size());
            it = by_base.find(base);
            entries_.push_back(std::move(e));
            held.emplace_back();
        }
        ImatrixEntry& e = entries_[it->second];
        if (is_sum2) {
            e.sum2     = fp;
            e.cols     = t.dims.empty() ? 0 : (int64_t)t.dims[0];
            e.n_expert = t.dims.size() > 1 ? (int64_t)t.dims[1] : 1;
            held[it->second].sum2 = numel;
        } else {
            e.counts = fp;
            held[it->second].counts = numel;
        }
    }

    /* THE QUERIES INDEX BY WHAT THE ENTRY CLAIMS, so the claim has to fit what the file holds.
     * `importance` reads cols floats at expert * cols and `count` reads counts[expert] for every
     * expert below n_expert -- both taken from the .in_sum2 dims -- so a .counts shorter than the
     * expert count, or an .in_sum2 whose trailing dims multiply to less than cols x n_expert,
     * would be read past its own extent. */
    std::vector<ImatrixEntry> kept;
    kept.reserve(entries_.size());
    for (size_t k = 0; k < entries_.size(); ++k) {
        ImatrixEntry& e = entries_[k];
        /* An entry with counts and no values is a collector bug, not something to weight by. */
        if (!e.sum2 || e.cols <= 0) {
            RAD_WARN("%s: '%s' has counts but no .in_sum2 -- dropped", path_.c_str(),
                     e.name.c_str());
            continue;
        }
        if (e.n_expert < 1 || (uint64_t)e.cols > held[k].sum2 / (uint64_t)e.n_expert) {
            RAD_WARN("%s: '%s.in_sum2' claims %lld column(s) for %lld expert(s) and holds %llu "
                     "value(s) -- dropped", path_.c_str(), e.name.c_str(), (long long)e.cols,
                     (long long)e.n_expert, (unsigned long long)held[k].sum2);
            continue;
        }
        if (e.counts && held[k].counts < (uint64_t)e.n_expert) {
            RAD_WARN("%s: '%s.counts' holds %llu value(s) for %lld expert(s) -- ignored, so this "
                     "tensor contributes no expert profile", path_.c_str(), e.name.c_str(),
                     (unsigned long long)held[k].counts, (long long)e.n_expert);
            e.counts = nullptr;
        }
        kept.push_back(std::move(e));
    }
    entries_ = std::move(kept);
    return RAD_OK;
}

/* ------------------------------------------------------------------ queries */
const ImatrixEntry* ImatrixFile::find(const std::string& name) const {
    auto it = index_.find(name);
    return it == index_.end() ? nullptr : &entries_[it->second];
}

bool ImatrixFile::importance(const std::string& name, int64_t expert,
                             std::vector<float>* out) const {
    const ImatrixEntry* e = find(name);
    if (!e || !e->sum2 || e->cols <= 0 || expert < 0 || expert >= e->n_expert) return false;
    if (!out) return true;

    const float* src = e->sum2 + expert * e->cols;

    /* NORMALISED TO MEAN 1, and that is the whole of it.
     *
     * The row is NOT divided by the expert's activation count first, although that looks like a way
     * to say "an expert that saw ten times the tokens is not ten times as important". The division
     * cannot do that: the result is a single expert's row scaled to mean 1, and a uniform divisor
     * applied to every column of that row is removed exactly by the scaling that follows it. What
     * it can do is flush small columns to zero against a large count, and an all-zero row then
     * takes the degenerate branch below -- so its only effect would be to lose weights.
     *
     * Popularity ACROSS experts is a different question and has its own query, `count`, which
     * reports the raw number for exactly this reason.
     *
     * Mean 1 is what makes a weighted error comparable with an unweighted one, and therefore what
     * makes the quantiser's error report mean the same thing with and without an imatrix. A
     * degenerate (all-zero) row falls back to uniform weights rather than to NaN. */
    out->resize((size_t)e->cols);
    std::memcpy(out->data(), src, (size_t)e->cols * sizeof(float));
    double sum = 0.0;
    for (float v : *out) sum += (double)v;

    const double mean = sum / (double)e->cols;
    if (!(mean > 0.0)) { std::fill(out->begin(), out->end(), 1.0f); return true; }
    const float inv = (float)(1.0 / mean);
    for (float& v : *out) v *= inv;
    return true;
}

double ImatrixFile::count(const std::string& name, int64_t expert) const {
    const ImatrixEntry* e = find(name);
    if (!e || !e->counts || expert < 0 || expert >= e->n_expert) return 0.0;
    float v = 0.0f;
    std::memcpy(&v, e->counts + expert, sizeof v);
    return (double)v;
}

}  /* namespace rad */
