/* gguf.h -- a reader for the GGUF container, and nothing else.
 *
 * spec §12 lists `ggml/src/gguf.cpp` as a lift, but that file is a reader, a writer and a
 * ggml_context builder welded together: it fills `struct ggml_tensor`, calls `ggml_blck_size`,
 * and pulls in `ggml.h` for both. Taking it whole would mean taking ggml, which §12 says
 * explicitly we do not. So the reader is rewritten here under our own name -- the wire format is
 * a header, a KV table, a tensor directory and a padded blob, and that is a page of code -- and
 * `vendor/UPSTREAM` records the file it was read from rather than copied from.
 *
 * What we keep from upstream is the one thing that is genuinely data rather than code: the block
 * geometry of every ggml dtype, without which a tensor's byte size cannot be computed. That table
 * is transcribed from `ggml_blck_size`/`ggml_type_size` and is marked as such below.
 *
 * The file is mmapped whole and never copied. rad-convert walks the directory, reads what it
 * wants through `tensor_data()`, and writes the .rad; nothing here interprets tensor CONTENT.
 */
#pragma once
#include "rad_internal.h"

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rad {

/* GGUF's own value type numbering. These are wire values -- fixed by the format, not by us. */
enum GgufType {
    GGUF_U8 = 0, GGUF_I8 = 1, GGUF_U16 = 2, GGUF_I16 = 3, GGUF_U32 = 4, GGUF_I32 = 5,
    GGUF_F32 = 6, GGUF_BOOL = 7, GGUF_STR = 8, GGUF_ARR = 9, GGUF_U64 = 10,
    GGUF_I64 = 11, GGUF_F64 = 12
};

/* One metadata entry.
 *
 * A scalar is stored as a one-element vector, so a caller reading `tokenizer.ggml.tokens` and a
 * caller reading `general.name` write the same code and neither branches on `is_array`. The cost
 * is one heap allocation per scalar KV at open, against a KV table that is a few hundred rows;
 * the benefit is that the every-caller path has no special case in it, and a special case in a
 * metadata reader is where a truncated vocab comes from. */
struct GgufKV {
    std::string key;
    uint32_t    type = GGUF_U8;   /* for an array, the ELEMENT type; GGUF_ARR never appears here */
    bool        is_array = false;

    /* Populated according to `type`: integers and bools widen into `i`, floats into `f`,
     * strings into `s`. The other two stay empty. */
    std::vector<int64_t>     i;
    std::vector<double>      f;
    std::vector<std::string> s;
};

/* One tensor directory row. `offset` is from the start of the data section, which is where GGUF
 * puts it; `data_off() + offset` is the file offset, and `tensor_data()` does that arithmetic. */
struct GgufTensor {
    std::string name;
    uint32_t    ggml_type = 0;
    uint32_t    n_dims = 0;
    int64_t     ne[4] = { 1, 1, 1, 1 };   /* GGUF is fixed at 4 dims, unused ones are 1 */
    uint64_t    offset = 0;
    uint64_t    bytes = 0;                /* computed from ne and the block geometry */
};

/* Block geometry for a ggml dtype. Transcribed from ggml.c's `type_traits` at llama.cpp
 * 06938ac12 -- see vendor/UPSTREAM. `blck` of 0 marks a removed or unknown type, and a tensor
 * carrying one is refused at open rather than silently mis-sized. */
struct GgufDtype {
    int         blck;        /* logical elements per stored block */
    size_t      block_bytes; /* bytes per stored block */
    const char* name;
};
const GgufDtype* gguf_dtype(uint32_t ggml_type);   /* null if out of range */

/* Map a ggml dtype onto RAD_* where the two agree exactly. Returns RAD_DT_INVALID for the
 * k-quants and the iq-quants: those have no RAD_* equivalent, and rad-convert has to dequantise
 * or refuse rather than reinterpret the bytes. */
uint32_t gguf_to_rad_dtype(uint32_t ggml_type);

class GgufFile {
public:
    GgufFile() = default;
    ~GgufFile();
    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    /* RAD_OK, or a negative RAD_E_*: RAD_E_IO if the file will not open or map,
     * RAD_E_FORMAT if the magic, version, or directory does not hold together. */
    int  open(const char* path);
    void close();
    bool is_open() const { return base_ != nullptr; }

    uint32_t version()   const { return version_; }
    uint64_t alignment() const { return alignment_; }
    uint64_t data_off()  const { return data_off_; }     /* file offset of the tensor blob */
    uint64_t file_bytes() const { return file_bytes_; }
    const void* base()   const { return base_; }         /* the mmap, or null */
    const std::string& path() const { return path_; }

    const std::vector<GgufKV>&     kv()      const { return kv_; }
    const std::vector<GgufTensor>& tensors() const { return tensors_; }

    const GgufKV*     find(std::string_view key)  const;
    const GgufTensor* tensor(std::string_view name) const;

    /* Typed reads. Each returns false and leaves *out untouched when the key is absent or holds
     * the wrong shape of value -- absence and a type mismatch are the same thing to a caller that
     * has a default, and the ones that do not have a default check the return. */
    bool get_i64(std::string_view key, int64_t* out) const;
    bool get_f64(std::string_view key, double* out) const;
    bool get_str(std::string_view key, std::string* out) const;
    bool get_strs(std::string_view key, std::vector<std::string>* out) const;
    bool get_i64s(std::string_view key, std::vector<int64_t>* out) const;
    bool get_f32s(std::string_view key, std::vector<float>* out) const;

    /* A pointer into the mmap. Null if the tensor's extent leaves the file, which is a corrupt
     * container rather than a caller error -- open() already refused those, so this is belt. */
    const void* tensor_data(const GgufTensor& t) const;

private:
    std::string path_;
    const uint8_t* base_ = nullptr;
    uint64_t  file_bytes_ = 0;
    uint32_t  version_ = 0;
    uint64_t  alignment_ = 32;
    uint64_t  data_off_ = 0;

    std::vector<GgufKV>     kv_;
    std::vector<GgufTensor> tensors_;
    std::unordered_map<std::string, size_t> kv_index_;
    std::unordered_map<std::string, size_t> tensor_index_;
};

}  /* namespace rad */
