/* safetensors.h -- an 8-byte length, a JSON object, and a blob.
 *
 * Written rather than depended on. The whole format is a header length, one JSON object mapping
 * tensor name to {dtype, shape, data_offsets}, and raw little-endian bytes; the parser below is
 * two hundred lines and it does not need to be a general JSON parser, because the document's shape
 * is fixed. A dependency here would be a dependency in the converter, which is the one tool that
 * has to run on a machine that has just been handed a checkpoint.
 *
 * mmap, no copy. A 46-shard model is opened one shard at a time, so peak RSS stays bounded
 * regardless of model size.
 */
#pragma once
#include "../rad_internal.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace rad {

struct StTensor {
    std::string          name;
    uint32_t             dtype = RAD_DT_INVALID;   /* RAD_* ; RAD_DT_INVALID if unrecognised */
    std::string          dtype_str;                /* as written, for the diagnostic */
    std::vector<int64_t> shape;
    uint64_t             offset = 0;               /* relative to the data section */
    uint64_t             bytes  = 0;
    const uint8_t*       data   = nullptr;

    int64_t numel() const {
        if (shape.empty()) return 0;
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return n;
    }
};

class SafeTensorsFile {
public:
    SafeTensorsFile() = default;
    ~SafeTensorsFile();
    SafeTensorsFile(const SafeTensorsFile&) = delete;
    SafeTensorsFile& operator=(const SafeTensorsFile&) = delete;

    int  open(const std::string& path);
    void close();

    const StTensor*              find(const std::string& name) const;
    const std::vector<StTensor>& tensors()  const { return tensors_; }
    const std::string&           metadata() const { return metadata_; }
    const std::string&           path()     const { return path_; }

private:
    int         fd_ = -1;
    uint8_t*    map_ = nullptr;
    uint64_t    map_bytes_ = 0;
    std::string path_, metadata_;
    std::vector<StTensor> tensors_;
    std::unordered_map<std::string, size_t> index_;
};

/* model.safetensors.index.json -> { tensor name: shard filename }. A single-file checkpoint has
 * no index; the caller falls back to opening model.safetensors directly. */
int st_load_index(const std::string& index_path,
                  std::unordered_map<std::string, std::string>* out);

}  /* namespace rad */
