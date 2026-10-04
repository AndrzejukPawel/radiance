/* radfile.h -- the .rad reader and writer.
 *
 * The container is frozen in abi/rad_format.h; this file implements exactly it and adds
 * nothing to the on-disk shape. What lives here is the two halves that must never disagree: the
 * writer that produces a layout and the reader that maps it.
 *
 * The reader is an mmap and an index build, and then it is done. Every invariant a reader relies
 * on is CHECKED AT OPEN rather than asserted at use:
 *
 *   - every offset and count is bounds-checked against the mapped length, so a truncated or
 *     corrupt file is RAD_E_FORMAT with a message and not a SIGSEGV three components later;
 *   - every rad_stroff lands inside a string blob whose last byte is NUL, so str() is a plain
 *     pointer and never a scan;
 *   - the expert directory's fixed stride is RE-DERIVED and verified, so
 *     `base + id * stride` on the critical path is arithmetic somebody already proved.
 *
 * The cost of the last one, stated plainly: a container whose experts within a layer are not all
 * the same shape and format is REFUSED, not read slowly. Addressing by arithmetic is the format's
 * reason to exist; a file that cannot be addressed that way is not a .rad, and reading it with a
 * table walk would silently give back the property the whole design was paying for.
 */
#pragma once
#include "../rad_internal.h"

#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rad {

/* ================================================================== vocab */
/* The MUTABLE side of the vocab -- VocabBuild, the tokenizer.json interpreter and the serialiser
 * that turns one into a RadVocabHeader -- belongs to core/text, because interpreting a declared
 * normaliser chain is that component's whole job (spec §12). What belongs here is the container's
 * half of the contract, and it is two things:
 *
 *   RadWriter::intern_string()   so the vocab section's strings go into the .rad's ONE string
 *                                blob rather than a second one nobody else can read;
 *   RadWriter::set_vocab_section() so the serialised bytes are placed and rebased by the writer
 *                                that owns every other offset in the file.
 *
 * core/text's `StringSink` is the interface that joins them, and rad-convert supplies the two-line
 * adapter. Keeping VocabBuild out of this header is what stops two definitions of it existing.
 *
 * The read-back view below is the container's own: pointers into the mapping, already
 * bounds-checked at open. core/text's vocab_deserialize() copies out of it. */
/* NOT named VocabView: core/sample/vocab_view.h declares an ABSTRACT CLASS of that name in this
 * same namespace, and two definitions of one name are an ODR violation the moment one translation
 * unit includes both -- which core/engine.cpp does, because it loads the container and drives the
 * sampler. This one is the container's read-back view of the vocab SECTION; the other is what the
 * sampler needs to know about a vocabulary. Different things, deliberately different names. */
struct VocabSection {
    const RadVocabHeader* h        = nullptr;
    const rad_stroff*     tok_text = nullptr;   /* [n_tokens] */
    const float*          tok_score= nullptr;   /* [n_tokens] or null */
    const uint8_t*        tok_type = nullptr;   /* [n_tokens] or null */
    const uint32_t*       merges   = nullptr;   /* [n_merges][2] or null */
    const RadVocabStep*   steps    = nullptr;   /* [n_steps] or null */
    explicit operator bool() const { return h != nullptr; }
};

/* ================================================================== reader */
class RadFile {
public:
    RadFile() = default;
    ~RadFile();
    RadFile(const RadFile&) = delete;
    RadFile& operator=(const RadFile&) = delete;

    /* mmap, validate, index. Returns RAD_OK, or RAD_E_IO / RAD_E_FORMAT with the reason logged at
     * Error naming the field that failed -- a bad container is diagnosed here or nowhere. */
    int  open(const char* path);
    void close();
    bool is_open() const { return base_ != nullptr; }

    const char*          path()   const { return path_.c_str(); }
    const RadFileHeader& header() const { return *hdr_; }
    const uint8_t*       base()   const { return base_; }
    int64_t              size()   const { return size_; }

    /* Drop the blob from the page cache and from this process's page tables.
     *
     * A container larger than RAM cannot be traversed twice without this. When the weight load
     * walks the whole file once per rank, the second walk meets a page cache that is entirely the
     * first walk's pages: nothing it needs is resident, and the readahead it triggers evicts the
     * pages it is about to want. The load then makes no forward progress while the device sits
     * idle behind a stream of NVMe reads that are immediately thrown away.
     *
     * Only the blob: the tables are small, read constantly, and wanted resident for the life of
     * the process. Advisory in both directions -- if the kernel declines, the load is exactly as
     * it was. */
    void drop_blob_cache() const;

    /* Drop one range of the blob behind a cursor that will not come back for it.
     *
     * The weight load reads the container once, front to back, and nothing re-reads a window it
     * has finished with -- but buffered reads leave every one of those bytes in the page cache,
     * and on a container larger than RAM that turns into reclaim pressure the read itself then
     * waits on, so the sweep rate decays as it advances. Dropping each window as it is consumed
     * holds the rate flat.
     *
     * THE PRECONDITION IS ONE CURSOR. Drop-behind is wrong with several cursors over one file,
     * because what one discards is what another is about to want -- the same reason a sequential
     * access hint is worse than silence in that arrangement. This call is correct only for a
     * single cursor moving in one direction with no revisits, where the pages are genuinely
     * dead. */
    void drop_range(int64_t off, int64_t bytes) const;

    /* Read `bytes` at `off` into `dst` with pread, bypassing the mapping entirely.
     *
     * The load phase wants this and not the mapping. An async copy whose source is pageable
     * memory is staged by the HIP runtime through its own pinned buffer, and that path degrades
     * without bound as a load proceeds: even with the source pages already resident, a single
     * small hipMemcpyAsync can block for seconds. Reading into a pinned buffer the loader owns
     * makes the device copy a real DMA from pinned memory, which has none of that behaviour.
     *
     * Returns RAD_OK, or RAD_E_IO with the reason logged. Short reads are retried; EINTR is not
     * an error. */
    int read_at(void* dst, int64_t bytes, int64_t off) const;

    /* Strings. Offset 0 is the empty string; every offset was bounds-checked at open. */
    const char* str(rad_stroff off) const {
        return (const char*)(base_ + hdr_->str_off + (off < str_bytes_ ? off : 0));
    }

    /* ------------------------------------------------------------ metadata */
    int64_t          meta_count() const { return (int64_t)hdr_->meta_count; }
    const RadFileKV& meta_at(int64_t i) const { return meta_[i]; }
    bool             get_i(const char* key, int64_t* out) const;
    bool             get_f(const char* key, double* out) const;
    const char*      get_s(const char* key, const char* dflt = nullptr) const;

    /* ------------------------------------------------------------ weight directory */
    int64_t             entry_count() const { return (int64_t)hdr_->dir_count; }
    const RadFileEntry& entry(int64_t i) const { return dir_[i]; }
    const RadFileEntry* find(std::string_view name) const;
    const uint8_t*      data(const RadFileEntry& e) const { return base_ + e.offset; }

    /* An entry's encoding and its planes, in the encoding's order. Every plane was checked at open
     * to be exactly the size its encoding and the entry's shape say. */
    const RadEncoding&  encoding(const RadFileEntry& e) const { return encs_[e.enc]; }
    const RadFilePlane& plane(const RadFileEntry& e, int k) const {
        return planes_[e.plane_first + (uint64_t)k];
    }
    const uint8_t*      data(const RadFilePlane& p) const { return base_ + p.offset; }
    int64_t             encoding_count() const { return (int64_t)hdr_->enc_count; }
    const RadEncoding&  encoding_at(int64_t i) const { return encs_[i]; }

    /* ------------------------------------------------------------ fixed-stride addressing */
    /* One layer's experts. All of them share a shape and a format, which is what makes
     *     addr = base + id * stride
     * O(1) arithmetic with no table walk on the critical path (spec §4.1). Verified at open. */
    struct ExpertGroup {
        int32_t  layer      = -1;
        int32_t  n_expert   = 0;
        int32_t  n_slot     = 0;      /* gate/up/down = 3 */
        int64_t  first_index= 0;      /* directory index of (expert 0, slot 0) */
        uint64_t base       = 0;      /* file offset of expert 0's first byte */
        uint64_t stride     = 0;      /* bytes between consecutive experts, aligned */
        uint64_t unit_bytes = 0;      /* payload of one expert: every slot's every plane */
        /* Per-slot displacement from an expert's base. Fixed across experts by construction. */
        uint64_t slot_off[8] = {0};
        uint64_t slot_bytes[8] = {0};
    };

    int64_t            group_count() const { return (int64_t)groups_.size(); }
    const ExpertGroup* group_for_layer(int32_t layer) const;

    /* The whole point of the format, in one line each. No bounds check on the hot path: the
     * caller has the group and therefore has n_expert and n_slot. */
    const uint8_t* expert_ptr(const ExpertGroup& g, int32_t expert, int32_t slot) const {
        return base_ + g.base + (uint64_t)expert * g.stride + g.slot_off[slot];
    }
    const RadFileEntry& expert_entry(const ExpertGroup& g, int32_t expert, int32_t slot) const {
        return dir_[g.first_index + (int64_t)expert * g.n_slot + slot];
    }

    /* ------------------------------------------------------------ vocab and profile */
    VocabSection              vocab() const { return vocab_; }
    int64_t                   profile_count() const { return (int64_t)hdr_->prof_count; }
    const RadFileProfile*     profile() const { return prof_; }

private:
    int  validate();
    int  index_experts();

    std::string path_;
    uint8_t*    base_ = nullptr;
    int64_t     size_ = 0;
    int         fd_   = -1;

    const RadFileHeader*  hdr_  = nullptr;
    const RadFileKV*      meta_ = nullptr;
    const RadFileEntry*   dir_  = nullptr;
    const RadFilePlane*   planes_ = nullptr;
    const RadEncoding*    encs_ = nullptr;
    const RadFileProfile* prof_ = nullptr;
    uint64_t              str_bytes_ = 1;

    VocabSection                                  vocab_;
    std::vector<ExpertGroup>                      groups_;
    std::unordered_map<std::string_view, int64_t> by_name_;
};

/* ================================================================== writer */
/* Streaming, because the blob is the model: a 100 GB checkpoint cannot be staged in RAM to be
 * re-sorted at the end. The tables (strings, directory, planes, encodings, vocab, profile) ARE held
 * in memory -- a 512-expert 48-layer model is ~50k directory rows, which is megabytes.
 *
 * A WEIGHT IS RESERVED, THEN WRITTEN. add_weight() places the entry and every plane of its
 * encoding in the blob and returns its index; write_plane() fills any part of any plane, in any
 * order, at its position. That is what a quantiser needs -- it produces a block of rows of EVERY
 * plane at once -- and what a 51 GB table needs, whose codes are produced a block at a time
 * without the whole ever existing anywhere. finish() refuses a plane left short.
 *
 * The consequence, and it is a real constraint on callers: WEIGHTS MUST BE ADDED IN MOVEMENT-UNIT
 * ORDER. An expert's slots must be added consecutively, and a layer's experts must be added
 * consecutively and identically shaped. The writer verifies it and refuses with RAD_E_INVAL naming
 * the offending pair, rather than producing a file whose stride is a lie. rad-convert sorts its
 * manifest before it starts writing; that sort is over the manifest, not over the bytes.
 */
class RadWriter {
public:
    RadWriter() = default;
    ~RadWriter();
    RadWriter(const RadWriter&) = delete;
    RadWriter& operator=(const RadWriter&) = delete;

    int  begin(const char* path);
    void abort();                       /* discard, unlink both the target and the spill -- or, in
                                         * append mode, cut the file back to its old length */

    void set_arch(std::string_view s)       { arch_ = s; }
    void set_model_name(std::string_view s) { model_ = s; }
    void set_quant(std::string_view s)      { quant_ = s; }
    void set_recipe(std::string_view s)     { recipe_ = s; }
    void set_created_by(std::string_view s) { created_by_ = s; }

    void meta_i(std::string_view k, int64_t v);
    void meta_f(std::string_view k, double v);
    void meta_s(std::string_view k, std::string_view v);

    struct Weight {
        std::string name;                /* the logical weight's name */
        std::string quantizer;           /* "" for the checkpoint's own planes */
        std::string options;             /* "k=v,k=v", sorted by key */
        RadEncoding enc{};
        uint32_t    rank  = 0;
        int64_t     shape[RAD_MAX_RANK] = {0};   /* logical */
        int32_t     layer = -1, expert = -1, slot = 0;
    };

    /* APPEND IN PLACE: extend the container `old` (opened at `path`) with new weights, without
     * copying the ones it already holds.
     *
     * A container is a header, its tables, and the blob, and every offset in them is absolute, so
     * nothing forces a new weight to live anywhere but past the end. The new weights' bytes are
     * written there, then a fresh set of tables that names both the old weights -- at the offsets
     * they already have, through add_existing() -- and the new ones, and the header is rewritten
     * last. Until that one write the file is its old self with a tail the old header does not
     * reach; after it, the old tables are dead bytes in front of the blob. finish() also leaves
     * `<path>.pre-append`: the old header and length, which is all it takes to put the file back.
     *
     * It is what makes adding a component to a 120 GB container a gigabyte of I/O instead of a
     * 120 GB copy the disk may not have room for. A new EXPERT weight is refused: a layer's experts
     * are written consecutively at a fixed stride, and an appended one cannot join its layer.
     *
     * The caller must be the file's only user. A process that has it mapped reads the header
     * through its mapping, and a header that changes under it names tables it never validated. */
    int  begin_append(const char* path, const RadFile& old);
    /* A weight `old` already holds, recorded at the offsets it has there. Its encoding and planes
     * are the old entry's; its name, provenance and movement group are `w`'s. */
    int64_t add_existing(const Weight& w, const RadFile& old, const RadFileEntry& e);

    /* Reserve the weight and every plane of its encoding. Returns the directory index, or a
     * negative status: an invalid encoding, a shape it cannot cover, a movement-unit order the
     * stride cannot hold. */
    int64_t add_weight(const Weight& w);
    /* Reserve and write in one call: `planes[k]` is plane k's whole bytes. */
    int64_t add_weight(const Weight& w, const void* const* planes);

    /* `bytes` of plane `k` of weight `idx`, at byte `off` within the plane. */
    int write_plane(int64_t idx, int k, int64_t off, const void* data, int64_t bytes);
    int64_t plane_bytes(int64_t idx, int k) const;

    /* Intern into the .rad's single string blob. Public because the vocab section's writer lives
     * in core/text and has to put its token text somewhere the reader can find it; core/text's
     * StringSink is the one-method interface rad-convert adapts onto this. */
    rad_stroff intern_string(std::string_view s) { return intern(s); }

    /* The serialised vocab section. Its internal offsets must be SECTION-RELATIVE (serialise with
     * a section_file_off of 0); finish() rebases them once it knows where the section lands, the
     * same way it rebases the directory. */
    int  set_vocab_section(const void* blob, int64_t bytes);

    void add_profile(int32_t layer, int32_t expert, float share);

    /* FIX THE LAYOUT AND WRITE THE BLOB WHERE IT STAYS. Once every weight, the metadata, the vocab
     * section and the profile are in, the tables' sizes -- and so the blob's offset -- are known:
     * after seal() a plane write lands in the target itself and finish() writes the tables in
     * front of it, with no spill and no 100 GB copy. A table changed after it fails finish(). A
     * writer that is not sealed spills the blob and splices it in at finish(). */
    int  seal();

    /* Writes the header and every table, then -- unsealed -- splices the spilled blob in at a
     * 2 MiB boundary. */
    int  finish();

    int64_t data_bytes() const { return blob_bytes_; }
    int64_t weight_count() const { return (int64_t)dir_.size(); }

private:
    rad_stroff intern(std::string_view s);
    void       layout(RadFileHeader* h);
    uint32_t   intern_encoding(const RadEncoding& e);
    int        check_unit_invariant(const Weight& w, uint64_t off, uint64_t unit_bytes);

    std::string path_, spill_path_;
    int         fd_out_ = -1, fd_spill_ = -1;
    int64_t     spill_base_ = 0;    /* where blob offset 0 is in fd_spill_: 0, append_base_ or
                                     * the sealed data_off */
    bool        began_ = false;
    bool        sealed_ = false, late_ = false;
    RadFileHeader hdr_{};           /* the sealed layout */

    /* Append mode (begin_append). New bytes go at `append_base_`, the old length rounded up to a
     * pool boundary; an entry recorded by add_existing() is already absolute and is flagged so
     * finish() does not rebase it. */
    int  finish_append();
    int  check_complete() const;
    bool     append_ = false;
    int64_t  old_size_ = 0, append_base_ = 0;
    uint64_t old_data_off_ = 0;
    RadFileHeader old_hdr_{};
    std::vector<uint8_t> dir_abs_;

    std::string arch_, model_, quant_, recipe_, created_by_;

    std::string                                   strblob_{std::string(1, '\0')};
    std::unordered_map<std::string, rad_stroff>   strmap_;
    std::vector<RadFileKV>                        meta_;
    std::vector<RadFileEntry>                     dir_;
    std::vector<RadFilePlane>                     planes_;
    std::vector<int64_t>                          written_;   /* parallel to planes_ */
    std::vector<RadEncoding>                      encs_;
    std::vector<RadFileProfile>                   prof_;

    /* The vocab section, assembled in memory and appended verbatim. */
    std::vector<uint8_t> vocab_blob_;
    bool                 have_vocab_ = false;

    int64_t blob_bytes_ = 0;    /* the blob's reserved length */

    /* Movement-unit bookkeeping, so the fixed stride is a property the writer PROVES. */
    int32_t  cur_layer_ = -2, cur_expert_ = -2;
    int32_t  grp_layer_ = -2;
    int32_t  grp_n_slot_ = 0, grp_slot_seen_ = 0, grp_n_expert_ = 0;
    uint64_t grp_first_off_ = 0, grp_expert_off_ = 0, grp_stride_ = 0;
    /* Expert 0's per-slot payload. ExpertGroup promises slot_off[] and slot_bytes[] are fixed
     * across the experts of a layer "by construction", and this is where that is constructed:
     * an expert whose slot differs in size is refused at ITS OWN add_weight, not at the next
     * one's. The offset check alone cannot see it -- an oversized LAST expert lands at the right
     * base and simply overruns past base + n*stride. */
    std::vector<uint64_t> grp_slot_bytes_;
};

/* ================================================================== helpers for the tools */
/* "layer 12 / expert 7" and friends, so rad-info and every diagnostic name a unit the same way. */
std::string rad_unit_name(int32_t layer, int32_t expert);

}  /* namespace rad */
