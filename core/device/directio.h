/* directio.h -- the file tier's reader.
 *
 * spec §5.4: "a file tier read with O_DIRECT into pinned buffers. Predictable latency, no
 * page-cache double-copy, and reads land where the mover can DMA from them without a bounce."
 * This file is that approach, narrowed to the one thing the weight tier and the KV tier both need.
 *
 * Two reasons the page cache is the wrong place for this and not merely a redundant one:
 *
 *   - it is a second copy of bytes that are already going to a pinned buffer, paid on every read,
 *     of a store measured in gigabytes;
 *   - it evicts. A weight-tier read pulling a hundred megabytes through the page cache throws out
 *     the mapped `.rad` pool's own pages, and the cost of that shows up in decode as a stall
 *     nobody can trace back to the read that caused it.
 *
 * The alignment rules are O_DIRECT's: the file offset, the transfer length and the buffer address
 * must all be block-aligned. Everything rad_dev_alloc returns already is (host.cpp), and .rad
 * pads its movement units to RAD_ALIGN_UNIT (rad_format.h), so the mover's every read takes the
 * path with no bounce in it. The bounce exists for the other callers.
 */
#pragma once
#include <cstdint>

namespace rad {

enum {
    RAD_DIO_READ   = 1,
    RAD_DIO_WRITE  = 2,      /* implies read: a partial block is a read-modify-write */
    RAD_DIO_CREATE = 4,
    RAD_DIO_TRUNC  = 8,

    /* Open buffered when the filesystem refuses O_DIRECT, instead of failing the open.
     *
     * Off by default, and that default is the point. The weight tier and the prefix cache are
     * sized in gigabytes and read in megabyte runs; a silent buffered fallback turns both into a
     * page-cache pump that is three times slower and looks like nothing at all in a profile. An
     * operator who pointed --prefix-cache-dir at tmpfs wants to be told, not accommodated. The flag is
     * for tests, which have to run on whatever /tmp happens to be. */
    RAD_DIO_ALLOW_BUFFERED = 16,
};

class DirectFile {
public:
    DirectFile() = default;
    ~DirectFile();
    DirectFile(const DirectFile&)            = delete;
    DirectFile& operator=(const DirectFile&) = delete;

    /* RAD_OK, or a negative RAD_E_*. `flags` is the RAD_DIO_* set above. */
    int  open(const char* path, int flags);
    void close();

    bool    is_open() const { return fd_ >= 0; }
    /* Whether the open actually got O_DIRECT. False only under RAD_DIO_ALLOW_BUFFERED. */
    bool    direct() const  { return direct_; }
    int     fd() const      { return fd_; }
    int64_t size() const;

    /* Exact reads and writes: a short transfer is RAD_E_IO, never a partial success. A mover that
     * has to check a byte count on every extent is a mover with a branch on the step path, and a
     * caller that forgets to check gets half a weight. */
    int  read_at(void* dst, int64_t off, int64_t bytes);
    int  write_at(const void* src, int64_t off, int64_t bytes);

    /* The alignment every fast-path extent must satisfy. 4096 rather than the device's own logical
     * block size (which is 512 on plenty of hardware, and queryable via STATX_DIOALIGN on a recent
     * kernel): 4096 is a superset of every real requirement, it is RAD_ALIGN_UNIT, and accepting
     * 512-aligned extents would only widen the fast path for extents the .rad container never
     * produces. One constant, no per-filesystem behaviour to reproduce a bug against. */
    static constexpr int64_t alignment() { return 4096; }

private:
    int  bounce_ready();

    int      fd_      = -1;
    bool     direct_  = false;
    uint8_t* bounce_  = nullptr;
    int64_t  bounce_bytes_ = 0;
};

}  /* namespace rad */
