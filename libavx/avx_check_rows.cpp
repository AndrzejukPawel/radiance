/* avx_check_rows.cpp -- embed_lookup_q over a FILE-BACKED table, the form a model serves it in.
 *
 * The geometry table checks the op against libref over a heap buffer, which the gather reads as
 * memory. A table in a container is not read that way: its rows come through a row cache every
 * thread shares, with direct reads of the file for the cache's misses and a reader thread a call
 * ahead (avx_ngram.cpp). What can go wrong there is WHICH bytes arrive -- a slot claimed twice, a
 * row decoded from a slot that was taken over, a wait on a claim that was given up -- and libref
 * has no such path to compare against, so the answer here is the file's own bytes.
 *
 * Two threads play the ranks of one call, as the engine's rank threads do: the same ids, rank 0
 * and 1 of 2, released together, so a thread claims rows, waits on rows the other has claimed, and
 * meets its own claim again when an id repeats within a call. Half the ids come from a small hot
 * set the cache keeps; the rest from a table four times the cache's 2^19 rows, so sets fill and
 * rows are evicted while the other thread may be reading them. Every call also hands over the NEXT
 * call's ids as `ahead`, as the engine's prefill does, so the reader thread claims and reads those
 * rows while this call's threads are still decoding theirs: a third thread in every race above.
 *
 * Every output row is compared bit for bit with that id's bytes in the file through the call's
 * scale, and a padding id must come back as a zero row. The row is 40 codes, so the AVX-512 decode
 * takes one full 32-code step and a masked tail. The table is a file in `dir`, and the filesystem
 * there decides how the gather aligns its direct reads: by the alignment the filesystem reports
 * (ext4, XFS), or by its block size where it reports none (btrfs, and a tmpfs since Linux 6.6). The
 * case says which it ran. */
#include "avx_harness.h"

#include <barrier>
#include <cerrno>
#include <cstring>
#include <thread>

#include <fcntl.h>
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

bool row_gather_case(const Lib& avx, const std::string& dir, int64_t calls, bool verbose,
                     bool bf16) {
    const RadKernelInfo* r = avx.row("embed_lookup_q");
    const char* what = bf16 ? "embed_lookup_q from a file, bf16 table" : "embed_lookup_q from a file";
    if (!r || !r->launch) {
        std::printf("  %s: no row\n", what);
        return false;
    }
    /* A bf16 table is twice the bytes a row and is read with no scale: its rows are their output,
     * so what this checks there is the cache and the reads at a row size the E4M3 table never
     * has. */
    /* The gather's misses are io_uring reads, and a sandbox can forbid io_uring outright -- Docker's
     * default seccomp profile does. There the gather refuses every call by design, so the case
     * says why it cannot run rather than counting a refusal per call. */
    {
        io_uring_params p{};
        const long ring = syscall(__NR_io_uring_setup, 1, &p);
        if (ring < 0) {
            std::printf("  %s: SKIPPED -- io_uring is unavailable here (%s)\n", what,
                        std::strerror(errno));
            return true;
        }
        ::close((int)ring);
    }
    const int64_t n = 40, V = int64_t(1) << 21, M = 512, hot = 4096;
    const int64_t rb = bf16 ? 2 * n : n;
    const float scale = 0.375f;

    const std::string path = dir + "/rad-avx-check-rows.bin";
    uint64_t x = 0x9e3779b97f4a7c15ull;
    auto rnd = [&x]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    {
        const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        if (fd < 0) { std::printf("  %s: cannot create %s\n", what, path.c_str());
                      return false; }
        std::vector<uint64_t> chunk(1 << 16);
        bool wrote = true;
        for (int64_t done = 0; done < V * rb && wrote; done += (int64_t)(chunk.size() * 8)) {
            for (auto& w : chunk) w = rnd();
            const size_t want = (size_t)std::min<int64_t>(V * rb - done, (int64_t)chunk.size() * 8);
            wrote = ::write(fd, chunk.data(), want) == (ssize_t)want;
        }
        wrote = wrote && ::fsync(fd) == 0;
        ::close(fd);
        if (!wrote) { ::unlink(path.c_str()); std::printf("  %s: short write\n", what);
                      return false; }
    }
    /* A filesystem without direct I/O -- one that refuses the open, as a tmpfs did before Linux
     * 6.6, or one that reports the file has none -- is one the gather refuses by design, so the
     * case cannot run there; it says so rather than counting a refusal per call. Every other
     * filesystem the gather has to read, with or without a reported alignment. */
    bool reported = false;
    {
        const int dfd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        struct statx sx {};
        const bool dio = dfd >= 0 &&
                         ::statx(dfd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &sx) == 0 &&
                         !((sx.stx_mask & STATX_DIOALIGN) && sx.stx_dio_offset_align == 0);
        reported = (sx.stx_mask & STATX_DIOALIGN) != 0;
        if (dfd >= 0) ::close(dfd);
        if (!dio) {
            ::unlink(path.c_str());
            std::printf("  %s: SKIPPED -- %s cannot do direct I/O\n", what, dir.c_str());
            return true;
        }
    }
    const char* granule = reported ? "the alignment the filesystem reports"
                                   : "no alignment reported, read by the block size";
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    void* map = fd < 0 ? MAP_FAILED : mmap(nullptr, (size_t)(V * rb), PROT_READ, MAP_SHARED, fd, 0);
    if (fd >= 0) ::close(fd);
    if (map == MAP_FAILED) { ::unlink(path.c_str()); std::printf("  %s: mmap\n", what);
                             return false; }
    const uint8_t* tab = static_cast<const uint8_t*>(map);

    std::vector<int32_t> hotset((size_t)hot);
    for (auto& h : hotset) h = (int32_t)(rnd() % (uint64_t)V);
    std::vector<int32_t> ids((size_t)(calls * M));
    for (int64_t c = 0; c < calls; ++c)
        for (int64_t m = 0; m < M; ++m) {
            const uint64_t v = rnd();
            int32_t& id = ids[(size_t)(c * M + m)];
            if (m > 0 && v % 29 == 0)  id = ids[(size_t)(c * M + m - 1)];   /* repeated in a call */
            else if (v % 16 == 0)      id = -1;                             /* padding */
            else if (v % 16 < 8)       id = hotset[(v >> 8) % (uint64_t)hot];
            else                       id = (int32_t)((v >> 8) % (uint64_t)V);
        }
    uint16_t lut[256];
    for (int b = 0; b < 256; ++b) lut[b] = rad_f32_to_bf16(rad_fp8e4m3_to_f32((uint8_t)b) * scale);

    auto tensor = [](void* p, uint32_t dt, std::initializer_list<int64_t> shape) {
        RadTensor t{};
        t.data = p;
        t.dtype = dt;
        t.rank = (uint32_t)shape.size();
        int i = 0;
        for (int64_t s : shape) t.shape[i++] = s;
        int64_t st = 1;
        for (int k = (int)t.rank - 1; k >= 0; --k) { t.stride[k] = st; st *= t.shape[k]; }
        return t;
    };

    std::barrier<> go(2);
    int64_t bad[2] = {0, 0}, refused[2] = {0, 0};
    auto rank = [&](int rk) {
        uint16_t sc = rad_f32_to_bf16(scale);
        std::vector<uint16_t> out((size_t)(M * n));
        std::vector<RadTensor> ts = {
            tensor(nullptr, RAD_I32, {M}),
            tensor(const_cast<uint8_t*>(tab), bf16 ? RAD_BF16 : RAD_F8E4M3, {V, n}),
            tensor(bf16 ? nullptr : &sc, RAD_BF16, {1}), tensor(out.data(), RAD_BF16, {M, n}),
            tensor(nullptr, RAD_I32, {M}) };
        RadParam ps[3] = {};
        ps[0].key = "M";       ps[0].kind = RAD_P_INT; ps[0].ival = M;
        ps[1].key = "n_embd";  ps[1].kind = RAD_P_INT; ps[1].ival = n;
        ps[2].key = "n_vocab"; ps[2].kind = RAD_P_INT; ps[2].ival = V;
        RadArgs a{};
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps; a.n_p = 3;
        a.rank = rk; a.world_size = 2;
        for (int64_t c = 0; c < calls; ++c) {
            ts[0].data = &ids[(size_t)(c * M)];
            /* The last call has nothing ahead of it, and says so with an absent operand. */
            ts[4].data = c + 1 < calls ? &ids[(size_t)((c + 1) * M)] : nullptr;
            std::fill(out.begin(), out.end(), (uint16_t)0xdead);
            go.arrive_and_wait();
            if (r->launch(&a, nullptr) != RAD_OK) { ++refused[rk]; continue; }
            for (int64_t m = 0; m < M; ++m) {
                const int32_t id = ids[(size_t)(c * M + m)];
                for (int64_t i = 0; i < n; ++i) {
                    uint16_t want = 0;
                    if (id >= 0 && bf16) std::memcpy(&want, tab + ((int64_t)id * n + i) * 2, 2);
                    else if (id >= 0)    want = lut[tab[(int64_t)id * n + i]];
                    if (out[(size_t)(m * n + i)] != want) { ++bad[rk]; break; }
                }
            }
        }
    };
    std::thread t0(rank, 0), t1(rank, 1);
    t0.join();
    t1.join();
    /* THE ADDRESS RANGE STAYS RESERVED. libavx resolves a table to its file once, by the address
     * range of its mapping, for the life of the process (avx_ngram.cpp, table_file) -- the
     * engine's container mapping is one. A case that unmapped here could have the next case's file
     * mapped at the same addresses, and the gather would read this file through the fd it kept:
     * every row of the next case wrong, on whichever run the kernel happens to reuse the range.
     * An inaccessible anonymous mapping over the same range frees the file and keeps the range. */
    if (mmap(map, (size_t)(V * rb), PROT_NONE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS |
             MAP_NORESERVE, -1, 0) == MAP_FAILED)
        munmap(map, (size_t)(V * rb));
    ::unlink(path.c_str());

    const bool pass = bad[0] + bad[1] + refused[0] + refused[1] == 0;
    if (verbose || !pass)
        std::printf("  %s in %s (%s): %lld calls x 2 ranks, rows wrong %lld/%lld, "
                    "calls refused %lld/%lld %s\n", what, dir.c_str(), granule, (long long)calls,
                    (long long)bad[0], (long long)bad[1], (long long)refused[0],
                    (long long)refused[1], pass ? "ok" : "FAIL");
    else
        std::printf("  %s in %s (%s): %lld calls x 2 ranks, every row exact\n", what, dir.c_str(),
                    granule, (long long)calls);
    return pass;
}
