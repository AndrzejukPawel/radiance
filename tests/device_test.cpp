/* device_test.cpp -- the device layer against its own contract.
 *
 * The interesting tests here are the ordering ones. Allocation and memcpy are hard to get wrong;
 * what is easy to get wrong is a backend whose "async" calls quietly execute inline, because then
 * every ordering bug above it -- a missing rad_event_wait, a staging buffer reused before the
 * mover's copy landed -- passes on the host backend and corrupts weights on a card. So the event tests
 * are built so that an out-of-order execution produces wrong BYTES, not merely a suspicious
 * timestamp, and they run the race several times.
 */
#include "rad_test.h"

#include "rad_internal.h"
#include "device/device.h"
#include "device/directio.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

using rad::DirectFile;
using rad::RAD_DIO_READ;
using rad::RAD_DIO_WRITE;
using rad::RAD_DIO_CREATE;
using rad::RAD_DIO_TRUNC;
using rad::RAD_DIO_ALLOW_BUFFERED;

namespace {

struct Stream {
    RadStream s = nullptr;
    explicit Stream(int hi = 0) { rad_stream_create(&s, hi); }
    ~Stream() { if (s) rad_stream_destroy(s); }
};

struct Event {
    RadEvent e = nullptr;
    Event() { rad_event_create(&e); }
    ~Event() { if (e) rad_event_destroy(e); }
};

/* Deterministic filler, so a torn copy shows up as a mismatch at a known index. */
void fill(uint8_t* p, int64_t n, uint32_t seed) {
    uint32_t x = seed | 1u;
    for (int64_t i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        p[i] = (uint8_t)(x >> 24);
    }
}

std::string temp_path(const char* tag) {
    /* The working directory, not /tmp: /tmp is tmpfs on most systems and tmpfs refuses
     * O_DIRECT outright, so the test would only ever exercise the buffered fallback. ctest runs
     * from the build directory, which is on whatever the repository is on. */
    return rad::fmt("./rad_%s_test_%d.bin", tag, (int)getpid());
}

}  /* namespace */

/* ------------------------------------------------------------------ identity */

TEST(backend_reports_itself) {
    const char* name = rad::device_backend_name();
    CHECK(name && *name);
    CHECK_EQ(rad::device_is_host(), std::strcmp(name, "host") == 0);

    const int n = rad_dev_count();
    CHECK(n >= 1);

    RadDeviceProps p{};
    CHECK_OK(rad_dev_props(0, &p));
    CHECK(p.name[0] != '\0');
    CHECK(p.arch[0] != '\0');
    CHECK(p.vram_bytes > 0);
    CHECK(p.vram_free >= 0);
    CHECK(p.n_cu >= 1);
    CHECK(p.warp_size >= 1);
    CHECK_EQ(p.device_id, 0);
    if (rad::device_is_host()) {
        CHECK_EQ(p.is_host_backend, 1);
        CHECK_EQ(std::string(p.arch), std::string("host"));
    }
    /* Out of range is refused, not clamped: a rank index that came from a bad --tp is a
     * configuration bug and silently serving device 0 twice is how two ranks share one card. */
    CHECK(rad_dev_props(n, &p) < 0);
    CHECK(rad_dev_set(-1) < 0);
    CHECK_OK(rad_dev_set(0));
}

TEST(host_vram_budget_comes_from_the_environment) {
    /* A real card reports its own VRAM. Print the marker the other suites use so ctest scores
     * this Skipped rather than Passed: a bare return leaves no trace of the case at all. */
    if (!rad::device_is_host()) {
        std::fprintf(stderr, "  SKIP host_vram_budget_comes_from_the_environment: host backend only\n");
        return;
    }
    const char* old = std::getenv("RADIANCE_HOST_VRAM_MIB");
    const std::string saved = old ? old : "";

    setenv("RADIANCE_HOST_VRAM_MIB", "777", 1);
    RadDeviceProps p{};
    CHECK_OK(rad_dev_props(0, &p));
    CHECK_EQ(p.vram_bytes, (int64_t)777 << 20);

    unsetenv("RADIANCE_HOST_VRAM_MIB");
    CHECK_OK(rad_dev_props(0, &p));
    /* The default is derived from physical RAM, so all that can be asserted is that it is a real
     * budget the planner can plan against rather than a placeholder. */
    CHECK(p.vram_bytes > (int64_t)64 << 20);

    if (!saved.empty()) setenv("RADIANCE_HOST_VRAM_MIB", saved.c_str(), 1);
}

/* ------------------------------------------------------------------ allocation */

TEST(alloc_free_roundtrip) {
    const int kinds[] = { RAD_MEM_DEVICE, RAD_MEM_HOST_PINNED, RAD_MEM_HOST, RAD_MEM_HOST_MAPPED };
    const int64_t n = 64 * 1024 + 17;   /* deliberately not a page multiple */

    for (int k : kinds) {
        void* p = rad_dev_alloc(n, k);
        CHECK(p != nullptr);
        if (!p) continue;

        /* Host-visible kinds are writable from here; a DEVICE allocation is not, on a real card. */
        if (rad::device_is_host() || k != RAD_MEM_DEVICE) {
            auto* b = (uint8_t*)p;
            fill(b, n, (uint32_t)k + 5u);
            uint8_t expect[64];
            fill(expect, 64, (uint32_t)k + 5u);
            CHECK_EQ(std::memcmp(b, expect, 64), 0);
        }
        if (k == RAD_MEM_HOST_MAPPED) {
            /* The zero-copy site of spec §5.1: one allocation, addressable from both sides. */
            CHECK(rad_dev_host_ptr(p) != nullptr);
            CHECK(rad_dev_device_ptr(p) != nullptr);
            /* AND THE TWO VIEWS ARE THE SAME ADDRESS, ON EVERY BACKEND, because the runtime
             * depends on it: Ctx::bind_buffers stores ONE base pointer per buffer, and the host
             * activation arena is RAD_MEM_HOST_MAPPED precisely so a host-site op and a device
             * kernel can be handed the same buffer. Under HIP's unified addressing they coincide;
             * where they would not, host-site execution cannot work and Ctx refuses to start.
             * This is the cheaper place to find that out than a model that will not load. */
            CHECK_EQ(rad_dev_host_ptr(p), rad_dev_device_ptr(p));
            CHECK_EQ(rad_dev_host_ptr(p), p);
        }
        /* Every host-visible pool this layer hands out is page-aligned, which is what lets a
         * pinned buffer be an O_DIRECT landing zone with no bounce (directio.h). A device
         * allocation only has to clear RAD_ALIGN_SUB, the container's sub-block. */
        CHECK_EQ((uintptr_t)p % (k == RAD_MEM_DEVICE && !rad::device_is_host() ? 256u : 4096u), 0u);
        rad_dev_free(p, k);
    }

    CHECK(rad_dev_alloc(0, RAD_MEM_HOST) == nullptr);
    CHECK(rad_dev_alloc(-1, RAD_MEM_HOST) == nullptr);
    CHECK(rad_dev_alloc(16, 99) == nullptr);
    rad_dev_free(nullptr, RAD_MEM_HOST);   /* must not fault */
    CHECK(rad_dev_host_ptr(nullptr) == nullptr);
}

TEST(device_allocation_moves_vram_free) {
    /* On a card, other processes move it too. */
    if (!rad::device_is_host()) {
        std::fprintf(stderr, "  SKIP device_allocation_moves_vram_free: host backend only\n");
        return;
    }
    RadDeviceProps before{}, during{};
    CHECK_OK(rad_dev_props(0, &before));
    const int64_t n = 32 << 20;
    void* p = rad_dev_alloc(n, RAD_MEM_DEVICE);
    CHECK(p != nullptr);
    CHECK_OK(rad_dev_props(0, &during));
    /* The planner's "did it fit" path is one of the things the host backend exists to falsify, and
     * a constant vram_free would let a plan that overcommits a real card pass here. */
    CHECK_EQ(before.vram_free - during.vram_free, n);
    rad_dev_free(p, RAD_MEM_DEVICE);
    CHECK_OK(rad_dev_props(0, &during));
    CHECK_EQ(before.vram_free, during.vram_free);
}

/* ------------------------------------------------------------------ transfers */

TEST(memcpy_async_then_stream_sync) {
    const int64_t n = 4 << 20;
    Stream st;
    CHECK(st.s != nullptr);

    auto* src = (uint8_t*)rad_dev_alloc(n, RAD_MEM_HOST_PINNED);
    auto* dst = (uint8_t*)rad_dev_alloc(n, RAD_MEM_HOST);
    CHECK(src && dst);
    if (!src || !dst) return;

    fill(src, n, 0xC0FFEEu);
    std::memset(dst, 0, (size_t)n);

    CHECK_OK(rad_memcpy_async(dst, src, n, st.s));
    CHECK_OK(rad_stream_sync(st.s));
    CHECK_EQ(std::memcmp(dst, src, (size_t)n), 0);

    /* Zero bytes is a no-op, not an error: a batch with an empty span should not need a branch. */
    CHECK_OK(rad_memcpy_async(dst, src, 0, st.s));

    rad_dev_free(src, RAD_MEM_HOST_PINNED);
    rad_dev_free(dst, RAD_MEM_HOST);
}

TEST(memset_async) {
    const int64_t n = 1 << 20;
    Stream st;
    /* Pinned, not RAD_MEM_HOST. rad_memset_async needs a device-ACCESSIBLE destination: pageable
     * host memory is memory the GPU has never been shown, and hipMemsetAsync refuses it. The host
     * backend cannot reproduce that refusal, so testing it against pageable memory here would
     * green-light code that fails on a card -- which is exactly the divergence the shared vtable
     * exists to prevent. See core/device/device.h. */
    auto* p = (uint8_t*)rad_dev_alloc(n, RAD_MEM_HOST_PINNED);
    CHECK(p != nullptr);
    if (!p) return;

    std::memset(p, 0, (size_t)n);
    CHECK_OK(rad_memset_async(p, 0x5A, n, st.s));
    CHECK_OK(rad_stream_sync(st.s));
    bool all = true;
    for (int64_t i = 0; i < n; ++i) if (p[i] != 0x5A) { all = false; break; }
    CHECK(all);

    /* Only the low byte of `value` is written, matching hipMemsetAsync and memset. */
    CHECK_OK(rad_memset_async(p, 0x1FF, n, st.s));
    CHECK_OK(rad_stream_sync(st.s));
    CHECK_EQ((int)p[0], 0xFF);
    CHECK_EQ((int)p[n - 1], 0xFF);

    rad_dev_free(p, RAD_MEM_HOST_PINNED);
}

TEST(memset_of_pageable_host_memory_is_refused_on_a_card) {
    /* The one asymmetry between the backends, pinned here so it is a documented contract rather
     * than something rediscovered on a card. On a card the destination must be
     * device-accessible and RAD_MEM_HOST is not; on the host backend every kind is the same
     * memory and there is nothing to refuse. Either way the message names the rule. */
    Stream st;
    auto* p = (uint8_t*)rad_dev_alloc(4096, RAD_MEM_HOST);
    CHECK(p != nullptr);
    if (!p) return;
    int s = rad_memset_async(p, 0, 4096, st.s);
    if (rad::device_is_host()) {
        CHECK_OK(s);
        CHECK_OK(rad_stream_sync(st.s));
    } else {
        CHECK_EQ(s, RAD_E_UNSUPPORTED);
        CHECK(std::strstr(rad_dev_last_error(), "device-accessible") != nullptr);
    }
    rad_dev_free(p, RAD_MEM_HOST);
}

TEST(memcpy_2d_respects_the_pitch) {
    /* A KV block's rows are strided by the pool's row pitch, not packed, so the padding between
     * rows is somebody else's data and must survive the copy untouched. */
    const int64_t W = 48, H = 7, DP = 64, SP = 55;
    Stream st;
    auto* s = (uint8_t*)rad_dev_alloc(SP * H, RAD_MEM_HOST_PINNED);
    auto* d = (uint8_t*)rad_dev_alloc(DP * H, RAD_MEM_HOST_PINNED);
    CHECK(s && d);
    if (!s || !d) return;

    fill(s, SP * H, 0xBEEF);
    CHECK_OK(rad_memset_async(d, 0xCC, DP * H, st.s));
    CHECK_OK(rad_memcpy_2d_async(d, DP, s, SP, W, H, st.s));
    CHECK_OK(rad_stream_sync(st.s));

    bool rows_ok = true, pad_ok = true;
    for (int64_t r = 0; r < H; ++r) {
        if (std::memcmp(d + r * DP, s + r * SP, (size_t)W) != 0) rows_ok = false;
        for (int64_t c = W; c < DP; ++c) if (d[r * DP + c] != 0xCC) pad_ok = false;
    }
    CHECK(rows_ok);
    CHECK(pad_ok);

    /* A pitch shorter than the row is a caller bug and is refused rather than clamped. */
    CHECK(rad_memcpy_2d_async(d, W - 1, s, SP, W, H, st.s) < 0);

    rad_dev_free(s, RAD_MEM_HOST_PINNED);
    rad_dev_free(d, RAD_MEM_HOST_PINNED);
}

/* THREE LAPS OF THE KERNEL ARGUMENT RING WITH NO HOST WAIT BETWEEN THEM. A device backend that
 * keeps kernel arguments in memory the host rewrites has to make each slot's new contents visible
 * to the kernel that reads them; a stale slot does not fail, it runs the previous lap's kernel
 * again. Every copy here moves a different 64-byte piece to a different place, so a stale
 * argument lands a piece where another belonged and the comparison names it. */
TEST(twelve_thousand_copies_each_land_where_their_own_arguments_say) {
    const int64_t piece = 64, n = 12000, bytes = piece * n;
    Stream st;
    auto* host = (uint8_t*)rad_dev_alloc(bytes, RAD_MEM_HOST_PINNED);
    auto* back = (uint8_t*)rad_dev_alloc(bytes, RAD_MEM_HOST_PINNED);
    auto* src  = (uint8_t*)rad_dev_alloc(bytes, RAD_MEM_DEVICE);
    auto* dst  = (uint8_t*)rad_dev_alloc(bytes, RAD_MEM_DEVICE);
    CHECK(host && back && src && dst);
    if (!host || !back || !src || !dst) return;

    fill(host, bytes, 0xA11CE);
    CHECK_OK(rad_memcpy_async(src, host, bytes, st.s));
    /* Piece i of dst comes from piece (i * 7919) mod n of src: a permutation, since 7919 is prime
     * and does not divide n, so no two copies share an argument. */
    for (int64_t i = 0; i < n; ++i)
        CHECK_OK(rad_memcpy_async(dst + i * piece, src + ((i * 7919) % n) * piece, piece, st.s));
    CHECK_OK(rad_memcpy_async(back, dst, bytes, st.s));
    CHECK_OK(rad_stream_sync(st.s));

    int64_t wrong = 0, first = -1;
    for (int64_t i = 0; i < n; ++i)
        if (std::memcmp(back + i * piece, host + ((i * 7919) % n) * piece, (size_t)piece) != 0) {
            if (first < 0) first = i;
            ++wrong;
        }
    CHECK_EQ(wrong, (int64_t)0);
    if (wrong) std::fprintf(stderr, "    first misplaced piece: %lld\n", (long long)first);

    rad_dev_free(host, RAD_MEM_HOST_PINNED);
    rad_dev_free(back, RAD_MEM_HOST_PINNED);
    rad_dev_free(src, RAD_MEM_DEVICE);
    rad_dev_free(dst, RAD_MEM_DEVICE);
}

/* ------------------------------------------------------------------ ordering */

TEST(event_on_one_stream_orders_work_on_another) {
    const int64_t n = 4 << 20;
    const int     chain = 8;    /* copies stream A does before it records */
    const int     rounds = 3;   /* the race, run more than once */

    Stream a, b;
    CHECK(a.s && b.s);

    auto* src = (uint8_t*)rad_dev_alloc(n, RAD_MEM_HOST_PINNED);
    auto* mid = (uint8_t*)rad_dev_alloc(n, RAD_MEM_DEVICE);
    auto* dst = (uint8_t*)rad_dev_alloc(n, RAD_MEM_HOST);
    CHECK(src && mid && dst);
    if (!src || !mid || !dst) return;

    for (int round = 0; round < rounds; ++round) {
        Event e_a, e_b;
        CHECK(e_a.e && e_b.e);

        fill(src, n, 0x51ED0000u + (uint32_t)round);

        /* Poison the intermediate from a *synchronised* point, so a read that jumps the queue sees
         * 0xEE and not a stale copy of the right answer. */
        CHECK_OK(rad_memset_async(mid, 0xEE, n, b.s));
        CHECK_OK(rad_stream_sync(b.s));

        /* Stream A: enough work that an unordered reader on B lands in the middle of it. */
        CHECK_OK(rad_memset_async(mid, 0x00, n, a.s));
        for (int i = 0; i < chain; ++i) CHECK_OK(rad_memcpy_async(mid, src, n, a.s));
        CHECK_OK(rad_event_record(e_a.e, a.s));

        /* Stream B waits on the event, not on the host: no rad_stream_sync(a) here, which is the
         * property spec §5.4 needs -- a transfer that involves no host synchronisation. */
        CHECK_OK(rad_event_wait(b.s, e_a.e));
        CHECK_OK(rad_memcpy_async(dst, mid, n, b.s));
        CHECK_OK(rad_event_record(e_b.e, b.s));

        CHECK_OK(rad_stream_sync(b.s));
        CHECK_EQ(std::memcmp(dst, src, (size_t)n), 0);

        /* The timestamps are taken when each stream reaches its record, so B's cannot precede A's
         * unless the wait did nothing. */
        float ms = -1.0f;
        CHECK_OK(rad_event_elapsed_ms(e_a.e, e_b.e, &ms));
        CHECK(ms >= 0.0f);

        CHECK_OK(rad_stream_sync(a.s));
        CHECK_EQ(rad_event_query(e_a.e), 1);
        CHECK_EQ(rad_event_query(e_b.e), 1);
    }

    rad_dev_free(src, RAD_MEM_HOST_PINNED);
    rad_dev_free(mid, RAD_MEM_DEVICE);
    rad_dev_free(dst, RAD_MEM_HOST);
}

/* TWO STREAMS THAT JOIN EACH OTHER HUNDREDS OF TIMES, with the host far ahead of both and one of
 * them lagging -- the shape of a model's two compute lanes, which meet every layer. A backend that
 * reuses an event's completion object while a wait enqueued against its previous record has not yet
 * been reached turns that wait into a wait on a LATER record, which sits behind a join the lagging
 * stream has not reached: a deadlock, not a wrong answer. Here it would hang; the bytes are checked
 * as well, because the joins are also what orders the copies. */
TEST(two_streams_joining_each_other_hundreds_of_times_neither_deadlock_nor_reorder) {
    const int64_t big = 1 << 20, small = 4096;
    const int     joins = 400;
    Stream a, b;
    Event ea, eb;
    auto* d0 = (uint8_t*)rad_dev_alloc(big, RAD_MEM_DEVICE);
    auto* d1 = (uint8_t*)rad_dev_alloc(big, RAD_MEM_DEVICE);
    auto* h  = (uint8_t*)rad_dev_alloc(big, RAD_MEM_HOST_PINNED);
    auto* r  = (uint8_t*)rad_dev_alloc(small, RAD_MEM_HOST_PINNED);
    CHECK(d0 && d1 && h && r);
    if (!d0 || !d1 || !h || !r) return;
    fill(h, big, 0x5EED);
    CHECK_OK(rad_memcpy_async(d0, h, big, a.s));
    for (int i = 0; i < joins; ++i) {
        /* b lags: a whole megabyte each round against a's four kilobytes. */
        CHECK_OK(rad_event_record(ea.e, a.s));
        CHECK_OK(rad_event_wait(b.s, ea.e));
        CHECK_OK(rad_memcpy_async(d1, d0, big, b.s));
        CHECK_OK(rad_event_record(eb.e, b.s));
        CHECK_OK(rad_event_wait(a.s, eb.e));
        CHECK_OK(rad_memcpy_async(d0, d1, small, a.s));
    }
    CHECK_OK(rad_memcpy_async(r, d0, small, a.s));
    CHECK_OK(rad_stream_sync(a.s));
    CHECK_OK(rad_stream_sync(b.s));
    CHECK_EQ(std::memcmp(r, h, (size_t)small), 0);
    rad_dev_free(d0, RAD_MEM_DEVICE);
    rad_dev_free(d1, RAD_MEM_DEVICE);
    rad_dev_free(h, RAD_MEM_HOST_PINNED);
    rad_dev_free(r, RAD_MEM_HOST_PINNED);
}

TEST(an_event_destroyed_while_queued_work_names_it_still_orders_that_work) {
    /* Teardown written against a card destroys events whose record or wait has not run yet --
     * the mover's own does -- and HIP allows it. The work already queued has to behave as if the
     * handle were still alive: the wait still waits, the copy behind it still sees the bytes, and
     * neither stream wedges. The events are recreated and destroyed again between rounds so that
     * memory freed too early is handed straight back out and overwritten. */
    const int64_t n = 4 << 20;
    Stream a, b;
    CHECK(a.s && b.s);
    auto* src = (uint8_t*)rad_dev_alloc(n, RAD_MEM_HOST_PINNED);
    auto* mid = (uint8_t*)rad_dev_alloc(n, RAD_MEM_DEVICE);
    auto* dst = (uint8_t*)rad_dev_alloc(n, RAD_MEM_HOST);
    CHECK(src && mid && dst);
    if (!src || !mid || !dst) return;

    for (int round = 0; round < 4; ++round) {
        fill(src, n, 0xDE570000u + (uint32_t)round);
        CHECK_OK(rad_memset_async(mid, 0xEE, n, b.s));
        CHECK_OK(rad_stream_sync(b.s));

        RadEvent ga = nullptr, eb = nullptr;
        CHECK_OK(rad_event_create(&ga));
        CHECK_OK(rad_event_create(&eb));
        for (int i = 0; i < 8; ++i) CHECK_OK(rad_memcpy_async(mid, src, n, a.s));
        CHECK_OK(rad_event_record(ga, a.s));
        CHECK_OK(rad_event_wait(b.s, ga));
        CHECK_OK(rad_memcpy_async(dst, mid, n, b.s));
        CHECK_OK(rad_event_record(eb, b.s));
        rad_event_destroy(eb);
        rad_event_destroy(ga);
        for (int k = 0; k < 4; ++k) {
            Event churn;
            CHECK_OK(rad_event_record(churn.e, b.s));
        }

        CHECK_OK(rad_stream_sync(b.s));
        CHECK_OK(rad_stream_sync(a.s));
        CHECK_EQ(std::memcmp(dst, src, (size_t)n), 0);
    }

    rad_dev_free(src, RAD_MEM_HOST_PINNED);
    rad_dev_free(mid, RAD_MEM_DEVICE);
    rad_dev_free(dst, RAD_MEM_HOST);
}

TEST(waiting_on_an_unrecorded_event_is_a_no_op) {
    /* The mover relies on this: a weight with no transfer in flight has an event nobody recorded,
     * and making that an error would put a branch on the issue path for nothing. */
    Stream st;
    Event e;
    CHECK_EQ(rad_event_query(e.e), 1);
    CHECK_OK(rad_event_wait(st.s, e.e));
    CHECK_OK(rad_stream_sync(st.s));
}

TEST(the_null_stream_is_refused) {
    /* HIP's stream 0 serialises against every other blocking stream in the process, which is the
     * opposite of what a transfer is for here. Refusing it on both backends means a caller cannot
     * pick it up by accident on the host backend and discover it on a card. */
    uint8_t a[16] = {}, b[16] = {};
    CHECK(rad_memcpy_async(a, b, sizeof a, nullptr) < 0);
    CHECK(std::strlen(rad_dev_last_error()) > 0);
    /* Drained by reading, per the header. */
    CHECK_EQ(std::strlen(rad_dev_last_error()), 0u);

    CHECK(rad_memset_async(a, 0, sizeof a, nullptr) < 0);
    CHECK(rad_stream_sync(nullptr) < 0);
}

/* ------------------------------------------------------------------ the file tier */

TEST(directfile_write_then_read_roundtrip) {
    const std::string path = temp_path("dio");
    const int64_t A = DirectFile::alignment();
    const int64_t body = 3 * A, tail = 123;

    DirectFile f;
    CHECK_OK(f.open(path.c_str(), RAD_DIO_READ | RAD_DIO_WRITE | RAD_DIO_CREATE | RAD_DIO_TRUNC |
                                  RAD_DIO_ALLOW_BUFFERED));
    if (!f.is_open()) return;
    if (!f.direct())
        fprintf(stderr, "       (note: %s has no O_DIRECT; the buffered fallback is under test)\n",
                path.c_str());
    CHECK_EQ(f.size(), (int64_t)0);

    /* A pinned buffer is what the mover reads into, so that is what the aligned path is tested
     * with: page-aligned address, aligned offset, aligned length, and therefore one pread with no
     * copy anywhere in between. */
    auto* w = (uint8_t*)rad_dev_alloc(body, RAD_MEM_HOST_PINNED);
    auto* r = (uint8_t*)rad_dev_alloc(body, RAD_MEM_HOST_PINNED);
    CHECK(w && r);
    if (!w || !r) { f.close(); ::unlink(path.c_str()); return; }
    fill(w, body, 0xD1EC7);

    CHECK_OK(f.write_at(w, 0, body));
    CHECK_EQ(f.size(), body);

    /* The misaligned tail: an unaligned length, from an unaligned stack buffer. This is the only
     * bounce in the read path and it exists because O_DIRECT will EINVAL the alternative. */
    uint8_t t[tail];
    fill(t, tail, 0x7A11);
    CHECK_OK(f.write_at(t, body, tail));
    CHECK_EQ(f.size(), body + tail);

    std::memset(r, 0, (size_t)body);
    CHECK_OK(f.read_at(r, 0, body));
    CHECK_EQ(std::memcmp(r, w, (size_t)body), 0);

    uint8_t t2[tail];
    std::memset(t2, 0, sizeof t2);
    CHECK_OK(f.read_at(t2, body, tail));
    CHECK_EQ(std::memcmp(t2, t, sizeof t2), 0);

    /* A slice that is aligned at neither end and straddles two blocks. */
    uint8_t slice[2000];
    CHECK_OK(f.read_at(slice, 1000, sizeof slice));
    CHECK_EQ(std::memcmp(slice, w + 1000, sizeof slice), 0);

    /* One byte past the end is RAD_E_IO, not a short read: read_at fills exactly what was asked
     * for or fails, so no caller has to check a byte count on the step path. */
    CHECK(f.read_at(t2, body, tail + 1) < 0);
    CHECK(f.read_at(r, body + tail + A, A) < 0);

    /* Reopen without O_TRUNC: the bytes are on disk, not in this handle. */
    f.close();
    CHECK_OK(f.open(path.c_str(), RAD_DIO_READ | RAD_DIO_ALLOW_BUFFERED));
    CHECK_EQ(f.size(), body + tail);
    std::memset(r, 0, (size_t)body);
    CHECK_OK(f.read_at(r, 0, body));
    CHECK_EQ(std::memcmp(r, w, (size_t)body), 0);
    f.close();

    CHECK(f.read_at(r, 0, A) < 0);          /* closed is RAD_E_STATE, not a crash */
    CHECK_EQ(f.size(), (int64_t)-1);

    rad_dev_free(w, RAD_MEM_HOST_PINNED);
    rad_dev_free(r, RAD_MEM_HOST_PINNED);
    ::unlink(path.c_str());
}

TEST(directfile_refuses_a_missing_file) {
    DirectFile f;
    CHECK(f.open("/nonexistent/rad/dio/path.bin", RAD_DIO_READ | RAD_DIO_ALLOW_BUFFERED) < 0);
    CHECK(!f.is_open());
    CHECK(f.open(nullptr, RAD_DIO_READ) < 0);
}

/* ------------------------------------------------------------------ elastic memory
 *
 * WHAT THESE ARE HOLDING DOWN. An elastic pool exists so two pools can trade VRAM -- the KV cache
 * gives back what no session is holding and the expert slab takes it -- and the trade is only safe
 * if three things are exactly true: an address never moves, a granule the caller still wants is
 * never unmapped, and a range that comes back is backed by real pages rather than merely listed as
 * backed. Each of the three fails silently in its own way. A moved address is a kernel reading
 * another group's blocks; a granule unmapped under live data is a fault at an address that was
 * valid a moment ago; a range that is listed but not mapped is a fault the first time a sequence
 * writes past the old boundary, under load, long after the commit that was supposed to back it.
 *
 * The interval arithmetic is the whole of the risk and none of it needs a card, so it is held down
 * here against the host backend, where a decommitted page can also be PROVEN gone by reading it
 * back as zero.
 */

TEST(elastic_memory_takes_address_space_and_no_pages) {
    if (!rad::VMem::supported()) {
        std::fprintf(stderr, "  SKIP elastic_memory_takes_address_space_and_no_pages: "
                             "no virtual memory management on this backend\n");
        return;
    }
    const int64_t g = rad::VMem::granularity();
    CHECK(g > 0);
    CHECK_EQ(g % 4096, 0);              /* a granule is whole pages, whatever else it is */

    rad::VMem v;
    CHECK_OK(v.reserve(8 * g));
    CHECK(v.base() != nullptr);
    CHECK_EQ(v.size(), 8 * g);
    CHECK_EQ(v.committed(), 0);         /* reserved is not allocated: this is the whole point */
    CHECK_EQ((int64_t)((uintptr_t)v.base() % (uintptr_t)g), 0);

    /* A SIZE THAT IS NOT A WHOLE NUMBER OF GRANULES ROUNDS UP. The last partial granule can never
     * be committed, so leaving it out of the range would reserve bytes nothing could ever back. */
    rad::VMem odd;
    CHECK_OK(odd.reserve(g + 1));
    CHECK_EQ(odd.size(), 2 * g);

    CHECK_EQ(v.reserve(0), RAD_E_INVAL);
    CHECK_EQ(v.size(), 0);              /* and a refused reserve released what it held */
}

TEST(a_committed_range_is_backed_and_keeps_its_address) {
    if (!rad::VMem::supported() || !rad::device_is_host()) {
        std::fprintf(stderr, "  SKIP a_committed_range_is_backed_and_keeps_its_address: "
                             "needs host-visible elastic memory\n");
        return;
    }
    const int64_t g = rad::VMem::granularity();
    rad::VMem v;
    CHECK_OK(v.reserve(8 * g));
    void* const addr = v.base();

    CHECK_OK(v.commit(0, 2 * g));
    CHECK_EQ(v.committed(), 2 * g);
    CHECK(v.is_committed(0, 2 * g));
    CHECK(!v.is_committed(0, 3 * g));

    fill((uint8_t*)v.base(), 2 * g, 0x5eedu);
    uint8_t expect[128];
    fill(expect, sizeof expect, 0x5eedu);
    CHECK_EQ(std::memcmp(v.base(), expect, sizeof expect), 0);

    /* Growing at the far end must not disturb what is already there, and must not move the base:
     * every block index the pool has handed out is an offset from it. */
    CHECK_OK(v.commit(6 * g, 2 * g));
    CHECK_EQ(v.committed(), 4 * g);
    CHECK_EQ(v.base(), addr);
    CHECK_EQ(std::memcmp(v.base(), expect, sizeof expect), 0);
}

TEST(commit_rounds_outward_and_decommit_rounds_inward) {
    if (!rad::VMem::supported()) {
        std::fprintf(stderr, "  SKIP commit_rounds_outward_and_decommit_rounds_inward: "
                             "no virtual memory management on this backend\n");
        return;
    }
    const int64_t g = rad::VMem::granularity();
    rad::VMem v;
    CHECK_OK(v.reserve(4 * g));

    /* One byte in the middle of a granule needs the whole granule, or the byte is not backed. */
    CHECK_OK(v.commit(g / 2, 1));
    CHECK_EQ(v.committed(), g);
    CHECK(v.is_committed(0, g));

    /* And a decommit that does not cover a whole granule takes nothing, because the rest of that
     * granule is data somebody still owns. */
    CHECK_OK(v.decommit(g / 2, 1));
    CHECK_EQ(v.committed(), g);
    CHECK_OK(v.decommit(0, g - 1));
    CHECK_EQ(v.committed(), g);
    CHECK_OK(v.decommit(0, g));
    CHECK_EQ(v.committed(), 0);
}


TEST(the_commit_quantum_is_the_pool_alignment_on_every_backend) {
    if (!rad::VMem::supported()) {
        std::fprintf(stderr, "  SKIP the_commit_quantum_is_the_pool_alignment_on_every_backend: "
                             "no virtual memory management on this backend\n");
        return;
    }
    /* WHAT A TEST PINS HERE HAS TO BE WHAT THE CARD RUNS, and the granule is what decides that.
     * A driver's own answer is a floor -- one reports a single host page -- and a backend that
     * took it would grow in pieces the host suite never exercises and pay a driver call for each
     * of them. Both backends take RAD_ALIGN_POOL instead, which every pool is already aligned to,
     * so a commit arithmetic checked on a machine with no card is the commit arithmetic a card
     * performs. */
    const int64_t g = rad::VMem::granularity();
    CHECK(g >= (int64_t)RAD_ALIGN_POOL);
    CHECK_EQ(g % (int64_t)RAD_ALIGN_POOL, 0);
}

TEST(committing_over_a_commit_maps_only_the_gap) {
    if (!rad::VMem::supported()) {
        std::fprintf(stderr, "  SKIP committing_over_a_commit_maps_only_the_gap: "
                             "no virtual memory management on this backend\n");
        return;
    }
    const int64_t g = rad::VMem::granularity();
    rad::VMem v;
    CHECK_OK(v.reserve(8 * g));

    /* A pool that grows a granule at a time asks for [0,n) every time it changes n. The driver
     * refuses a map that overlaps a map, so the overlap has to be subtracted here -- and this is
     * the case that would otherwise work in every test that grows from nothing and fail the first
     * time a live pool grew twice. */
    CHECK_OK(v.commit(0, 2 * g));
    CHECK_OK(v.commit(0, 5 * g));
    CHECK_EQ(v.committed(), 5 * g);
    CHECK_OK(v.commit(0, 5 * g));
    CHECK_EQ(v.committed(), 5 * g);

    /* Two islands and then the bridge between them. */
    rad::VMem w;
    CHECK_OK(w.reserve(8 * g));
    CHECK_OK(w.commit(0, g));
    CHECK_OK(w.commit(4 * g, g));
    CHECK_EQ(w.committed(), 2 * g);
    CHECK_OK(w.commit(0, 5 * g));
    CHECK_EQ(w.committed(), 5 * g);
    CHECK(w.is_committed(0, 5 * g));

    /* Past the end clamps rather than failing: the range is the maximum by construction and a
     * caller asking for all of it is asking for what is left. */
    CHECK_OK(w.commit(7 * g, 4 * g));
    CHECK_EQ(w.committed(), 6 * g);
}

TEST(decommitting_the_middle_leaves_the_ends) {
    if (!rad::VMem::supported()) {
        std::fprintf(stderr, "  SKIP decommitting_the_middle_leaves_the_ends: "
                             "no virtual memory management on this backend\n");
        return;
    }
    const int64_t g = rad::VMem::granularity();
    rad::VMem v;
    CHECK_OK(v.reserve(8 * g));
    CHECK_OK(v.commit(0, 4 * g));
    CHECK_OK(v.decommit(g, g));

    CHECK_EQ(v.committed(), 3 * g);
    CHECK(v.is_committed(0, g));
    CHECK(!v.is_committed(g, g));
    CHECK(v.is_committed(2 * g, 2 * g));
    /* And the span across the hole is not committed, which is the question a caller actually asks
     * before handing a range to a kernel. */
    CHECK(!v.is_committed(0, 4 * g));

    /* The hole fills back in and the extents merge, so the next question is answered by one
     * comparison rather than by a list that grows for the life of the process. */
    CHECK_OK(v.commit(g, g));
    CHECK_EQ(v.committed(), 4 * g);
    CHECK(v.is_committed(0, 4 * g));
}

TEST(a_recommitted_range_comes_back_zeroed) {
    if (!rad::VMem::supported() || !rad::device_is_host()) {
        std::fprintf(stderr, "  SKIP a_recommitted_range_comes_back_zeroed: "
                             "needs host-visible elastic memory\n");
        return;
    }
    const int64_t g = rad::VMem::granularity();
    rad::VMem v;
    CHECK_OK(v.reserve(4 * g));
    CHECK_OK(v.commit(0, 2 * g));

    fill((uint8_t*)v.base(), 2 * g, 0xabcdu);
    CHECK((( uint8_t*)v.base())[g] != 0);

    /* THE PAGES REALLY WENT BACK. A decommit that merely forgot the range would leave the old
     * bytes in place and the VRAM spent, and every test above would still pass -- so the proof
     * that the memory was returned is that what comes back is not what was there. */
    CHECK_OK(v.decommit(0, 2 * g));
    CHECK_EQ(v.committed(), 0);
    CHECK_OK(v.commit(0, 2 * g));
    const uint8_t* b = (const uint8_t*)v.base();
    for (int64_t i = 0; i < 2 * g; i += 4093) CHECK_EQ((int)b[i], 0);
}

TEST(elastic_memory_is_accounted_and_released_whole) {
    if (!rad::VMem::supported()) {
        std::fprintf(stderr, "  SKIP elastic_memory_is_accounted_and_released_whole: "
                             "no virtual memory management on this backend\n");
        return;
    }
    const int64_t g = rad::VMem::granularity();
    const int64_t before = rad::rad_dev_vmem_committed();
    {
        rad::VMem v;
        CHECK_OK(v.reserve(8 * g));
        CHECK_OK(v.commit(0, 3 * g));
        CHECK_EQ(rad::rad_dev_vmem_committed() - before, 3 * g);
        CHECK_OK(v.decommit(0, g));
        CHECK_EQ(rad::rad_dev_vmem_committed() - before, 2 * g);
        /* Destruction releases what is still committed. A range freed with pages in it leaks the
         * pages, and on some drivers refuses the address free as well -- which is why the order is
         * fixed inside release() rather than asked of every caller. */
    }
    CHECK_EQ(rad::rad_dev_vmem_committed(), before);
}

RAD_TEST_MAIN()
