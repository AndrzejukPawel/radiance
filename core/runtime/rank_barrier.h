/* rank_barrier.h -- the release point for the tensor-parallel rank threads.
 *
 * WHY THIS IS NOT IPC. Radiance is one process. One thread per rank, each owning one device
 * (spec §1). The ranks therefore share an address space, so the step batch is shared BY POINTER
 * rather than broadcast: there is nothing to serialise, no shared-memory segment to map, no file
 * descriptor, no pickling, and no handshake protocol that can get out of step with itself. All
 * that is left for the barrier to do is establish happens-before between the scheduler's release
 * and each rank's first rad_issue, and between the ranks' last issue and the scheduler's next
 * batch build.
 *
 * A futex round trip through a condvar is ~1-2 us on this class of machine. The multiprocessing
 * equivalent -- serialise the batch, write it down a pipe, wake N processes, have each one
 * deserialise it -- is tens of microseconds per step before any work happens, and it is most of
 * what a Python engine's multiprocessing budget actually buys. That difference is the reason §1
 * is written the way it is, and this class is the whole of the cost on our side.
 *
 * Sense reversal, not a counter reset, so a fast rank that comes back round for the next step
 * cannot slip through the barrier its slower peers have not left yet.
 */
#pragma once
#include "../rad_core.h"

#include <condition_variable>
#include <mutex>

namespace rad {

class RankBarrier {
public:
    explicit RankBarrier(int n_ranks) : n_(n_ranks < 1 ? 1 : n_ranks), waiting_(0), gen_(0) {}

    RankBarrier(const RankBarrier&) = delete;
    RankBarrier& operator=(const RankBarrier&) = delete;

    void arrive_and_wait();

    /* Release everyone regardless of how many have arrived, and refuse every later arrival. The
     * engine calls this on shutdown and on a device fault: a rank that has already exited must not
     * leave its peers parked on a barrier that can no longer be completed (spec §17 -- a device
     * fault is not recoverable in-process, and the engine exits rather than serving from a wedged
     * queue, which it cannot do while a thread is blocked here). */
    void abort();


private:
    std::mutex              m_;
    std::condition_variable cv_;
    int                     n_;
    int                     waiting_;
    uint64_t                gen_;
    bool                    aborted_ = false;
};

}  /* namespace rad */
