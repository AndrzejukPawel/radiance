/* rank_barrier.cpp -- see rank_barrier.h for why this is a condvar and not an IPC handshake. */
#include "rank_barrier.h"

namespace rad {

void RankBarrier::arrive_and_wait() {
    std::unique_lock<std::mutex> lk(m_);
    if (aborted_) return;

    const uint64_t my_gen = gen_;
    if (++waiting_ == n_) {
        /* Last one in flips the generation and releases the rest. The counter is reset here rather
         * than by the wakers, so a rank that re-enters immediately observes a generation it has
         * not seen and blocks instead of racing through. */
        waiting_ = 0;
        ++gen_;
        lk.unlock();
        cv_.notify_all();
        return;
    }
    cv_.wait(lk, [&] { return aborted_ || gen_ != my_gen; });
}

void RankBarrier::abort() {
    {
        std::lock_guard<std::mutex> lk(m_);
        aborted_ = true;
        waiting_ = 0;
        ++gen_;
    }
    cv_.notify_all();
}

}  /* namespace rad */
