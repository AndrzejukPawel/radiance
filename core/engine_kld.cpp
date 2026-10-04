/* engine_kld.cpp -- the KL mode's two halves on the Engine: the driver that runs in place of the
 * HTTP server, and the copy of a step's scored rows off the cards. The arithmetic and the files
 * are core/kld.h's. */
#include "engine_priv.h"
#include "kld.h"

#include "device/device.h"
#include "server/sink.h"

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

namespace rad {

int Engine::run_kld() {
    /* THE LOGITS BUFFER AS EACH RANK DECLARED IT. It has to hold a step's tokens after the
     * sampler's rows (RadBuildCtx::max_out_rows); a plugin that sized it by its own rule alone
     * did not read the field, and is refused here rather than overrun at the first step. */
    const int64_t need = cfg_.max_tok + cfg_.max_seqs;
    const int64_t nv = meta_.n_vocab;
    kld_vocab0_.clear();
    kld_width_.clear();
    for (const auto& rk : ranks_) {
        const Program& p = rk->program;
        if (p.logits_buf == RAD_NULL_HANDLE || (size_t)p.logits_buf >= p.buffers.size()) {
            RAD_ERR("kld: rank %d declared no logits buffer", rk->index);
            return RAD_E_UNSUPPORTED;
        }
        const RadBufDecl& d = p.buffers[p.logits_buf].decl;
        if (d.dtype != RAD_F32 || d.rank != 2 || d.shape[0] < need) {
            RAD_ERR("kld: the '%s' plugin's logits buffer is %lld rows of %s; the KL mode needs "
                    "f32 rows for a whole step, %lld (it reads RadBuildCtx::max_out_rows)",
                    plugins_.arch_plugin.c_str(), (long long)d.shape[0],
                    d.dtype == RAD_F32 ? "f32" : "another dtype", (long long)need);
            return RAD_E_UNSUPPORTED;
        }
        /* The columns a rank owns, by the rule declare_sampler checked at startup: a contiguous
         * equal split, or the whole vocabulary on every rank -- read from rank 0 alone then. */
        const int64_t w = d.shape[1];
        if (w >= nv) {
            if (!kld_width_.empty()) continue;
            kld_vocab0_.push_back(0);
            kld_width_.push_back(nv);
            continue;
        }
        const int64_t per = (nv + cfg_.tp - 1) / cfg_.tp;
        kld_vocab0_.push_back(per * rk->index);
        kld_width_.push_back(w);
    }
    int64_t cover = 0;
    for (int64_t w : kld_width_) cover += w;
    if (cover != nv) {
        RAD_ERR("kld: the ranks' logits columns add up to %lld of a %lld-token vocabulary",
                (long long)cover, (long long)nv);
        return RAD_E_STATE;
    }
    kld_host_.assign(kld_width_.size(), nullptr);
    for (size_t r = 0; r < kld_width_.size(); ++r) {
        kld_host_[r] = static_cast<float*>(
            rad_dev_alloc(cfg_.max_tok * kld_width_[r] * (int64_t)sizeof(float), RAD_MEM_HOST_PINNED));
        if (!kld_host_[r]) {
            RAD_ERR("kld: no pinned host memory for a step's logits (%lld MiB a rank); lower "
                    "--max-num-batched-tokens", (long long)(cfg_.max_tok * kld_width_[r] * 4 >> 20));
            return RAD_E_NOMEM;
        }
    }

    auto kld = std::make_unique<Kld>();
    RAD_TRY(kld->open(cfg_, vocab_, nv, cfg_.model));
    /* Published before add_all takes the scheduler's lock, and read by the step loop only on a
     * step that carries a scored row -- one planned under that lock, after it. */
    kld_ = std::move(kld);

    /* EVERY DOCUMENT AT ONCE, under one hold of the queue: the steps they are packed into are
     * then a function of the corpus and the flags alone, so a rerun batches them identically. */
    const size_t n = kld_->n_docs();
    std::vector<std::unique_ptr<server::Sink>> sinks(n);
    std::vector<std::unique_ptr<Request>> reqs(n);
    for (size_t d = 0; d < n; ++d) {
        sinks[d] = std::make_unique<server::Sink>(64);
        auto r = std::make_unique<Request>();
        r->id         = d + 1;
        r->prompt     = kld_->tokens(d);
        r->max_tokens = 1;
        r->ignore_eos = true;
        r->sp.temp    = 0.0f;
        r->score      = true;
        r->sink       = sinks[d].get();
        sinks[d]->id  = r->id;
        reqs[d] = std::move(r);
    }
    std::vector<int> rc(n, RAD_OK);
    sched_.add_all(reqs.data(), n, rc.data());
    for (size_t d = 0; d < n; ++d) {
        if (rc[d] >= 0) continue;
        RAD_ERR("kld: the scheduler refused document %zu (%zu tokens): %s", d,
                kld_->tokens(d).size(), rad_strerror(rc[d]));
        return rc[d];
    }

    const auto t0 = std::chrono::steady_clock::now();
    auto last = t0;
    for (size_t first = 0;;) {
        while (first < n && sinks[first]->done()) ++first;
        if (first == n) break;
        const int ls = loop_status_.load(std::memory_order_acquire);
        if (ls < 0) return ls;
        sinks[first]->wait(200);
        const auto now = std::chrono::steady_clock::now();
        if (now - last >= std::chrono::seconds(15)) {
            last = now;
            const double s = std::chrono::duration<double>(now - t0).count();
            RAD_INFO("kld: %lld / %lld positions, %zu / %zu documents, %.0f s",
                     (long long)kld_->rows_done(), (long long)kld_->n_rows(), first, n, s);
        }
    }
    int bad = 0;
    for (size_t d = 0; d < n; ++d) {
        const server::Finish f = sinks[d]->finish_reason();
        if (f == server::Finish::Error || f == server::Finish::Cancelled) {
            RAD_ERR("kld: document %zu did not finish: %s", d, rad_strerror(sinks[d]->status()));
            ++bad;
        }
        sched_.reap(sinks[d]->id);
    }
    if (bad) return RAD_E_STATE;
    RAD_INFO("kld: %zu documents in %.0f s", n,
             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return kld_->finish();
}

/* A STEP'S SCORED ROWS, OFF THE CARDS. They follow the sampler's rows in the logits buffer, in
 * the entry order the batch builder wrote them (core/sched/batch.cpp, "THE SCORED ROWS"), so the
 * walk below is that loop again and the count is checked against what the batch named. Every
 * rank's piece on its own stream, behind the step; the next step is not issued until they are
 * read, because the KL mode does not pipeline. */
int Engine::score_rows(const RadBatch* b, const StepOut& so) {
    const int64_t n = b->n_out - so.n_rows;
    if (n <= 0) return RAD_OK;
    const StepPlan& plan = sched_.plan();
    std::vector<KldSpan> spans;
    int64_t row = 0;
    for (const StepEntry& e : plan.e) {
        if (!(e.req && e.req->score && e.is_prefill && e.hidden_row < 0)) continue;
        spans.push_back({ e.req->id, (int64_t)e.ctx_len, (int64_t)e.n_tokens, row });
        row += e.n_tokens;
    }
    if (row != n || n > cfg_.max_tok) {
        RAD_ERR("kld: step %d names %lld scored rows and its plan %lld", b->step, (long long)n,
                (long long)row);
        return RAD_E_STATE;
    }
    for (size_t r = 0; r < kld_width_.size(); ++r) {
        Rank& rk = *ranks_[r];
        RAD_TRY(rad_dev_set(rk.device));
        const auto* src = static_cast<const float*>(rk.ctx.buf_ptr(rk.program.logits_buf));
        if (!src) return RAD_E_STATE;
        RAD_TRY(rad_memcpy_async(kld_host_[r], src + (int64_t)so.n_rows * kld_width_[r],
                                 n * kld_width_[r] * (int64_t)sizeof(float), rk.ctx.stream()));
    }
    for (size_t r = 0; r < kld_width_.size(); ++r) {
        RAD_TRY(rad_dev_set(ranks_[r]->device));
        RAD_TRY(rad_stream_sync(ranks_[r]->ctx.stream()));
    }
    RAD_TRY(rad_dev_set(ranks_[0]->device));
    std::vector<KldShard> shards(kld_width_.size());
    for (size_t r = 0; r < shards.size(); ++r)
        shards[r] = { kld_host_[r], kld_vocab0_[r], kld_width_[r] };
    return kld_->score(spans, shards);
}

}  /* namespace rad */
