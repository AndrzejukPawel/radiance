/* stager.cpp -- see stager.h. */
#include "place/stager.h"

#include <algorithm>

namespace rad {

PrefillStager::~PrefillStager() {
    for (int b = 0; b < 2; ++b) {
        if (done_[b]) rad_event_destroy(done_[b]);
        if (free_[b]) rad_event_destroy(free_[b]);
    }
}

int PrefillStager::init(const Program& prog, const Plan& plan, Mover* mover, char* region,
                        int64_t bytes, int64_t min_rows) {
    layers_.clear();
    if (!mover || !region || bytes <= 0) return RAD_OK;

    /* A ROUTED LAYER IS ITS EXPERT UNITS, and the ops that read it are the span their declared
     * uses cover: the first is the one that must find the copies, the last the one after which
     * nothing reads them again -- once the block's last pass of rows has issued it (after_op). */
    std::map<int32_t, Layer> by_layer;
    for (size_t u = 0; u < plan.units.size(); ++u) {
        const MoveUnit& mu = plan.units[u];
        if (mu.expert < 0 || mu.layer < 0 || mu.first_use_op < 0 || mu.last_use_op < 0) continue;
        Layer& L = by_layer[mu.layer];
        L.layer = mu.layer;
        L.units.push_back((int32_t)u);
        L.first_op = L.first_op < 0 ? mu.first_use_op : std::min(L.first_op, mu.first_use_op);
        L.last_op = std::max(L.last_op, mu.last_use_op);
    }
    for (auto& kv : by_layer) layers_.push_back(std::move(kv.second));
    std::sort(layers_.begin(), layers_.end(),
              [](const Layer& a, const Layer& b) { return a.first_op < b.first_op; });
    /* Two layers' spans must not overlap: a layer's copies are released at its last op, and the
     * next layer's buffer is the one the layer before it released. */
    for (size_t i = 1; i < layers_.size(); ++i)
        if (layers_[i].first_op <= layers_[i - 1].last_op) {
            RAD_WARN("prefill staging: routed layers %d and %d are read by interleaved ops (%d..%d "
                     "and %d..%d); staging is off", layers_[i - 1].layer, layers_[i].layer,
                     layers_[i - 1].first_op, layers_[i - 1].last_op, layers_[i].first_op,
                     layers_[i].last_op);
            layers_.clear();
            return RAD_OK;
        }
    if (layers_.empty()) return RAD_OK;

    first_of_.assign(prog.ops.size(), -1);
    last_of_.assign(prog.ops.size(), -1);
    for (size_t i = 0; i < layers_.size(); ++i) {
        if ((size_t)layers_[i].first_op < first_of_.size()) first_of_[(size_t)layers_[i].first_op] = (int32_t)i;
        if ((size_t)layers_[i].last_op < last_of_.size()) last_of_[(size_t)layers_[i].last_op] = (int32_t)i;
    }

    mover_ = mover;
    cap_ = bytes / 2 / RAD_ALIGN_UNIT * RAD_ALIGN_UNIT;
    buf_[0] = region;
    buf_[1] = region + cap_;
    min_rows_ = min_rows;
    for (int b = 0; b < 2; ++b) {
        if (rad_event_create_local(&done_[b]) < 0 || rad_event_create_local(&free_[b]) < 0) {
            layers_.clear();
            return RAD_E_DEVICE;
        }
    }
    started_.assign(layers_.size(), 0);
    return RAD_OK;
}

bool PrefillStager::begin_pass(const RadBatch* b) {
    armed_ = full_ && enabled() && b && b->n_tok >= min_rows_;
    if (!armed_) return false;
    std::fill(started_.begin(), started_.end(), (uint8_t)0);
    holds_[0] = holds_[1] = -1;
    first_op_ = true;
    pass_kind_ = b->draft_pass;
    last_issued_ = -1;
    return true;
}

bool PrefillStager::will_issue(int32_t li) const {
    const auto it = pass_last_.find(pass_kind_);
    return it == pass_last_.end() || layers_[(size_t)li].first_op <= it->second;
}

int PrefillStager::start(int32_t li) {
    started_[(size_t)li] = 1;
    const int b = li & 1;
    if (holds_[b] >= 0) return RAD_OK;       /* the layer before still holds it: stream this one */
    const int rc = mover_->stage_units(layers_[(size_t)li].units, buf_[b], cap_,
                                       free_rec_[b] ? free_[b] : nullptr, done_[b], &staged_[b]);
    if (rc < 0) return rc;
    if (!staged_[b].empty()) { holds_[b] = li; read_[b] = false; }
    return RAD_OK;
}

/* The host half of letting a buffer go: its units are published at their pool slots again, which
 * the next residency sync hands to the weight tables. The device half is `free_`. */
void PrefillStager::release(int32_t b) {
    mover_->unstage_units(staged_[b]);
    staged_[b].clear();
    holds_[b] = -1;
    read_[b] = false;
}

int PrefillStager::finish(int32_t b, RadStream s) {
    RAD_TRY(rad_event_record(free_[b], s));
    free_rec_[b] = true;
    release(b);
    return RAD_OK;
}

int PrefillStager::before_op(rad_op h, RadStream s, bool* resync) {
    *resync = false;
    if ((int32_t)h > last_issued_) last_issued_ = (int32_t)h;

    /* THE PASS'S FIRST ROUTED LAYER, staged at its first op of any kind, so the copy runs behind
     * whatever the pass issues before the layer. */
    if (first_op_) {
        first_op_ = false;
        const auto it = std::lower_bound(layers_.begin(), layers_.end(), (int32_t)h,
                                         [](const Layer& L, int32_t op) { return L.first_op < op; });
        if (it != layers_.end()) {
            const int32_t li = (int32_t)(it - layers_.begin());
            if (!started_[(size_t)li] && will_issue(li)) {
                RAD_TRY(start(li));
                *resync = true;
            }
        }
    }

    if ((size_t)h >= first_of_.size()) return RAD_OK;
    const int32_t li = first_of_[(size_t)h];
    if (li < 0) return RAD_OK;
    /* This layer's copies were published a layer ago: the one wait that orders its ops behind
     * them. */
    for (int b = 0; b < 2; ++b)
        if (holds_[b] == li) RAD_TRY(rad_event_wait(s, done_[b]));
    /* The layer before is done with its buffer: every pass of rows issued its last op, the last
     * of them recorded `free_`, and a layer's ops do not interleave with the next one's. Released
     * here and not at that op, whose next run would read copies already overwritten. */
    for (int b = 0; b < 2; ++b)
        if (holds_[b] >= 0 && holds_[b] != li && read_[b]) release(b);
    /* And the next layer's copies go out now, behind it. */
    const int32_t next = li + 1;
    if ((size_t)next < layers_.size() && !started_[(size_t)next] && will_issue(next)) {
        RAD_TRY(start(next));
        *resync = true;
    }
    return RAD_OK;
}

/* THE LAYER'S LAST OP: `free_` goes behind it, so the copies two layers on can start as early as
 * they could before -- and goes behind it again if the block issues the op for another pass of
 * rows, which is why the copies themselves are not released until the next layer starts. */
int PrefillStager::after_op(rad_op h, RadStream s) {
    if ((size_t)h >= last_of_.size()) return RAD_OK;
    const int32_t li = last_of_[(size_t)h];
    if (li < 0) return RAD_OK;
    for (int b = 0; b < 2; ++b)
        if (holds_[b] == li) {
            RAD_TRY(rad_event_record(free_[b], s));
            free_rec_[b] = true;
            read_[b] = true;
        }
    return RAD_OK;
}

int PrefillStager::end_pass(RadStream s) {
    if (!armed_) return RAD_OK;
    armed_ = false;
    pass_last_[pass_kind_] = last_issued_;
    int rc = RAD_OK;
    for (int b = 0; b < 2; ++b)
        if (holds_[b] >= 0) {
            const int r = finish(b, s);
            if (r < 0 && rc >= 0) rc = r;
        }
    return rc;
}

/* ================================================================== the file stager */
FileStager::~FileStager() {
    for (int b = 0; b < 2; ++b) {
        if (done_[b]) rad_event_destroy(done_[b]);
        if (free_[b]) rad_event_destroy(free_[b]);
    }
}

int FileStager::init(const Program& prog, const Plan& plan, Mover* mover) {
    layers_.clear();
    if (!mover) return RAD_OK;

    std::map<int32_t, Layer> by_layer;
    for (size_t u = 0; u < plan.units.size(); ++u) {
        const MoveUnit& mu = plan.units[u];
        if (mu.tier != Tier::SSD || mu.access != RAD_ACCESS_CONDITIONAL || mu.layer < 0) continue;
        if (mu.first_use_op < 0 || mu.last_use_op < 0) continue;   /* declared and never read */
        Layer& L = by_layer[mu.layer];
        L.layer = mu.layer;
        L.units.push_back((int32_t)u);
        L.first_op = L.first_op < 0 ? mu.first_use_op : std::min(L.first_op, mu.first_use_op);
        L.last_op = std::max(L.last_op, mu.last_use_op);
    }
    for (auto& kv : by_layer) layers_.push_back(std::move(kv.second));
    if (layers_.empty()) return RAD_OK;
    std::sort(layers_.begin(), layers_.end(),
              [](const Layer& a, const Layer& b) { return a.first_op < b.first_op; });
    /* A layer's copies stay published until the next one starts, so two layers whose ops
     * interleave would need both buffers at once and a third for the read ahead. That is a plugin
     * shape this tier cannot serve, and a file-tier unit has no address to fall back on. */
    for (size_t i = 1; i < layers_.size(); ++i)
        if (layers_[i].first_op <= layers_[i - 1].last_op) {
            RAD_ERR("file stager: routed layers %d and %d are read by interleaved ops (%d..%d and "
                    "%d..%d), and a routed layer read from the container is published one layer "
                    "at a time", layers_[i - 1].layer, layers_[i].layer, layers_[i - 1].first_op,
                    layers_[i - 1].last_op, layers_[i].first_op, layers_[i].last_op);
            layers_.clear();
            return RAD_E_UNSUPPORTED;
        }

    buf_[0] = mover->file_buffer(0);
    buf_[1] = mover->file_buffer(1);
    cap_ = mover->file_buffer_bytes();
    if (!buf_[0] || !buf_[1] || cap_ < plan.file_layer_max) {
        RAD_ERR("file stager: the plan reads %zu routed layers from the container and the mover "
                "holds no buffers for them", layers_.size());
        layers_.clear();
        return RAD_E_STATE;
    }
    first_of_.assign(prog.ops.size(), -1);
    for (size_t i = 0; i < layers_.size(); ++i)
        if ((size_t)layers_[i].first_op < first_of_.size())
            first_of_[(size_t)layers_[i].first_op] = (int32_t)i;
    mover_ = mover;
    for (int b = 0; b < 2; ++b)
        if (rad_event_create_local(&done_[b]) < 0 || rad_event_create_local(&free_[b]) < 0) {
            layers_.clear();
            return RAD_E_DEVICE;
        }
    return RAD_OK;
}

/* EVERY PASS: there is no other address for a file-tier unit, so a pass the stager sat out would
 * reach its first routed layer with null expert pointers. */
bool FileStager::begin_pass(const RadBatch* b) {
    armed_ = enabled() && b;
    if (!armed_) return false;
    holds_[0] = holds_[1] = -1;
    cur_ = -1;
    first_op_ = true;
    pass_kind_ = b->draft_pass;
    last_issued_ = -1;
    return true;
}

bool FileStager::will_issue(int32_t li) const {
    const auto it = pass_last_.find(pass_kind_);
    return it == pass_last_.end() || layers_[(size_t)li].first_op <= it->second;
}

int FileStager::start(int32_t li) {
    const int b = li & 1;
    if (holds_[b] >= 0) return RAD_E_STATE;     /* released before the next read, by construction */
    const int rc = mover_->stage_file_units(layers_[(size_t)li].units, buf_[b], cap_,
                                            free_rec_[b] ? free_[b] : nullptr, done_[b],
                                            &staged_[b]);
    if (rc < 0) return rc;
    if (!staged_[b].empty()) holds_[b] = li;
    return RAD_OK;
}

int FileStager::finish(int32_t b, RadStream s) {
    RAD_TRY(rad_event_record(free_[b], s));
    free_rec_[b] = true;
    mover_->unstage_units(staged_[b]);
    staged_[b].clear();
    holds_[b] = -1;
    return RAD_OK;
}

int FileStager::before_op(rad_op h, RadStream s, bool* resync) {
    *resync = false;
    if ((int32_t)h > last_issued_) last_issued_ = (int32_t)h;

    /* THE PASS'S FIRST ROUTED LAYER, read at its first op of any kind. */
    if (first_op_) {
        first_op_ = false;
        const auto it = std::lower_bound(layers_.begin(), layers_.end(), (int32_t)h,
                                         [](const Layer& L, int32_t op) { return L.first_op < op; });
        if (it != layers_.end()) {
            const int32_t li = (int32_t)(it - layers_.begin());
            if (will_issue(li)) {
                RAD_TRY(start(li));
                *resync = true;
            }
        }
    }

    if ((size_t)h >= first_of_.size()) return RAD_OK;
    const int32_t li = first_of_[(size_t)h];
    if (li < 0 || li == cur_) return RAD_OK;

    /* Entering layer `li`: the layer before is behind every op issued so far on this stream. */
    for (int b = 0; b < 2; ++b)
        if (holds_[b] >= 0 && holds_[b] != li) { RAD_TRY(finish(b, s)); *resync = true; }
    /* A layer the read-ahead skipped -- the learned bound said this pass ends before it -- is read
     * here rather than issued without addresses. */
    if (holds_[li & 1] != li) {
        RAD_TRY(start(li));
        *resync = true;
    }
    if (holds_[li & 1] == li) RAD_TRY(rad_event_wait(s, done_[li & 1]));
    cur_ = li;

    const int32_t next = li + 1;
    if ((size_t)next < layers_.size() && will_issue(next)) {
        RAD_TRY(start(next));
        *resync = true;
    }
    return RAD_OK;
}

int FileStager::after_op(rad_op h, RadStream s) {
    (void)h;
    (void)s;
    return RAD_OK;
}

int FileStager::end_pass(RadStream s) {
    if (!armed_) return RAD_OK;
    armed_ = false;
    pass_last_[pass_kind_] = last_issued_;
    int rc = RAD_OK;
    for (int b = 0; b < 2; ++b)
        if (holds_[b] >= 0) {
            const int r = finish(b, s);
            if (r < 0 && rc >= 0) rc = r;
        }
    cur_ = -1;
    return rc;
}

}  /* namespace rad */
