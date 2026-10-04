/* stager.h -- the seam a step-time weight stager plugs into.
 *
 * The run phase resolves weights through the residency table (runtime/residency.h), and the mover
 * changes that table between steps. A stager changes it WITHIN a step: it copies a weight that
 * lives off the card into VRAM ahead of the op that reads it and publishes the copy for that op,
 * then publishes the host address again once the op is behind it. The run phase does not know what
 * is staged or why; it calls the stager around the ops of a pass it armed for and re-reads the
 * residency table when the stager says it changed it.
 *
 * A PASS THE STAGER ARMS FOR IS ISSUED LIVE, never recorded or played from a tape: a tape replays
 * the packets of the pass it recorded and not the host work between them, and the copies and
 * publishes here are exactly that host work.
 */
#pragma once
#include "../rad_core.h"
#include "rad_device.h"

namespace rad {

class OpStager {
public:
    virtual ~OpStager() = default;

    /* The pass is about to be issued: whether the stager takes part in it. */
    virtual bool begin_pass(const RadBatch* b) = 0;

    /* Before op `h` is issued on stream `s`. `*resync` is set when the residency table changed,
     * so the run phase brings the op's weight tables up to date before resolving them. */
    virtual int before_op(rad_op h, RadStream s, bool* resync) = 0;

    /* After op `h` was issued on stream `s`: the point a staged weight's last reader is behind. */
    virtual int after_op(rad_op h, RadStream s) = 0;

    /* The pass has been issued, on `s`: nothing the stager published may outlive it. */
    virtual int end_pass(RadStream s) = 0;
};

/* TWO STAGERS ON ONE PASS, each called only on the passes it armed for. A stager that declined a
 * pass is not handed its ops: one that arms only for large passes -- because its buffers are lent
 * out the rest of the time -- must not stage into memory it does not own because another stager
 * armed. The order is fixed, so the copies and stream waits each issues land in the same order on
 * every pass. */
class StagerChain final : public OpStager {
public:
    void set(OpStager* a, OpStager* b) { s_[0] = a; s_[1] = b; }

    bool begin_pass(const RadBatch* b) override {
        bool any = false;
        for (int i = 0; i < 2; ++i) {
            on_[i] = s_[i] != nullptr && s_[i]->begin_pass(b);
            any = any || on_[i];
        }
        return any;
    }
    int before_op(rad_op h, RadStream s, bool* resync) override {
        *resync = false;
        for (int i = 0; i < 2; ++i) {
            if (!on_[i]) continue;
            bool r = false;
            const int rc = s_[i]->before_op(h, s, &r);
            if (rc < 0) return rc;
            *resync = *resync || r;
        }
        return RAD_OK;
    }
    int after_op(rad_op h, RadStream s) override {
        for (int i = 0; i < 2; ++i)
            if (on_[i]) { const int rc = s_[i]->after_op(h, s); if (rc < 0) return rc; }
        return RAD_OK;
    }
    int end_pass(RadStream s) override {
        int rc = RAD_OK;
        for (int i = 0; i < 2; ++i) {
            if (!on_[i]) continue;
            const int r = s_[i]->end_pass(s);
            if (r < 0 && rc >= 0) rc = r;
            on_[i] = false;
        }
        return rc;
    }

private:
    OpStager* s_[2] = { nullptr, nullptr };
    bool      on_[2] = { false, false };
};

}  /* namespace rad */
