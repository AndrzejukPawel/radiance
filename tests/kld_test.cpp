/* kld_test.cpp -- the KL scorer (core/kld.h): its arithmetic and its reference files.
 *
 * No device and no model. A reference is recorded from synthetic logits split over two vocabulary
 * shards, the way two ranks hand them over; a candidate is scored against it; and every position's
 * KL, argmax agreement and NLL is held to a direct computation in double from the two sets of f32
 * logits. The f16 the reference keeps is the only approximation, so the tolerance is what that
 * costs -- and a candidate identical to the reference has to score zero.
 */
#include "rad_test.h"

#include "kld.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace rad;

namespace {

constexpr int64_t V = 1000;          /* two shards of 500 */
constexpr int64_t W = V / 2;

struct TmpDir {
    std::string path;
    TmpDir() {
        char t[] = "/tmp/kld_test.XXXXXX";
        path = mkdtemp(t) ? t : "";
    }
    ~TmpDir() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};

/* Logits with a confident head and a long tail: one token a few nats clear of a normal spread,
 * which is the shape where an f16 head would cost the most and the exact one has to cover it. */
std::vector<float> make_logits(int64_t rows, uint32_t seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> n(0.0f, 2.5f);
    std::uniform_int_distribution<int> pick(0, (int)V - 1);
    std::vector<float> l((size_t)(rows * V));
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t i = 0; i < V; ++i) l[(size_t)(r * V + i)] = n(g);
        l[(size_t)(r * V + pick(g))] += (r % 3 == 0) ? 2.0f : 9.0f;
    }
    return l;
}

/* The two shards a step hands over, from full rows. */
struct Shards {
    std::vector<float> a, b;
    std::vector<KldShard> s;
    Shards(const std::vector<float>& full, int64_t rows) : a((size_t)(rows * W)), b((size_t)(rows * W)) {
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t i = 0; i < W; ++i) {
                a[(size_t)(r * W + i)] = full[(size_t)(r * V + i)];
                b[(size_t)(r * W + i)] = full[(size_t)(r * V + W + i)];
            }
        s = { { a.data(), 0, W }, { b.data(), W, W } };
    }
};

double lse(const float* l) {
    double m = l[0];
    for (int64_t i = 1; i < V; ++i) m = std::max(m, (double)l[i]);
    double s = 0;
    for (int64_t i = 0; i < V; ++i) s += std::exp((double)l[i] - m);
    return m + std::log(s);
}

double kl_direct(const float* p, const float* q) {
    const double zp = lse(p), zq = lse(q);
    double kl = 0;
    for (int64_t i = 0; i < V; ++i) {
        const double lp = (double)p[i] - zp;
        kl += std::exp(lp) * (lp - ((double)q[i] - zq));
    }
    return kl;
}

int64_t argmax(const float* l) {
    int64_t a = 0;
    for (int64_t i = 1; i < V; ++i) if (l[i] > l[a]) a = i;
    return a;
}

/* Two documents: 40 tokens scored from 8, and 25 scored from 0 -- so 31 + 24 rows. */
const std::vector<int32_t> kDocA = [] { std::vector<int32_t> t(40); for (int i = 0; i < 40; ++i) t[i] = (i * 37) % V; return t; }();
const std::vector<int32_t> kDocB = [] { std::vector<int32_t> t(25); for (int i = 0; i < 25; ++i) t[i] = (i * 91 + 5) % V; return t; }();

/* Positions [0, n) of a document, one logits row each, handed over in two steps the way a
 * chunked prefill would: the first chunk ends mid-document. */
void feed(Kld& k, uint64_t id, int64_t n, const std::vector<float>& full) {
    const int64_t cut = n / 2;
    {
        std::vector<float> part(full.begin(), full.begin() + cut * V);
        Shards sh(part, cut);
        CHECK_OK(k.score({ { id, 0, cut, 0 } }, sh.s));
    }
    {
        std::vector<float> part(full.begin() + cut * V, full.begin() + n * V);
        Shards sh(part, n - cut);
        CHECK_OK(k.score({ { id, cut, n - cut, 0 } }, sh.s));
    }
}

std::vector<float> read_rows(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    in.seekg(0, std::ios::end);
    std::vector<float> v((size_t)in.tellg() / sizeof(float));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(v.size() * sizeof(float)));
    return v;
}

void record(const std::string& dir, const std::vector<float>& la, const std::vector<float>& lb) {
    Kld rec;
    CHECK_OK(rec.add_doc("a", kDocA, 8));
    CHECK_OK(rec.add_doc("b", kDocB, 0));
    CHECK_OK(rec.record_into(dir, V, "reference"));
    CHECK_EQ(rec.n_rows(), (int64_t)(31 + 24));
    feed(rec, 1, 40, la);
    feed(rec, 2, 25, lb);
    CHECK_OK(rec.finish());
}

}  /* namespace */

/* THE NUMBERS THEMSELVES. Every scored position of a perturbed candidate against a direct double
 * computation from both sets of logits, and the rows outside each document's range left out. */
TEST(a_candidate_scores_what_a_direct_computation_gives) {
    TmpDir t;
    REQUIRE(!t.path.empty());
    const std::vector<float> ra = make_logits(40, 1), rb = make_logits(25, 2);
    record(t.path + "/ref", ra, rb);

    /* The candidate: the reference plus noise, enough to flip some argmaxes. */
    std::vector<float> ca = ra, cb = rb;
    std::mt19937 g(7);
    std::normal_distribution<float> n(0.0f, 0.6f);
    for (float& v : ca) v += n(g);
    for (float& v : cb) v += n(g);

    Kld k;
    CHECK_OK(k.score_against(t.path + "/ref", t.path + "/rep.json", V, "candidate"));
    feed(k, 1, 40, ca);
    feed(k, 2, 25, cb);
    CHECK_OK(k.finish());

    const std::vector<float> rows = read_rows(t.path + "/rep.json.rows");
    REQUIRE_EQ(rows.size(), (size_t)(55 * 4));
    int64_t r = 0, flips = 0;
    auto check_doc = [&](const std::vector<float>& ref, const std::vector<float>& cand,
                         const std::vector<int32_t>& tok, int64_t from) {
        for (int64_t p = from; p + 1 < (int64_t)tok.size(); ++p, ++r) {
            const float* lp = ref.data() + p * V;
            const float* lq = cand.data() + p * V;
            const double want = kl_direct(lp, lq);
            CHECK_NEAR(rows[(size_t)(r * 4 + 0)], want, 2e-5 + 1e-3 * want);
            CHECK_NEAR(rows[(size_t)(r * 4 + 1)], lse(lq) - lq[tok[(size_t)p + 1]], 1e-4);
            CHECK_NEAR(rows[(size_t)(r * 4 + 2)], lse(lp) - lp[tok[(size_t)p + 1]], 1e-4);
            const bool same = argmax(lp) == argmax(lq);
            CHECK_EQ(rows[(size_t)(r * 4 + 3)], same ? 1.0f : 0.0f);
            flips += !same;
        }
    };
    check_doc(ra, ca, kDocA, 8);
    check_doc(rb, cb, kDocB, 0);
    CHECK_EQ(r, (int64_t)55);
    CHECK(flips > 0);         /* the noise has to have moved something for the check to mean it */
}

/* THE CONTROL. A candidate identical to the reference: every KL is the f16 storage's second-order
 * residue, and every argmax agrees. */
TEST(the_reference_against_itself_scores_zero) {
    TmpDir t;
    REQUIRE(!t.path.empty());
    const std::vector<float> ra = make_logits(40, 3), rb = make_logits(25, 4);
    record(t.path + "/ref", ra, rb);
    Kld k;
    CHECK_OK(k.score_against(t.path + "/ref", t.path + "/rep.json", V, "same"));
    feed(k, 1, 40, ra);
    feed(k, 2, 25, rb);
    CHECK_OK(k.finish());
    const std::vector<float> rows = read_rows(t.path + "/rep.json.rows");
    REQUIRE_EQ(rows.size(), (size_t)(55 * 4));
    for (size_t r = 0; r < 55; ++r) {
        CHECK(std::fabs(rows[r * 4 + 0]) < 1e-6f);
        CHECK_EQ(rows[r * 4 + 3], 1.0f);
        CHECK_NEAR(rows[r * 4 + 1], rows[r * 4 + 2], 1e-5);
    }
}

/* A REFERENCE IS NEVER WRITTEN OVER, and one whose recording did not finish is never read. */
TEST(a_reference_is_written_once_and_read_only_when_complete) {
    TmpDir t;
    REQUIRE(!t.path.empty());
    const std::vector<float> ra = make_logits(40, 5), rb = make_logits(25, 6);
    record(t.path + "/ref", ra, rb);
    {
        Kld again;
        CHECK_OK(again.add_doc("a", kDocA, 8));
        CHECK_EQ(again.record_into(t.path + "/ref", V, "x"), RAD_E_INVAL);
    }
    /* A recording that stops short: the manifest is never written, finish() says what is missing,
     * and the directory is not a reference. */
    {
        Kld half;
        CHECK_OK(half.add_doc("a", kDocA, 8));
        CHECK_OK(half.add_doc("b", kDocB, 0));
        CHECK_OK(half.record_into(t.path + "/half", V, "x"));
        feed(half, 1, 40, ra);
        CHECK_EQ(half.finish(), RAD_E_STATE);
    }
    Kld k;
    CHECK(k.score_against(t.path + "/half", "", V, "x") < 0);
    /* And a vocabulary that is not the reference's. */
    Kld v;
    CHECK_EQ(v.score_against(t.path + "/ref", "", V + 1, "x"), RAD_E_INVAL);
}

/* A DOCUMENT WITH NOTHING TO SCORE is refused when it is added, not discovered as a missing row. */
TEST(a_document_must_leave_a_position_to_score) {
    Kld k;
    CHECK_EQ(k.add_doc("x", { 1 }, 0), RAD_E_INVAL);
    CHECK_EQ(k.add_doc("x", { 1, 2, 3 }, 2), RAD_E_INVAL);
    CHECK_EQ(k.add_doc("x", { 1, 2, 3 }, -1), RAD_E_INVAL);
    CHECK_OK(k.add_doc("x", { 1, 2, 3 }, 1));
    CHECK_EQ(k.n_rows(), (int64_t)1);
}

RAD_TEST_MAIN()
