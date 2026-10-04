/* util_test.cpp -- core/util/rad_util.cpp: the bedrock every other translation unit links against.
 *
 * WHY THIS FILE EXISTS. Nothing here is interesting on its own, and every one of these is small
 * enough to read and believe -- which is what leaves such code untested. The two that are not
 * merely cosmetic are `Geometry` and `constraint_holds`, and they decide WHICH KERNEL RUNS. A
 * geometry that hands a kernel dangling parameter pointers, or a constraint that holds when it
 * should not, does not fail -- it selects a different implementation, which produces plausible
 * output on a path nobody is looking at. The rest are here because a formatter that rounds the
 * wrong way makes every report that prints a size disagree with every other one.
 *
 * No device, no model, no core: one translation unit and its header.
 */
#include "rad_test.h"

#include "rad_internal.h"

#include <string>

using namespace rad;

/* ================================================================== small helpers */

TEST(align_up_rounds_up_and_leaves_an_exact_multiple_alone) {
    CHECK_EQ(align_up(0, 256), 0ll);
    CHECK_EQ(align_up(1, 256), 256ll);
    CHECK_EQ(align_up(255, 256), 256ll);
    CHECK_EQ(align_up(256, 256), 256ll);       /* already aligned: not bumped to the next unit */
    CHECK_EQ(align_up(257, 256), 512ll);
    CHECK_EQ(align_up(12345, 1), 12345ll);     /* an alignment of one is the identity */
    /* The sizes this is actually used on are pool extents, which are large. */
    CHECK_EQ(align_up(1073741825ll, 4096), 1073745920ll);
}

TEST(humanb_changes_unit_at_the_boundary_and_not_before) {
    CHECK_EQ(humanb(0), std::string("0 B"));
    CHECK_EQ(humanb(1), std::string("1 B"));
    /* 1023 IS STILL BYTES. A ladder that switched a unit early would print "1.00 KiB" for a
     * number below a kibibyte, and every size in every report is read against the others. */
    CHECK_EQ(humanb(1023), std::string("1023 B"));
    CHECK_EQ(humanb(1024), std::string("1.00 KiB"));
    CHECK_EQ(humanb(1536), std::string("1.50 KiB"));
    CHECK_EQ(humanb(1048576), std::string("1.00 MiB"));
    CHECK_EQ(humanb(1073741824ll), std::string("1.00 GiB"));
    CHECK_EQ(humanb(1099511627776ll), std::string("1.00 TiB"));
    /* THE LADDER STOPS AT TiB and says so in TiB rather than inventing a unit. */
    CHECK_EQ(humanb(1125899906842624ll), std::string("1024.00 TiB"));
}

TEST(fmt_sizes_its_own_buffer) {
    CHECK_EQ(fmt("%d", 7), std::string("7"));
    CHECK_EQ(fmt("%s=%lld", "M", 64ll), std::string("M=64"));
    CHECK_EQ(fmt("%s", ""), std::string(""));
    /* LONGER THAN ANY FIXED BUFFER SOMEBODY MIGHT HAVE REACHED FOR. It measures with vsnprintf
     * first, so there is no cap to find -- but a regression to a stack buffer would truncate
     * here and nowhere else, and the strings this formats include whole selection tables. */
    const std::string big = fmt("%s", std::string(10000, 'x').c_str());
    CHECK_EQ(big.size(), (size_t)10000);
    CHECK_EQ(big.front(), 'x');
    CHECK_EQ(big.back(), 'x');
}

TEST(rad_mono_ns_moves_forward_and_is_not_the_wall_clock) {
    const uint64_t a = rad_mono_ns();
    CHECK(a > 0);
    uint64_t b = a;
    /* Spin rather than sleep: the property is that it advances at all, and a test that sleeps
     * for it is a test that takes a millisecond to assert one comparison. */
    for (int i = 0; i < 1000000 && b == a; ++i) b = rad_mono_ns();
    CHECK(b >= a);
    /* Not a Unix timestamp. A monotonic clock counts from an arbitrary epoch -- usually boot --
     * and anything comparing it against time() would be comparing two different origins. */
    CHECK(a < 1000000000000000000ull);
}

/* SPLIT_WS SPLITS ON COMMAS TOO, which its name does not say and RAD_C_IN depends on: a kernel
 * writing RAD_CIN("dtype", "bf16,fp8") and one writing "bf16 fp8" mean the same set. */
TEST(split_ws_takes_spaces_tabs_and_commas) {
    auto v = split_ws("a b c");
    CHECK_EQ(v.size(), 3u);
    CHECK_EQ(std::string(v[0]), std::string("a"));
    CHECK_EQ(std::string(v[2]), std::string("c"));

    v = split_ws("bf16,fp8");
    CHECK_EQ(v.size(), 2u);
    CHECK_EQ(std::string(v[1]), std::string("fp8"));

    v = split_ws("\ta ,, b\t,c  ");
    CHECK_EQ(v.size(), 3u);
    CHECK_EQ(std::string(v[0]), std::string("a"));
    CHECK_EQ(std::string(v[1]), std::string("b"));
    CHECK_EQ(std::string(v[2]), std::string("c"));

    /* An empty set is empty rather than one empty token, or a constraint naming no values would
     * hold for a geometry whose value is the empty string. */
    CHECK_EQ(split_ws("").size(), 0u);
    CHECK_EQ(split_ws("   ,, \t ").size(), 0u);
    CHECK_EQ(split_ws("one").size(), 1u);
}

TEST(the_log_level_is_readable_and_settable) {
    const Log saved = log_level();
    log_set_level(Log::Trace);
    CHECK((int)log_level() == (int)Log::Trace);
    log_set_level(Log::Error);
    CHECK((int)log_level() == (int)Log::Error);
    log_set_level(saved);
    CHECK((int)log_level() == (int)saved);
}

/* ================================================================== Geometry */

TEST(geometry_stores_by_kind_and_a_reader_of_the_wrong_kind_misses) {
    Geometry g;
    g.set_i("M", 64);
    g.set_s("dtype", "w4a8");
    g.set_f("eps", 1e-6);

    long long i = 0;
    CHECK(g.get_i("M", &i));
    CHECK_EQ(i, 64ll);
    CHECK_EQ(std::string(g.get_s("dtype")), std::string("w4a8"));
    double d = 0;
    CHECK(g.get_f("eps", &d));
    CHECK_NEAR(d, 1e-6, 1e-12);

    /* THE KIND IS PART OF THE MATCH. get_i on a string slot missing is what makes a constraint
     * naming a float key fail rather than comparing against a truncation of it. */
    CHECK(!g.get_i("dtype", &i));
    CHECK(!g.get_i("eps", &i));
    CHECK(!g.get_f("M", &d));
    CHECK(g.get_s("M") == nullptr);
    CHECK(g.get_s("missing") == nullptr);
    CHECK(!g.get_i("missing", &i));

    /* has() is kind-agnostic: it answers whether the key is carried at all. */
    CHECK(g.has("M")); CHECK(g.has("dtype")); CHECK(g.has("eps"));
    CHECK(!g.has("N"));

    /* A null out pointer is a membership test. */
    CHECK(g.get_i("M", nullptr));
    CHECK(!g.get_i("dtype", nullptr));
}

TEST(setting_a_key_again_replaces_its_value_and_its_kind) {
    Geometry g;
    g.set_i("x", 1);
    g.set_i("x", 2);
    long long i = 0;
    CHECK(g.get_i("x", &i));
    CHECK_EQ(i, 2ll);
    CHECK_EQ(g.n_params(), 1);            /* replaced, not appended */

    g.set_s("x", "str");
    CHECK(!g.get_i("x", &i));             /* it is a string now and nothing else */
    CHECK_EQ(std::string(g.get_s("x")), std::string("str"));
    CHECK_EQ(g.n_params(), 1);
}

TEST(the_param_view_is_rebuilt_after_a_mutation) {
    Geometry g;
    g.set_i("M", 8);
    CHECK_EQ(g.n_params(), 1);
    CHECK_EQ(g.params()[0].ival, 8ll);

    /* The view is cached, so the question is whether the cache is invalidated. A stale view hands
     * a kernel the geometry it was resolved against rather than the one it is being run on. */
    g.set_i("M", 9);
    CHECK_EQ(g.params()[0].ival, 9ll);
    g.set_i("N", 4096);
    CHECK_EQ(g.n_params(), 2);

    /* A string param carries sval and an integer one carries a null sval, which is what lets a
     * kernel tell "not set" from "the empty string". */
    g.set_s("dtype", "bf16");
    const RadParam* p = g.params();
    int seen = 0;
    for (int k = 0; k < g.n_params(); ++k) {
        if (std::string(p[k].key) == "dtype") {
            CHECK(p[k].sval != nullptr);
            CHECK_EQ(std::string(p[k].sval), std::string("bf16"));
            ++seen;
        } else {
            CHECK(p[k].sval == nullptr);
        }
    }
    CHECK_EQ(seen, 1);
}

/* THE COPY MUST OWN ITS OWN STRINGS, and the header says why at length: the cached view holds
 * char* into the object's own storage, so a copy that took the view along with the storage would
 * hand a kernel pointers into the source -- which dies first about as often as not, since every
 * Resolved holds a Geometry and every Band holds two. */
TEST(a_copied_geometry_points_into_itself_and_not_at_the_source) {
    Geometry a;
    a.set_s("dtype", "w4a8");
    a.set_i("M", 64);
    const char* a_key = a.params()[0].key;         /* force the source to build its view */

    Geometry b = a;
    CHECK_EQ(b.n_params(), 2);
    CHECK_EQ(std::string(b.params()[0].key), std::string("dtype"));
    CHECK(b.params()[0].key != a_key);             /* a different object's storage */
    CHECK(b.params()[0].sval != a.params()[0].sval);
    CHECK_EQ(std::string(b.params()[0].sval), std::string("w4a8"));

    /* And the two are independent afterwards, in both directions. */
    a.set_i("M", 1);
    long long i = 0;
    CHECK(b.get_i("M", &i));
    CHECK_EQ(i, 64ll);
    b.set_i("M", 2);
    CHECK(a.get_i("M", &i));
    CHECK_EQ(i, 1ll);

    /* Assignment is the same rule. */
    Geometry c;
    c.set_i("junk", 7);
    c = a;
    CHECK(!c.has("junk"));
    CHECK_EQ(c.n_params(), 2);
    CHECK(c.params()[0].key != a.params()[0].key);
}

TEST(a_moved_geometry_does_not_leave_the_view_pointing_at_a_stolen_buffer) {
    Geometry a;
    a.set_s("dtype", "fp8");
    a.set_i("K", 5120);
    (void)a.params();                              /* build the view, then move out from under it */

    Geometry b = std::move(a);
    CHECK_EQ(b.n_params(), 2);
    CHECK_EQ(std::string(b.params()[0].sval), std::string("fp8"));
    /* The moved-from object is empty rather than holding a view of a buffer it no longer owns. */
    CHECK_EQ(a.n_params(), 0);

    Geometry c;
    c.set_i("x", 1);
    c = std::move(b);
    CHECK_EQ(c.n_params(), 2);
    CHECK(!c.has("x"));
    CHECK_EQ(b.n_params(), 0);
}

TEST(geometry_str_prints_every_kind_in_declaration_order) {
    Geometry g;
    g.set_i("M", 64);
    g.set_i("N", 8192);
    g.set_s("dtype", "w4a8");
    CHECK_EQ(g.str(), std::string("M=64 N=8192 dtype=w4a8"));

    Geometry f;
    f.set_f("eps", 1e-6);
    CHECK_EQ(f.str(), std::string("eps=1e-06"));

    CHECK_EQ(Geometry{}.str(), std::string(""));
}

/* ================================================================== constraint matching */

static RadConstraint c_eq(const char* k, long long v)  { return RadConstraint RAD_CEQ(k, v); }
static RadConstraint c_le(const char* k, long long v)  { return RadConstraint RAD_CLE(k, v); }
static RadConstraint c_ge(const char* k, long long v)  { return RadConstraint RAD_CGE(k, v); }
static RadConstraint c_div(const char* k, long long v) { return RadConstraint RAD_CDIV(k, v); }
static RadConstraint c_in(const char* k, const char* s) { return RadConstraint RAD_CIN(k, s); }

TEST(the_four_integer_constraints_compare_the_way_they_read) {
    Geometry g;
    g.set_i("M", 64);

    CHECK(constraint_holds(c_eq("M", 64), g));
    CHECK(!constraint_holds(c_eq("M", 63), g));

    CHECK(constraint_holds(c_le("M", 64), g));      /* inclusive at the bound */
    CHECK(constraint_holds(c_le("M", 65), g));
    CHECK(!constraint_holds(c_le("M", 63), g));

    CHECK(constraint_holds(c_ge("M", 64), g));      /* inclusive at the bound */
    CHECK(constraint_holds(c_ge("M", 1), g));
    CHECK(!constraint_holds(c_ge("M", 65), g));

    CHECK(constraint_holds(c_div("M", 16), g));
    CHECK(constraint_holds(c_div("M", 1), g));
    CHECK(!constraint_holds(c_div("M", 5), g));
}

/* A DIVISOR OF ZERO IS NOT "DIVIDES EVERYTHING", it is a kernel-set typo. Answering true would
 * route every geometry to that row, which is the widest possible band by accident. */
TEST(a_zero_divisor_holds_for_nothing_rather_than_faulting) {
    Geometry g;
    g.set_i("M", 64);
    CHECK(!constraint_holds(c_div("M", 0), g));
    g.set_i("Z", 0);
    CHECK(!constraint_holds(c_div("Z", 0), g));
    CHECK(constraint_holds(c_div("Z", 16), g));     /* zero is divisible by anything non-zero */
}

/* A CONSTRAINT WHOSE KEY THE GEOMETRY DOES NOT CARRY DOES NOT HOLD. The header names what this
 * buys: libr4d's two-shot all-reduce rows are reachable only by a query that asks for `hops`,
 * because a query that omits it fails the more constrained row. */
TEST(an_absent_key_fails_every_constraint) {
    Geometry g;
    g.set_i("M", 64);
    CHECK(!constraint_holds(c_eq("hops", 2), g));
    CHECK(!constraint_holds(c_le("hops", 99), g));
    CHECK(!constraint_holds(c_ge("hops", 0), g));
    CHECK(!constraint_holds(c_div("hops", 1), g));
    CHECK(!constraint_holds(c_in("hops", "1 2"), g));
}

/* A FLOAT PARAMETER CAN NEVER SATISFY ONE, and that is the intended answer rather than an
 * oversight: a tolerance is not a predicate, and a kernel discriminating on eps to that precision
 * has a bug. */
TEST(a_float_parameter_satisfies_no_constraint) {
    Geometry g;
    g.set_f("eps", 1e-6);
    g.set_f("scale", 1.0);
    CHECK(!constraint_holds(c_eq("scale", 1), g));
    CHECK(!constraint_holds(c_le("scale", 2), g));
    CHECK(!constraint_holds(c_ge("eps", 0), g));
    CHECK(!constraint_holds(c_div("scale", 1), g));
    CHECK(!constraint_holds(c_in("scale", "1"), g));
}

TEST(a_set_constraint_matches_a_string_and_an_integers_decimal_form) {
    Geometry g;
    g.set_s("dtype", "fp8");
    CHECK(constraint_holds(c_in("dtype", "bf16 fp8"), g));
    CHECK(constraint_holds(c_in("dtype", "bf16,fp8"), g));
    CHECK(!constraint_holds(c_in("dtype", "bf16 w4a8"), g));
    /* A PREFIX IS NOT A MEMBER. Matching "fp8" against a set containing "fp8e5" would route a
     * kernel compiled for one encoding at another. */
    CHECK(!constraint_holds(c_in("dtype", "fp8e5m2"), g));

    /* The integer form, which is what makes RAD_CIN("causal", "0 1") work. */
    Geometry i;
    i.set_i("causal", 1);
    CHECK(constraint_holds(c_in("causal", "0 1"), i));
    i.set_i("causal", 2);
    CHECK(!constraint_holds(c_in("causal", "0 1"), i));
    i.set_i("neg", -1);
    CHECK(constraint_holds(c_in("neg", "-1 0"), i));

    /* An empty or absent set matches nothing, including the empty string. */
    Geometry e;
    e.set_s("k", "");
    CHECK(!constraint_holds(c_in("k", ""), e));
    RadConstraint nullset = c_in("k", nullptr);
    CHECK(!constraint_holds(nullset, e));
}

TEST(an_unknown_constraint_operator_holds_for_nothing) {
    Geometry g;
    g.set_i("M", 64);
    RadConstraint c{};
    c.key = "M";
    c.op = 99;
    c.ival = 64;
    CHECK(!constraint_holds(c, g));
}

/* THE MISS COLUMN OF THE SELECTION TABLE (spec §16). Its job is to say which kernel was rejected
 * and why in terms an operator can act on, so it has to name the key, the bound and the value --
 * and it has to distinguish "your number is wrong" from "you did not supply that number at all". */
TEST(constraint_why_names_the_key_the_bound_and_what_was_there) {
    Geometry g;
    g.set_i("M", 64);

    const std::string le = constraint_why(c_le("M", 32), g);
    CHECK(le.find("M") != std::string::npos);
    CHECK(le.find("<=") != std::string::npos);
    CHECK(le.find("32") != std::string::npos);
    CHECK(le.find("64") != std::string::npos);

    const std::string dv = constraint_why(c_div("M", 5), g);
    CHECK(dv.find("%") != std::string::npos);
    CHECK(dv.find("5") != std::string::npos);

    /* Absent reads differently from wrong, because the repair is different. */
    const std::string ab = constraint_why(c_ge("hops", 2), g);
    CHECK(ab.find("did not name") != std::string::npos);
    CHECK(ab.find("hops") != std::string::npos);

    Geometry s;
    s.set_s("dtype", "fp8");
    const std::string in = constraint_why(c_in("dtype", "bf16"), s);
    CHECK(in.find("bf16") != std::string::npos);
    CHECK(in.find("fp8") != std::string::npos);

    const std::string miss = constraint_why(c_in("dtype", "bf16"), g);
    CHECK(miss.find("<absent>") != std::string::npos);
}

RAD_TEST_MAIN()
