// Large-N perfect-halver search (n up to ~24) by iterated local search, with a
// CEGAR validity check and the monotonicity ("k-zero inputs only") reduction.
//
// WHY THIS TOOL (vs search_halve.cpp): at large n the full 2^n truth table is
// huge (n=24 -> 48 MB per check), so re-materialising it per candidate is
// memory-bandwidth-bound and ~100x too slow.  This tool instead:
//
//   1. MONOTONICITY REDUCTION (idea borrowed from the SAT finders in python/):
//      a comparator network is a valid split@k halver iff it correctly halves
//      every 0/1 input with EXACTLY k zeros (== popcount(x)==m, m=n-k).  Any
//      other input reduces to a k-zero one (flip spare zeros up / spare ones
//      down) and comparator networks are monotone, so the k-zero inputs are
//      sufficient.  That is C(n,k) inputs instead of 2^n -- ~6.3x fewer at n=24
//      (measured: 11.9 ms -> 1.9 ms for the exhaustive sweep) and the gap grows
//      with n.  We pack those inputs as bit-columns and AND/OR the network over
//      them.
//
//   2. CEGAR: most trial nets are invalid, so we first test a growing set of
//      WITNESS inputs (counterexamples seen before) cheaply; only nets that
//      survive all witnesses pay the full k-zero sweep, whose counterexample is
//      appended to the witness set.  A net is accepted only if the FULL k-zero
//      sweep finds no violation -- so no invalid net is ever accepted.
//
// The finally-chosen net must ALSO pass the repo's independent exhaustive
// verifier (tools/verify_small_halve, real 2^n) before being committed.
//
// Build: g++ -O3 -march=native -std=c++23 optimizers/search_halve_cegar.cpp -o /tmp/shb
// Run:   /tmp/shb <N> <seed> <restarts> <iters> <poolK>
//        seeds  from /tmp/seeds_<N>.txt (one net per line, "{a,b},{c,d},...")
//        pool   to   /tmp/h<N>_<seed>_pool.inc

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

using std::uint32_t;
using std::uint64_t;
using Comp = std::pair<int, int>;
using Net = std::vector<Comp>;

static int N, M, KW;                  // N inputs, split M=N/2, KW = k-zero words
static long KC;                       // number of k-zero inputs = C(n, n-M)
static std::vector<uint64_t> kcol;    // [N][KW]: column-packed k-zero inputs
static std::vector<uint32_t> kmask;   // [KC]: the input mask for each column bit
static std::vector<uint32_t> witnesses;

// Build the column-packed k-zero (popcount == m == n-M) inputs once.
static void build_kzero() {
    const int m = N - M;              // number of ONE bits in a k-zero input
    KC = 0;
    for (uint64_t x = (1u << m) - 1; x < (1u << N);) {
        ++KC;
        uint64_t c = x & (~x + 1), r = x + c;
        x = (((r ^ x) >> 2) / c) | r;
    }
    KW = (int)((KC + 63) / 64);
    kcol.assign((size_t)N * KW, 0);
    kmask.assign(KC, 0);
    long j = 0;
    for (uint64_t x = (1u << m) - 1; x < (1u << N);) {
        kmask[j] = (uint32_t)x;
        for (int i = 0; i < N; ++i)
            if ((x >> i) & 1) kcol[(size_t)i * KW + (j >> 6)] |= (1ull << (j & 63));
        ++j;
        uint64_t c = x & (~x + 1), r = x + c;
        x = (((r ^ x) >> 2) / c) | r;
    }
}

// Exhaustive (over k-zero inputs) check.  Returns a violating input mask, or -1.
static long full_cex(const Net& net) {
    static thread_local std::vector<uint64_t> s;
    if (s.size() != kcol.size()) s.resize(kcol.size());
    std::memcpy(s.data(), kcol.data(), kcol.size() * sizeof(uint64_t));
    for (auto [a, b] : net) {
        uint64_t* wa = s.data() + (size_t)a * KW;
        uint64_t* wb = s.data() + (size_t)b * KW;
        for (int k = 0; k < KW; ++k) {
            uint64_t lo = wa[k] & wb[k], hi = wa[k] | wb[k];
            wa[k] = lo;
            wb[k] = hi;
        }
    }
    for (int k = 0; k < KW; ++k) {
        uint64_t B = 0, T = ~0ull;
        for (int i = 0; i < M; ++i) B |= s[(size_t)i * KW + k];
        for (int i = M; i < N; ++i) T &= s[(size_t)i * KW + k];
        uint64_t viol = B & ~T;
        if (viol) {
            long idx = (long)k * 64 + __builtin_ctzll(viol);
            return (idx < KC) ? (long)kmask[idx] : -1;
        }
    }
    return -1;
}

static bool quick_ok(const Net& net) {
    int a[32];
    for (uint32_t w : witnesses) {
        for (int i = 0; i < N; ++i) a[i] = (w >> i) & 1;
        for (auto [x, y] : net) {
            int lo = a[x] & a[y], hi = a[x] | a[y];
            a[x] = lo;
            a[y] = hi;
        }
        int B = 0, T = 1;
        for (int i = 0; i < M; ++i) B |= a[i];
        for (int i = M; i < N; ++i) T &= a[i];
        if (B & ~T) return false;
    }
    return true;
}

static bool valid(const Net& net) {
    if (!quick_ok(net)) return false;
    long cex = full_cex(net);
    if (cex >= 0) {
        if (witnesses.size() < 300000) witnesses.push_back((uint32_t)cex);
        return false;
    }
    return true;
}

static int depth(const Net& net) {
    std::vector<int> last(N, 0);
    int d = 0;
    for (auto [a, b] : net) {
        int L = std::max(last[a], last[b]) + 1;
        last[a] = last[b] = L;
        if (L > d) d = L;
    }
    return d;
}

static Net greedy_delete(Net net, std::mt19937_64& rng) {
    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<int> idx(net.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
        std::shuffle(idx.begin(), idx.end(), rng);
        for (int i : idx) {
            Net trial = net;
            trial.erase(trial.begin() + i);
            if (valid(trial)) { net = std::move(trial); changed = true; break; }
        }
    }
    return net;
}

static Comp rand_comp(std::mt19937_64& rng) {
    std::uniform_int_distribution<int> d(0, N - 1);
    int a = d(rng), b = d(rng);
    while (a == b) b = d(rng);
    if (a > b) std::swap(a, b);
    return {a, b};
}

static std::vector<Net> g_pool;
static size_t g_pool_minsize = 9999;
static void pool_add(const Net& n) {
    if (n.size() > g_pool_minsize) return;
    if (n.size() < g_pool_minsize) { g_pool.clear(); g_pool_minsize = n.size(); }
    for (auto& e : g_pool) if (e == n) return;
    g_pool.push_back(n);
}

static Net ils(Net seed, std::mt19937_64& rng, int iters) {
    Net best = greedy_delete(seed, rng);
    if (valid(best)) pool_add(best);
    for (int it = 0; it < iters; ++it) {
        Net cur = best;
        int R = 1 + (int)(rng() % 4);
        for (int r = 0; r < R; ++r)
            cur.insert(cur.begin() + (rng() % (cur.size() + 1)), rand_comp(rng));
        cur = greedy_delete(cur, rng);
        if (valid(cur)) {
            if (cur.size() <= best.size()) pool_add(cur);
            if (cur.size() < best.size() ||
                (cur.size() == best.size() && depth(cur) < depth(best)))
                best = std::move(cur);
        }
    }
    return best;
}

int main(int argc, char** argv) {
    N = (argc > 1) ? std::atoi(argv[1]) : 24;
    M = N / 2;
    build_kzero();

    unsigned long seed0 = (argc > 2) ? std::strtoul(argv[2], nullptr, 10) : 1;
    int restarts = (argc > 3) ? std::atoi(argv[3]) : 8;
    int iters = (argc > 4) ? std::atoi(argv[4]) : 1500;
    int K = (argc > 5) ? std::atoi(argv[5]) : 40;

    std::vector<Net> seeds;
    {
        char sp[64]; std::snprintf(sp, sizeof sp, "/tmp/seeds_%d.txt", N);
        FILE* sf = std::fopen(sp, "r");
        if (!sf) { std::fprintf(stderr, "no seed file %s\n", sp); return 2; }
        char* line = nullptr; size_t cap = 0;
        while (getline(&line, &cap, sf) != -1) {
            Net net; int a, b;
            for (const char* p = line; *p; ++p)
                if (*p == '{' && std::sscanf(p, "{%d,%d}", &a, &b) == 2 &&
                    a >= 0 && b >= 0 && a < N && b < N && a != b)
                    net.push_back({a, b});
            if (!net.empty()) seeds.push_back(std::move(net));
        }
        std::free(line); std::fclose(sf);
        if (seeds.empty()) { std::fprintf(stderr, "empty seeds %s\n", sp); return 2; }
    }

    // Pre-seed witnesses with random k-zero inputs (the hard region).
    std::mt19937_64 wr(0xC0FFEEull + seed0);
    for (int t = 0; t < 3000 && KC > 0; ++t) witnesses.push_back(kmask[wr() % KC]);

    Net cur_h = seeds.front();
    std::printf("N=%d M=%d  k-zero inputs=%ld (C(%d,%d))  seeds=%zu  cur_h size=%zu d=%d valid=%s\n",
                N, M, KC, N, N - M, seeds.size(), cur_h.size(), depth(cur_h),
                full_cex(cur_h) < 0 ? "yes" : "NO");

    Net best = cur_h;
    std::mt19937_64 rng(seed0);
    for (int r = 0; r < restarts; ++r) {
        Net res = ils(seeds[r % seeds.size()], rng, iters);
        if (valid(res) && (res.size() < best.size() ||
                           (res.size() == best.size() && depth(res) < depth(best))))
            best = res;
        std::fprintf(stderr, "restart %d: best=%zu (d%d) pool=%zu@%zu wit=%zu\n",
                     r, best.size(), depth(best), g_pool.size(), g_pool_minsize,
                     witnesses.size());
    }

    std::printf("=== BEST ===\nsize=%zu depth=%d: ", best.size(), depth(best));
    for (auto [a, b] : best) std::printf("{%d,%d},", a, b);
    std::printf("\n");

    std::sort(g_pool.begin(), g_pool.end(),
              [](const Net& a, const Net& b) { return depth(a) < depth(b); });
    char path[80]; std::snprintf(path, sizeof path, "/tmp/h%d_%lu_pool.inc", N, seed0);
    FILE* f = std::fopen(path, "w");
    int emitted = 0;
    for (auto& n : g_pool) {
        if (emitted >= K) break;
        std::fprintf(f, "inline constexpr std::array<P,%zu> p%d = {{", n.size(), emitted);
        for (auto [a, b] : n) std::fprintf(f, "{%d,%d},", a, b);
        std::fprintf(f, "}}; // depth %d\n", depth(n));
        ++emitted;
    }
    std::fprintf(f, "#define HN_POOL_N %d\n#define HN_POOL_LIST ", emitted);
    for (int i = 0; i < emitted; ++i) std::fprintf(f, "X(p%d) ", i);
    std::fprintf(f, "\n");
    std::fclose(f);
    std::fprintf(stderr, "wrote %s (%d nets, minsize %zu)\n", path, emitted, g_pool_minsize);
    return 0;
}
