// Enhanced large-N perfect-halver search, tuned to BREAK a comparator-count
// floor (target: n=24, 82 -> 81).  The validity core is IDENTICAL to
// search_halve_cegar.cpp (CEGAR witness pre-filter + monotonicity / k-zero
// exhaustive sweep) -- that part is the trusted proposer and is copied verbatim.
//
// WHAT IS DIFFERENT: the search DRIVER.  The original ILS perturbs a single fixed
// `best` and only moves sideways when depth drops, so it repeatedly probes one
// net's neighbourhood.  To find a doorway from the size-S plateau down to S-1 you
// want to WALK the plateau widely.  This driver does:
//   - sideways acceptance: move `cur` to a different same-size net w.p. P_SIDE,
//   - adaptive kick: escalate insert count on stagnation, reset on improvement,
//   - in-process reseed: on long stagnation, jump to a random pooled min net,
//   - bounded witness set (keeps quick_ok cheap at large iteration counts).
// The finally-chosen net STILL must pass tools/verify_small_halve{,_rev}.
//
// Build: g++ -O3 -march=native -std=c++23 optimizers/search_halve_cegar2.cpp -o /tmp/shb2
// Run:   /tmp/shb2 <N> <seed> <restarts> <iters> <poolK>
//        seeds from /tmp/seeds_<N>.txt ; pool to /tmp/h<N>_<seed>_pool.inc

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

static int N, M, KW;
static long KC;
static std::vector<uint64_t> kcol;
static std::vector<uint32_t> kmask;
static std::vector<uint32_t> witnesses;
static size_t WIT_CAP = 8000;          // bounded: keeps quick_ok cheaper than a full sweep

static void build_kzero() {
    const int m = N - M;
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

static std::mt19937_64* g_rng = nullptr;
static void add_witness(uint32_t w) {
    if (witnesses.size() < WIT_CAP) witnesses.push_back(w);
    else witnesses[(*g_rng)() % WIT_CAP] = w;   // reservoir-style replace, bounded cost
}

static bool valid(const Net& net) {
    if (!quick_ok(net)) return false;
    long cex = full_cex(net);
    if (cex >= 0) { add_witness((uint32_t)cex); return false; }
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

// Plateau-walking ILS.
static Net ils(Net seed, std::mt19937_64& rng, int iters) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const double P_SIDE = 0.35;        // accept a different same-size net (walk)
    const int ESCALATE = 60;           // stagnation steps per +1 kick strength
    const int RESTART_AFTER = 600;     // long stagnation -> jump to a pooled net

    Net cur = greedy_delete(std::move(seed), rng);
    pool_add(cur);
    Net best = cur;
    int stagn = 0;
    for (int it = 0; it < iters; ++it) {
        int R = 1 + stagn / ESCALATE;
        if (R > 6) R = 6;
        Net work = cur;
        for (int r = 0; r < R; ++r)
            work.insert(work.begin() + (rng() % (work.size() + 1)), rand_comp(rng));
        Net cand = greedy_delete(std::move(work), rng);

        if (cand.size() < cur.size()) {
            pool_add(cand);
            if (cand.size() < best.size() ||
                (cand.size() == best.size() && depth(cand) < depth(best)))
                best = cand;
            cur = std::move(cand);
            stagn = 0;
        } else if (cand.size() == cur.size()) {
            pool_add(cand);
            if (cand.size() == best.size() && depth(cand) < depth(best)) best = cand;
            if (cand != cur && U(rng) < P_SIDE) cur = std::move(cand);  // sideways walk
            ++stagn;
        } else {
            ++stagn;
        }

        if (stagn > RESTART_AFTER && !g_pool.empty()) {
            cur = g_pool[rng() % g_pool.size()];
            stagn = 0;
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

    std::mt19937_64 rng(seed0);
    g_rng = &rng;
    std::mt19937_64 wr(0xC0FFEEull + seed0);
    for (int t = 0; t < 3000 && KC > 0; ++t) witnesses.push_back(kmask[wr() % KC]);

    Net cur_h = seeds.front();
    std::printf("N=%d M=%d  k-zero=%ld  seeds=%zu  cur size=%zu d=%d valid=%s  WIT_CAP=%zu\n",
                N, M, KC, seeds.size(), cur_h.size(), depth(cur_h),
                full_cex(cur_h) < 0 ? "yes" : "NO", WIT_CAP);

    Net best = cur_h;
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
