// Depth-targeted perfect-halver search: hold the comparator count at a known floor
// S and MINIMIZE DEPTH.  The validity core (CEGAR witness pre-filter + k-zero
// monotonicity sweep) is identical to search_halve_cegar2.cpp.  What differs:
//
//   * depth_greedy_delete: among all VALID single-comparator removals, take the one
//     whose result has the SMALLEST depth (best-improvement on depth), not the first
//     removable one.  This steers prunings toward shallow nets.
//   * objective is (size, depth) lexicographic, but the plateau walk is driven by
//     DEPTH: at the floor size it accepts moves that lower depth and walks sideways
//     among equal-depth nets, so it explores the depth landscape at fixed size.
//   * stops and reports the first net reaching size==floor && depth<=DTARGET.
//
// Build: g++ -O3 -march=native -std=c++23 optimizers/search_halve_depth.cpp -o /tmp/shd
// Run:   /tmp/shd <N> <floorS> <Dtarget> <seed> <restarts> <iters> <poolK>
//        seeds from /tmp/seeds_<N>.txt ; pool to /tmp/h<N>d_<seed>_pool.inc
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
static size_t WIT_CAP = 8000;
static int FLOOR_S, D_TARGET;

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
    else witnesses[(*g_rng)() % WIT_CAP] = w;
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

// Best-improvement-on-depth greedy delete: at each step remove the comparator whose
// removal keeps the net valid AND yields the smallest depth; stop when no valid
// removal exists.  Ties on depth broken randomly (shuffled scan order).
static Net depth_greedy_delete(Net net, std::mt19937_64& rng) {
    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<int> idx(net.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
        std::shuffle(idx.begin(), idx.end(), rng);
        int bestd = 1 << 30, bestpos = -1;
        for (int i : idx) {
            Net trial = net;
            trial.erase(trial.begin() + i);
            if (valid(trial)) {
                int d = depth(trial);
                if (d < bestd) { bestd = d; bestpos = i; }
            }
        }
        if (bestpos >= 0) {
            net.erase(net.begin() + bestpos);
            changed = true;
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

static std::vector<Net> g_pool;          // floor-size nets, kept by lowest depth
static int g_pool_mindepth = 1 << 30;
static size_t g_pool_minsize = 9999;
static void pool_add(const Net& n) {
    if (n.size() > g_pool_minsize) return;
    int d = depth(n);
    if (n.size() < g_pool_minsize) {
        g_pool.clear(); g_pool_minsize = n.size(); g_pool_mindepth = d;
    }
    if (d > g_pool_mindepth + 1) return;        // keep only near-best depth
    if (d < g_pool_mindepth) { g_pool.clear(); g_pool_mindepth = d; }
    for (auto& e : g_pool) if (e == n) return;
    g_pool.push_back(n);
}

static Net g_hit;                        // first net hitting size==floor && depth<=target
static bool g_have_hit = false;

// Depth-driven plateau ILS at fixed (floor) size.
static Net ils(Net seed, std::mt19937_64& rng, int iters) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const double P_SIDE = 0.35;
    const int ESCALATE = 60;
    const int RESTART_AFTER = 600;

    Net cur = depth_greedy_delete(std::move(seed), rng);
    pool_add(cur);
    Net best = cur;
    int stagn = 0;
    for (int it = 0; it < iters && !g_have_hit; ++it) {
        int R = 1 + stagn / ESCALATE;
        if (R > 6) R = 6;
        Net work = cur;
        for (int r = 0; r < R; ++r)
            work.insert(work.begin() + (rng() % (work.size() + 1)), rand_comp(rng));
        Net cand = depth_greedy_delete(std::move(work), rng);

        bool better = cand.size() < best.size() ||
                      (cand.size() == best.size() && depth(cand) < depth(best));
        if (better) best = cand;
        if ((int)cand.size() == FLOOR_S && depth(cand) <= D_TARGET && !g_have_hit) {
            g_hit = cand; g_have_hit = true; return best;
        }

        if (cand.size() < cur.size()) {
            pool_add(cand); cur = std::move(cand); stagn = 0;
        } else if (cand.size() == cur.size()) {
            pool_add(cand);
            int dc = depth(cand), dd = depth(cur);
            if (dc < dd) { cur = std::move(cand); stagn = 0; }          // descend depth
            else if (dc == dd && cand != cur && U(rng) < P_SIDE) {      // walk sideways
                cur = std::move(cand); ++stagn;
            } else ++stagn;
        } else ++stagn;

        if (stagn > RESTART_AFTER && !g_pool.empty()) {
            cur = g_pool[rng() % g_pool.size()];
            stagn = 0;
        }
    }
    return best;
}

int main(int argc, char** argv) {
    N = (argc > 1) ? std::atoi(argv[1]) : 20;
    FLOOR_S = (argc > 2) ? std::atoi(argv[2]) : 60;
    D_TARGET = (argc > 3) ? std::atoi(argv[3]) : 12;
    M = N / 2;
    build_kzero();

    unsigned long seed0 = (argc > 4) ? std::strtoul(argv[4], nullptr, 10) : 1;
    int restarts = (argc > 5) ? std::atoi(argv[5]) : 8;
    int iters = (argc > 6) ? std::atoi(argv[6]) : 2000;
    int K = (argc > 7) ? std::atoi(argv[7]) : 60;

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
    std::printf("N=%d floor=%d Dtarget=%d  k-zero=%ld seeds=%zu  seed0 size=%zu d=%d\n",
                N, FLOOR_S, D_TARGET, KC, seeds.size(), cur_h.size(), depth(cur_h));

    Net best = cur_h;
    for (int r = 0; r < restarts && !g_have_hit; ++r) {
        Net res = ils(seeds[r % seeds.size()], rng, iters);
        if (res.size() < best.size() ||
            (res.size() == best.size() && depth(res) < depth(best)))
            best = res;
        std::fprintf(stderr, "restart %d: best=%zu/d%d pool=%zu@sz%zu/d%d wit=%zu%s\n",
                     r, best.size(), depth(best), g_pool.size(), g_pool_minsize,
                     g_pool_mindepth, witnesses.size(), g_have_hit ? "  HIT!" : "");
    }

    if (g_have_hit) {
        // double-check the hit is truly valid before celebrating
        bool ok = valid(g_hit);
        std::printf("=== HIT size=%zu depth=%d valid=%s ===\n", g_hit.size(),
                    depth(g_hit), ok ? "yes" : "NO");
        for (auto [a, b] : g_hit) std::printf("{%d,%d},", a, b);
        std::printf("\n");
        best = g_hit;
    }

    std::printf("=== BEST size=%zu depth=%d ===\n", best.size(), depth(best));
    for (auto [a, b] : best) std::printf("{%d,%d},", a, b);
    std::printf("\n");

    std::sort(g_pool.begin(), g_pool.end(),
              [](const Net& a, const Net& b) { return depth(a) < depth(b); });
    char path[80]; std::snprintf(path, sizeof path, "/tmp/h%dd_%lu_pool.inc", N, seed0);
    FILE* f = std::fopen(path, "w");
    int emitted = 0;
    for (auto& n : g_pool) {
        if (emitted >= K) break;
        std::fprintf(f, "inline constexpr std::array<P,%zu> p%d = {{", n.size(), emitted);
        for (auto [a, b] : n) std::fprintf(f, "{%d,%d},", a, b);
        std::fprintf(f, "}}; // depth %d\n", depth(n));
        ++emitted;
    }
    std::fclose(f);
    std::fprintf(stderr, "wrote %s (%d nets, minsize %zu mindepth %d)\n", path, emitted,
                 g_pool_minsize, g_pool_mindepth);
    return 0;
}
