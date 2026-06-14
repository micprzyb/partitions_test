// Standalone search for a small / shallow perfect halver on N inputs (N<=12),
// split@(N/2).  Generic version of search_halve12.cpp.
//
// Validity via the 0-1 principle over all 2^N inputs at once ("truth-table
// column" trick: wire i = 2^N-bit mask of its value per input; CE(a<b) is
// min=AND into a / max=OR into b).  Halver property (split@M=N/2): no input has
// a bottom wire = 1 while a top wire = 0, i.e. (OR_{0..M-1}) & ~(AND_{M..N-1})==0.
//
// Build: g++ -O3 -march=native -std=c++23 tools/search_halve.cpp -o /tmp/sh
// Run:   /tmp/sh <N> <seed> <restarts> <iters> <poolK>
//        (writes distinct minima to /tmp/hN_pool.inc)

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

using std::uint64_t;
using Comp = std::pair<int, int>;
using Net = std::vector<Comp>;

static int N, M, NW;                  // N inputs, split M=N/2, NW 64-bit words
// Flat truth-table state: wire i words live at [i*NW, i*NW+NW).  Flat (vs
// vector-of-vectors) makes the per-validity-check reset a single contiguous
// memcpy with no per-row allocation -- important at n>=16 where NW is large.
static std::vector<uint64_t> g_init;  // size N*NW

static void build_init() {
    NW = (1 << N) / 64;
    if (NW < 1) NW = 1;
    g_init.assign((size_t)N * NW, 0);
    for (int x = 0; x < (1 << N); ++x)
        for (int i = 0; i < N; ++i)
            if ((x >> i) & 1) g_init[(size_t)i * NW + (x >> 6)] |= (1ull << (x & 63));
}

static bool valid(const Net& net) {
    static thread_local std::vector<uint64_t> s;
    if (s.size() != g_init.size()) s.resize(g_init.size());
    std::memcpy(s.data(), g_init.data(), g_init.size() * sizeof(uint64_t));
    for (auto [a, b] : net) {
        uint64_t* wa = s.data() + (size_t)a * NW;
        uint64_t* wb = s.data() + (size_t)b * NW;
        for (int k = 0; k < NW; ++k) {
            uint64_t lo = wa[k] & wb[k], hi = wa[k] | wb[k];
            wa[k] = lo;
            wb[k] = hi;
        }
    }
    for (int k = 0; k < NW; ++k) {
        uint64_t B = 0, T = ~0ull;
        for (int i = 0; i < M; ++i) B |= s[(size_t)i * NW + k];
        for (int i = M; i < N; ++i) T &= s[(size_t)i * NW + k];
        if (B & ~T) return false;
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

static void print_net(const Net& net) {
    std::printf("size=%zu depth=%d: ", net.size(), depth(net));
    for (auto [a, b] : net) std::printf("{%d,%d},", a, b);
    std::printf("\n");
}

int main(int argc, char** argv) {
    N = (argc > 1) ? std::atoi(argv[1]) : 12;
    M = N / 2;
    build_init();

    // Seeds: read every net (one per line, "{a,b},{c,d},...") from
    // /tmp/seeds_<N>.txt.  Put the current halver first (it seeds best_*).
    std::vector<Net> seeds;
    {
        char sp[64]; std::snprintf(sp, sizeof sp, "/tmp/seeds_%d.txt", N);
        FILE* sf = std::fopen(sp, "r");
        if (!sf) { std::fprintf(stderr, "no seed file %s\n", sp); return 2; }
        char* line = nullptr; size_t cap = 0; ssize_t len;
        while ((len = getline(&line, &cap, sf)) != -1) {
            Net net; int a, b; const char* p = line;
            while (*p) {
                if (*p == '{') {
                    if (std::sscanf(p, "{%d,%d}", &a, &b) == 2 &&
                        a >= 0 && b >= 0 && a < N && b < N && a != b)
                        net.push_back({a, b});
                }
                ++p;
            }
            if (!net.empty()) seeds.push_back(std::move(net));
        }
        std::free(line); std::fclose(sf);
        if (seeds.empty()) { std::fprintf(stderr, "empty seed file %s\n", sp); return 2; }
    }
    Net cur_h = seeds.front();
    std::printf("N=%d M=%d  seeds=%zu  cur_h valid=%d size=%zu depth=%d\n",
                N, M, seeds.size(), valid(cur_h), cur_h.size(), depth(cur_h));

    unsigned long seed0 = (argc > 2) ? std::strtoul(argv[2], nullptr, 10) : 12345;
    int restarts = (argc > 3) ? std::atoi(argv[3]) : 200;
    int iters = (argc > 4) ? std::atoi(argv[4]) : 400;
    int K = (argc > 5) ? std::atoi(argv[5]) : 40;

    Net best_size = cur_h, best_depth = cur_h;
    auto consider = [&](const Net& n) {
        if (!valid(n)) return;
        if (n.size() < best_size.size() ||
            (n.size() == best_size.size() && depth(n) < depth(best_size))) best_size = n;
        if (depth(n) < depth(best_depth) ||
            (depth(n) == depth(best_depth) && n.size() < best_depth.size())) best_depth = n;
    };

    std::mt19937_64 rng(seed0);
    for (int r = 0; r < restarts; ++r) {
        Net seed = seeds[r % seeds.size()];
        consider(ils(seed, rng, iters));
        if (r % 20 == 0)
            std::fprintf(stderr, "restart %d: best_size=%zu (d%d) best_depth=%d (s%zu) pool=%zu@%zu\n",
                         r, best_size.size(), depth(best_size), depth(best_depth),
                         best_depth.size(), g_pool.size(), g_pool_minsize);
    }

    std::printf("=== BEST BY SIZE ===\n"); print_net(best_size);
    std::printf("=== BEST BY DEPTH ===\n"); print_net(best_depth);

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
