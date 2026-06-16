// Violation-annealing perfect-halver search -- the "fresh engine" for breaking a
// comparator-count plateau the always-valid greedy ILS cannot leave (see Plan23
// PLAN-B, Track 2).  Target n=23 -> 77 (size-agnostic; works for any N/target).
//
// IDEA.  The greedy/ILS search only ever holds VALID nets, so it is trapped on
// the size-S plateau: there is no valid (S-1)-net one deletion away, and it
// cannot cross invalid intermediates.  Here we instead FIX the size at a target
// S and minimise the number of VIOLATING k-zero inputs
//        V(net) = #{ k-zero inputs x : after net, max(bottom) > min(top) }
// by simulated annealing over size-preserving moves (retarget an endpoint /
// reposition a comparator / replace a comparator).  If V hits 0 we have a valid
// S-comparator halver.  Start = a valid (S+1)-net minus its best single deletion
// (so V>0 but small).  Allowing V>0 lets the walk tunnel through the invalid
// region between the S+1 plateau and a hypothetical valid S-net.
//
// Validity core (build_kzero, bit-packed k-zero columns) is the SAME trusted
// machinery as search_halve_cegar*.cpp; viol_count just popcounts the violation
// columns instead of returning the first counterexample.  Any V==0 net found is
// still only a PROPOSAL -- gate it through tools/verify_small_halve{,_rev}.
//
// Build: g++ -O3 -march=native -std=c++23 optimizers/anneal_halve.cpp -o /tmp/anneal
// Run:   /tmp/anneal <N> <seed> <target_size> <steps> <restarts> [T0] [cooling]
//        reads start nets from /tmp/seeds_<N>.txt ; writes any valid <=target
//        net to /tmp/anneal_<N>_<seed>.txt and prints it.

#include <algorithm>
#include <cmath>
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
static std::vector<uint64_t> kcol;   // [N][KW] column-packed k-zero inputs

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
    long j = 0;
    for (uint64_t x = (1u << m) - 1; x < (1u << N);) {
        for (int i = 0; i < N; ++i)
            if ((x >> i) & 1) kcol[(size_t)i * KW + (j >> 6)] |= (1ull << (j & 63));
        ++j;
        uint64_t c = x & (~x + 1), r = x + c;
        x = (((r ^ x) >> 2) / c) | r;
    }
}

// Number of violating k-zero inputs (0 == valid halver).
static long viol_count(const Net& net) {
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
    long tot = 0;
    for (int k = 0; k < KW; ++k) {
        uint64_t B = 0, T = ~0ull;
        for (int i = 0; i < M; ++i) B |= s[(size_t)i * KW + k];
        for (int i = M; i < N; ++i) T &= s[(size_t)i * KW + k];
        tot += __builtin_popcountll(B & ~T);
    }
    return tot;
}

static std::mt19937_64* g_rng;
static Comp rand_comp() {
    std::uniform_int_distribution<int> d(0, N - 1);
    int a = d(*g_rng), b = d(*g_rng);
    while (a == b) b = d(*g_rng);
    if (a > b) std::swap(a, b);
    return {a, b};
}

// Reduce a valid net to exactly `S` comparators.  Each removal is chosen RANDOMLY
// among the few least-damaging deletions (smallest resulting V) -- randomised so
// repeated calls give diverse 73-net starts instead of one deterministic basin.
static Net reduce_to(Net net, int S) {
    while ((int)net.size() > S) {
        std::vector<std::pair<long, int>> cand;
        for (size_t i = 0; i < net.size(); ++i) {
            Net t = net; t.erase(t.begin() + i);
            long v = viol_count(t);
            cand.push_back({v, (int)i});
            if (v == 0) { cand = {{0, (int)i}}; break; }
        }
        std::sort(cand.begin(), cand.end());
        int top = (int)std::min<size_t>(5, cand.size());      // randomise among best 5
        int pick = cand[(*g_rng)() % top].second;
        net.erase(net.begin() + pick);
    }
    return net;
}

// Steepest-descent polish: try replacing each comparator with every other (a,b)
// and every reposition; keep the single best move that lowers V.  One sweep is
// O(|net| * N^2) viol_counts -- run only as a rare "punch" when V is already low.
static long polish(Net& net, long V) {
    bool improved = true;
    while (improved) {
        improved = false;
        long bestV = V; Net best = net;
        for (size_t i = 0; i < net.size(); ++i) {
            for (int a = 0; a < N; ++a)
                for (int b = a + 1; b < N; ++b) {
                    Net t = net; t[i] = {a, b};
                    long v = viol_count(t);
                    if (v < bestV) { bestV = v; best = t; }
                }
        }
        if (bestV < V) { net = best; V = bestV; improved = true; }
    }
    return V;
}

int main(int argc, char** argv) {
    N = (argc > 1) ? std::atoi(argv[1]) : 23;
    M = N / 2;
    unsigned long seed0 = (argc > 2) ? std::strtoul(argv[2], nullptr, 10) : 1;
    int S = (argc > 3) ? std::atoi(argv[3]) : 77;
    long steps = (argc > 4) ? std::atol(argv[4]) : 200000;
    int restarts = (argc > 5) ? std::atoi(argv[5]) : 8;
    double T0 = (argc > 6) ? std::atof(argv[6]) : 6.0;
    double cooling = (argc > 7) ? std::atof(argv[7]) : 0.9999;
    build_kzero();
    std::mt19937_64 rng(seed0);
    g_rng = &rng;

    // Load start nets (any valid halver; we reduce to S).
    std::vector<Net> seeds;
    {
        char sp[64]; std::snprintf(sp, sizeof sp, "/tmp/seeds_%d.txt", N);
        FILE* sf = std::fopen(sp, "r");
        if (!sf) { std::fprintf(stderr, "no %s\n", sp); return 2; }
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
        if (seeds.empty()) { std::fprintf(stderr, "empty seeds\n"); return 2; }
    }

    std::printf("N=%d M=%d k-zero=%ld target_size=%d steps=%ld restarts=%d T0=%.2f cool=%.5f\n",
                N, M, KC, S, steps, restarts, T0, cooling);

    std::uniform_real_distribution<double> U(0.0, 1.0);
    long global_best = -1; Net global_net;

    for (int r = 0; r < restarts; ++r) {
        Net cur = reduce_to(seeds[r % seeds.size()], S);
        long V = viol_count(cur);
        long bestV = V; Net bestNet = cur;
        double T = T0;
        int stagn = 0;
        for (long step = 0; step < steps && bestV > 0; ++step) {
            // size-preserving proposal
            Net nb = cur;
            int mv = rng() % 3;
            if (mv == 0) {                        // retarget one endpoint set
                nb[rng() % nb.size()] = rand_comp();
            } else if (mv == 1) {                 // reposition a comparator
                Comp c = nb[rng() % nb.size()];
                nb.erase(nb.begin() + (rng() % nb.size()));
                nb.insert(nb.begin() + (rng() % (nb.size() + 1)), c);
            } else {                              // retarget + reposition (bigger kick)
                size_t i = rng() % nb.size();
                nb.erase(nb.begin() + i);
                nb.insert(nb.begin() + (rng() % (nb.size() + 1)), rand_comp());
            }
            long V2 = viol_count(nb);
            long dV = V2 - V;
            if (dV <= 0 || U(rng) < std::exp(-(double)dV / T)) {
                cur = std::move(nb); V = V2;
                if (V < bestV) { bestV = V; bestNet = cur; stagn = 0; }
                else ++stagn;
            } else ++stagn;
            T *= cooling;
            if (stagn > 2500) {                     // full reheat on stagnation
                T = T0;
                // when already close to valid, punch with a steepest-descent sweep
                if (bestV > 0 && bestV <= 200) {
                    Net p = bestNet; long pv = polish(p, bestV);
                    if (pv < bestV) { bestV = pv; bestNet = p; cur = p; V = pv; }
                }
                stagn = 0;
            }
        }
        std::fprintf(stderr, "restart %d: bestV=%ld (size %d)%s\n",
                     r, bestV, S, bestV == 0 ? "  *** VALID ***" : "");
        if (global_best < 0 || bestV < global_best) { global_best = bestV; global_net = bestNet; }
        if (bestV == 0) break;
    }

    std::printf("=== BEST bestV=%ld size=%zu ===\n", global_best, global_net.size());
    if (global_best == 0) {
        for (auto [a, b] : global_net) std::printf("{%d,%d},", a, b);
        std::printf("\n");
        char path[80]; std::snprintf(path, sizeof path, "/tmp/anneal_%d_%lu.txt", N, seed0);
        FILE* f = std::fopen(path, "w");
        for (auto [a, b] : global_net) std::fprintf(f, "{%d,%d},", a, b);
        std::fprintf(f, "\n"); std::fclose(f);
        std::fprintf(stderr, "WROTE valid %d-CE net to %s\n", S, path);
    }
    return 0;
}
