// bench_cyclic_partitions.cpp -- benchmark matrix for the cyclic partition /
// cyclic offset partition kernels in refactored_partitions.hpp (plan:
// cyclic_partitions.txt section 9).  CSV to stdout.
//
//   g++ -std=c++23 -O3 -march=native bench_cyclic_partitions.cpp -o bench_cyc
//   taskset -c 0 ./bench_cyc plain   > plain.csv
//   taskset -c 0 ./bench_cyc offset  > offset.csv
//
// Methodology: min ns/elem over reps (min filters scheduler noise); batched
// independent blocks for small n (defeats branch-predictor memorization and
// amortizes call/timer overhead); data restored from a pristine copy OUTSIDE
// the timed region; fixed seeds; contenders interleaved inside each rep so
// thermal drift hits all of them equally.

#include "refactored_partitions.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

using i32 = std::int32_t;
using i64 = std::int64_t;
using p32 = std::pair<i32, i32>;
using p64 = std::pair<i64, i64>;
using pf32 = std::pair<float, i32>;
using pd64 = std::pair<double, i64>;
using LL = partitions::detail::cyclic::lex_less_branchless;
using LS = partitions::detail::net::lex_less_semibranch;
using LP = partitions::detail::net::lex_less_packed;

static inline std::uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static volatile std::int64_t g_sink;

// value domain [0, 2^20); key at percentile p
constexpr std::uint64_t kDom = 1ull << 20;

template <class T>
T mk(std::uint64_t f, std::uint64_t s);
template <> i32 mk<i32>(std::uint64_t f, std::uint64_t) { return (i32)f; }
template <> i64 mk<i64>(std::uint64_t f, std::uint64_t) { return (i64)f; }
template <> p32 mk<p32>(std::uint64_t f, std::uint64_t s) {
    return {(i32)f, (i32)s};
}
template <> p64 mk<p64>(std::uint64_t f, std::uint64_t s) {
    return {(i64)f, (i64)s};
}
template <> pf32 mk<pf32>(std::uint64_t f, std::uint64_t s) {
    return {(float)f, (i32)s};  // exact: f < 2^20 < 2^24
}
template <> pd64 mk<pd64>(std::uint64_t f, std::uint64_t s) {
    return {(double)f, (i64)s};
}

struct first_proj {
    template <class P>
    auto operator()(const P& p) const { return p.first; }
};

enum algo : int {
    A_FLAT = 0,    // flat sized_partition, contiguous layout (reference)
    A_DISPATCH,    // cyclic_partition_count
    A_GAP,         // gap_cyclic
    A_SEGBR,       // seg_bridge
    A_SEGFILL,     // seg_fill
    A_HOARE,       // wrap_hoare (naive-cost exhibit)
    // offset bench:
    O_FLAT,        // flat offset_partition (reference)
    O_DISPATCH,    // cyclic_offset_partition
    O_GAP,         // gap_cyclic(offset)
    O_WFILL,       // off_w1_fill / off_w2_fill
    O_WSEG,        // off_w1_seg / off_w2_seg
};
static const char* algo_name[] = {"flat",  "dispatch", "gap",   "seg_bridge",
                                  "seg_fill", "hoare", "flat_off", "dispatch",
                                  "gap",   "w_fill",   "w_seg"};

// One benchmark cell: nblocks independent cyclic buffers of capacity cap laid
// out back to back; each holds a logical range of length n with len2 wrapped.
template <class T, class Key, class Comp, class Proj>
void run_cell(const char* tname, const char* cname, const char* mode,
              std::ptrdiff_t n, double wfrac, double offfrac, double p,
              const std::vector<int>& algos) {
    using D = std::ptrdiff_t;
    const D cap = n + 8;
    // genuinely wrapped: the direct kernel calls require len1,len2 >= 1
    const D len2 = std::clamp<D>((D)(wfrac * (double)n + 0.5), 1, n - 1);
    const D len1 = n - len2;
    const D f = cap - len1;  // S1 = [f, cap), S2 = [0, len2)
    const D off = std::min<D>((D)(offfrac * (double)n + 0.5), n);
    const bool offset_mode = std::string(mode) == "offset";

    std::size_t nblocks =
        std::max<std::size_t>(1, (1ull << 22) / ((std::size_t)cap * sizeof(T)));
    const std::size_t total = nblocks * (std::size_t)cap;

    // deterministic per-cell seed
    std::mt19937_64 rng(0x9e3779b97f4a7c15ull ^ (std::uint64_t)n * 0x10001 ^
                        (std::uint64_t)(wfrac * 1000) * 131 ^
                        (std::uint64_t)(offfrac * 1000) * 137 ^
                        (std::uint64_t)(p * 1000) * 139);

    const std::uint64_t kf = (std::uint64_t)(p * (double)kDom);
    Key key;
    if constexpr (std::is_same_v<Key, T>)
        key = mk<T>(kf, 0);  // lex: key = {kf, 0} (or the scalar kf itself)
    else
        key = (Key)kf;  // first-only projection: projected key

    // pristine: the cyclic layout; pristine_flat: same logical sequence laid
    // out contiguously from the block start (the flat reference must see the
    // same values at the same LOGICAL positions -- in offset mode the >= key
    // prefix has to stay the logical prefix).
    std::vector<T> pristine(total), pristine_flat(total), work(total);
    for (std::size_t b = 0; b < nblocks; ++b) {
        T* blk = pristine.data() + b * cap;
        T* blkf = pristine_flat.data() + b * cap;
        // logical range: prefix [0, off) all >= key (offset mode), rest random
        for (D k = 0; k < n; ++k) {
            std::uint64_t vf = rng() % kDom;
            if (offset_mode && k < off) vf = kf + rng() % kDom;  // >= key
            T v = mk<T>(vf, rng() % kDom);
            blk[(f + k) % cap] = v;
            blkf[k] = v;
        }
    }

    Comp comp{};
    Proj proj{};
    const int reps = n <= 4096 ? 21 : n <= 65536 ? 15 : 9;
    std::vector<double> best(algos.size(), 1e30);

    for (int rep = 0; rep < reps; ++rep) {
        for (std::size_t ai = 0; ai < algos.size(); ++ai) {
            const int a = algos[ai];
            const auto& src =
                (a == A_FLAT || a == O_FLAT) ? pristine_flat : pristine;
            std::copy(src.begin(), src.end(), work.begin());
            std::int64_t sink = 0;
            const std::uint64_t t0 = now_ns();
            for (std::size_t b = 0; b < nblocks; ++b) {
                T* bb = work.data() + b * cap;
                T* be = bb + cap;
                T* first = bb + f;
                T* last = bb + len2;
                switch (a) {
                    case A_FLAT:
                        // contiguous layout of the same values: logical range
                        // starts at bb (cap > n so it fits flat)
                        sink += partitions::sized_partition(bb, bb + n, key,
                                                            comp, proj) - bb;
                        break;
                    case A_DISPATCH:
                        sink += partitions::cyclic_partition_count(
                            first, last, bb, be, key, comp, proj);
                        break;
                    case A_GAP:
                        sink += partitions::detail::cyclic::gap_cyclic(
                            first, last, bb, be, (D)0, key, comp, proj);
                        break;
                    case A_SEGBR:
                        sink += partitions::detail::cyclic::seg_bridge(
                            first, last, bb, be, key, comp, proj);
                        break;
                    case A_SEGFILL:
                        sink += partitions::detail::cyclic::seg_fill(
                            first, last, bb, be, key, comp, proj);
                        break;
                    case A_HOARE:
                        sink += partitions::detail::cyclic::wrap_hoare(
                            first, last, bb, be, key, comp, proj);
                        break;
                    case O_FLAT:
                        sink += partitions::offset_partition(bb, bb + n, off,
                                                             key, comp, proj);
                        break;
                    case O_DISPATCH:
                        sink += partitions::cyclic_offset_partition(
                            first, last, bb, be, off, key, comp, proj);
                        break;
                    case O_GAP:
                        if (off < n)
                            sink += partitions::detail::cyclic::gap_cyclic(
                                first, last, bb, be, off, key, comp, proj);
                        break;
                    case O_WFILL:
                        sink += off <= len1
                                    ? partitions::detail::cyclic::off_w1_fill(
                                          first, last, bb, be, off, key, comp,
                                          proj)
                                    : partitions::detail::cyclic::off_w2_fill(
                                          first, last, bb, be, off, key, comp,
                                          proj);
                        break;
                    case O_WSEG:
                        sink += off <= len1
                                    ? partitions::detail::cyclic::off_w1_seg(
                                          first, last, bb, be, off, key, comp,
                                          proj)
                                    : partitions::detail::cyclic::off_w2_seg(
                                          first, last, bb, be, off, key, comp,
                                          proj);
                        break;
                }
            }
            const std::uint64_t t1 = now_ns();
            g_sink += sink;
            const double ns =
                (double)(t1 - t0) / ((double)nblocks * (double)n);
            if (ns < best[ai]) best[ai] = ns;
        }
    }
    for (std::size_t ai = 0; ai < algos.size(); ++ai)
        std::printf("%s,%s,%s,%td,%.4f,%.4f,%.2f,%s,%.4f\n", mode, tname,
                    cname, n, wfrac, offfrac, p, algo_name[algos[ai]],
                    best[ai]);
    std::fflush(stdout);
}

// ------------------------------------------------------- find-min benchmark
// mode "min": ns/CALL is the honest unit at n < 12 (ns/elem printed too).
// find_min does not mutate, so no per-rep restore; the working set is kept
// large enough (>= 2 MiB of blocks) that the branch predictor cannot
// memorize the wide kernel's update pattern across reps.

#if defined(__AVX2__)
// M4 prototype: masked-load SIMD min+position for i64 identity, n <= 12
// ONLY (vs[]/bases[] sized for ceil(12/4)+1 chunks per segment; larger n
// must not route here).  One vpmaskmov per <=4-lane chunk, +INF blend for
// dead lanes, vector min by cmpgt+blend, horizontal min, position by
// cmpeq+movmsk+tzcnt.
static inline i64* simd_min_i64(i64* first, i64* last, i64* bb, i64* be) {
    alignas(32) static const i64 kMaskLut[8] = {-1, -1, -1, -1, 0, 0, 0, 0};
    const __m256i vinf = _mm256_set1_epi64x(INT64_MAX);
    __m256i vs[8], vbest = vinf;
    i64* bases[8];
    int nv = 0;
    auto seg = [&](i64* s, i64* e) {
        for (i64* p = s; p < e; p += 4) {
            std::size_t rem = (std::size_t)(e - p);
            const __m256i m = _mm256_loadu_si256(
                (const __m256i*)(kMaskLut + (rem >= 4 ? 0 : 4 - rem)));
            __m256i v = _mm256_maskload_epi64((const long long*)p, m);
            v = _mm256_blendv_epi8(vinf, v, m);
            bases[nv] = p;
            vs[nv++] = v;
            vbest = _mm256_blendv_epi8(vbest, v,
                                       _mm256_cmpgt_epi64(vbest, v));
        }
    };
    seg(first, be);
    seg(bb, last);
    // horizontal min of vbest
    __m256i t = _mm256_permute4x64_epi64(vbest, 0x4E);
    vbest = _mm256_blendv_epi8(vbest, t, _mm256_cmpgt_epi64(vbest, t));
    t = _mm256_permute4x64_epi64(vbest, 0xB1);
    vbest = _mm256_blendv_epi8(vbest, t, _mm256_cmpgt_epi64(vbest, t));
    // position: first lane equal to the min
    for (int i = 0; i < nv; ++i) {
        unsigned mm = (unsigned)_mm256_movemask_pd(
            _mm256_castsi256_pd(_mm256_cmpeq_epi64(vs[i], vbest)));
        if (mm) return bases[i] + std::countr_zero(mm);
    }
    return first;  // unreachable for n >= 1
}
#endif

enum malgo : int {
    M_DISPATCH = 0,  // cyclic_find_min (user path)
    M_M1,            // forced unified single-accumulator
    M_M2,            // forced unified two-accumulator
    M_M3,            // per-segment find_min + combine
    M_FLAT,          // partitions::find_min on a flat layout (reference)
    M_STD,           // std::min_element on a flat layout (floor exhibit)
    M_NAIVE,         // logical %-indexed loop (naive exhibit)
    M_SIMD,          // M4 prototype (i64 identity only)
};
static const char* malgo_name[] = {"dispatch", "m1",  "m2",    "m3",
                                   "flat",     "std", "naive", "simd"};

template <class T, class Key, class Comp, class Proj>
void run_min_cell(const char* tname, const char* cname, std::ptrdiff_t n,
                  double wfrac, const std::vector<int>& algos) {
    using D = std::ptrdiff_t;
    const D cap = n + 8;
    const D len2 = (n >= 2 && wfrac > 0.0)
                       ? std::clamp<D>((D)(wfrac * (double)n + 0.5), 1, n - 1)
                       : 0;
    const D len1 = n - len2;
    const D f = len2 > 0 ? cap - len1 : 0;
    std::size_t nblocks = std::max<std::size_t>(
        512, (1ull << 21) / ((std::size_t)cap * sizeof(T)));
    const std::size_t total = nblocks * (std::size_t)cap;
    std::mt19937_64 rng(0xabcdef ^ (std::uint64_t)n * 7919 ^
                        (std::uint64_t)(wfrac * 1000));
    std::vector<T> buf(total), flat(total);
    for (std::size_t b = 0; b < nblocks; ++b)
        for (D k = 0; k < n; ++k) {
            T v = mk<T>(rng() % kDom, rng() % kDom);
            buf[b * cap + (std::size_t)((f + k) % cap)] = v;
            flat[b * cap + (std::size_t)k] = v;
        }
    Comp comp{};
    Proj proj{};
    // effective substituted comparator/projection (what the dispatcher runs;
    // forced kernels must be measured with the same predicate to be fair):
    // min scans keep the RAW comparator except the packed-u64 int-pair image
    constexpr bool packed =
        partitions::detail::minsel::use_packed_min<T*, Comp, Proj>;
    auto ecomp = [&] {
        if constexpr (packed) return std::less<>{};
        else return comp;
    }();
    auto eproj = [&] {
        if constexpr (packed) return partitions::detail::minsel::pack_proj{};
        else return proj;
    }();
    using EK = std::remove_cvref_t<decltype(std::invoke(eproj, buf[0]))>;

    const int reps = 25;
    std::vector<double> best(algos.size(), 1e30);
    for (int rep = 0; rep < reps; ++rep) {
        for (std::size_t ai = 0; ai < algos.size(); ++ai) {
            std::int64_t sink = 0;
            const std::uint64_t t0 = now_ns();
            for (std::size_t b = 0; b < nblocks; ++b) {
                T* bb = buf.data() + b * cap;
                T* be = bb + cap;
                T* first = bb + f;
                T* last = bb + len2;  // len2==0: flat placement at bb
                T* fl = flat.data() + b * cap;
                if (len2 == 0) { first = fl; last = fl + n; bb = fl; be = fl + cap; }
                switch (algos[ai]) {
                    case M_DISPATCH:
                        sink += partitions::cyclic_find_min(first, last, bb,
                                                            be, comp, proj) -
                                bb;
                        break;
                    case M_M1: {
                        EK bst = std::invoke(eproj, *first);
                        T* pos = first;
                        if (len2 > 0) {
                            partitions::detail::minsel::scan1(first + 1, be,
                                                              bst, pos, ecomp,
                                                              eproj);
                            partitions::detail::minsel::scan1(bb, last, bst,
                                                              pos, ecomp,
                                                              eproj);
                        } else {
                            partitions::detail::minsel::scan1(first + 1, last,
                                                              bst, pos, ecomp,
                                                              eproj);
                        }
                        sink += pos - bb;
                        break;
                    }
                    case M_M2: {
                        EK b0 = std::invoke(eproj, *first), b1 = b0;
                        T *p0 = first, *p1 = first;
                        if (len2 > 0) {
                            partitions::detail::minsel::scan2(
                                first + 1, be, b0, p0, b1, p1, ecomp, eproj);
                            partitions::detail::minsel::scan2(
                                bb, last, b0, p0, b1, p1, ecomp, eproj);
                        } else {
                            partitions::detail::minsel::scan2(
                                first + 1, last, b0, p0, b1, p1, ecomp, eproj);
                        }
                        sink += (std::invoke(ecomp, b1, b0) ? p1 : p0) - bb;
                        break;
                    }
                    case M_M3: {
                        if (len2 > 0) {
                            T* m1 = partitions::find_min(first, be, comp, proj);
                            T* m2 = partitions::find_min(bb, last, comp, proj);
                            sink += (std::invoke(comp, std::invoke(proj, *m2),
                                                 std::invoke(proj, *m1))
                                         ? m2
                                         : m1) -
                                    bb;
                        } else {
                            sink += partitions::find_min(first, last, comp,
                                                         proj) -
                                    bb;
                        }
                        break;
                    }
                    case M_FLAT:
                        sink += partitions::find_min(fl, fl + n, comp, proj) -
                                fl;
                        break;
                    case M_STD:
                        sink += std::min_element(
                                    fl, fl + n,
                                    [&](const T& a, const T& b) {
                                        return std::invoke(
                                            comp, std::invoke(proj, a),
                                            std::invoke(proj, b));
                                    }) -
                                fl;
                        break;
                    case M_NAIVE: {
                        std::size_t mi = 0;
                        for (std::size_t k = 1; k < (std::size_t)n; ++k)
                            if (std::invoke(
                                    comp,
                                    std::invoke(proj,
                                                bb[(f + (D)k) % cap]),
                                    std::invoke(proj, bb[(f + (D)mi) % cap])))
                                mi = k;
                        sink += (D)mi;
                        break;
                    }
                    case M_SIMD:
#if defined(__AVX2__)
                        if constexpr (std::is_same_v<T, i64> &&
                                      std::is_same_v<Proj, std::identity>) {
                            if (n > 12) break;  // prototype bound (vs[8])
                            if (len2 > 0)
                                sink += simd_min_i64(first, be, bb, last) - bb;
                            else
                                sink += simd_min_i64(first, last, last, last) -
                                        bb;
                        }
#endif
                        break;
                }
            }
            const std::uint64_t t1 = now_ns();
            g_sink += sink;
            const double nscall = (double)(t1 - t0) / (double)nblocks;
            if (nscall < best[ai]) best[ai] = nscall;
        }
    }
    for (std::size_t ai = 0; ai < algos.size(); ++ai)
        std::printf("min,%s,%s,%td,%.4f,0,0.50,%s,%.4f\n", tname, cname, n,
                    wfrac, malgo_name[algos[ai]], best[ai]);
    std::fflush(stdout);
}

template <class T, class Key, class Comp, class Proj>
void min_type(const char* tname, const char* cname, bool with_simd) {
    std::vector<int> algos = {M_DISPATCH, M_M1, M_M2, M_M3,
                              M_FLAT,     M_STD, M_NAIVE};
    if (with_simd) algos.push_back(M_SIMD);
    for (std::ptrdiff_t n : {2, 3, 4, 5, 6, 8, 10, 11, 12, 16, 24, 32, 64,
                             128, 1024})
        for (double w : {0.5, 0.125})  // mid split + small-S2 split
            run_min_cell<T, Key, Comp, Proj>(tname, cname, n, w, algos);
    // flat placement reference row (w encoded as 0)
    for (std::ptrdiff_t n : {4, 8, 11, 16, 64, 1024})
        run_min_cell<T, Key, Comp, Proj>(tname, cname, n, 0.0, algos);
}

// ------------------------------------------------------------------ drivers
template <class T, class Key, class Comp, class Proj>
void plain_type(const char* tname, const char* cname) {
    const std::ptrdiff_t small_ns[] = {16, 24, 32, 48, 64, 96, 128,
                                       192, 256, 384, 512, 768, 1024};
    const std::ptrdiff_t big_ns[] = {4096, 16384, 65536, 1 << 18, 1 << 20,
                                     1 << 22};
    std::vector<int> algos = {A_FLAT, A_DISPATCH, A_GAP, A_SEGBR, A_SEGFILL};
    std::vector<int> algos_h = algos;
    algos_h.push_back(A_HOARE);

    // slice 1: wrap sweep at p = 0.5 (hoare included up to 65536)
    for (double w : {1.0 / 64, 0.125, 0.5, 0.875, 63.0 / 64}) {
        for (auto n : small_ns)
            run_cell<T, Key, Comp, Proj>(tname, cname, "plain", n, w, 0, 0.5,
                                         algos_h);
        for (auto n : big_ns)
            run_cell<T, Key, Comp, Proj>(tname, cname, "plain", n, w, 0, 0.5,
                                         n <= 65536 ? algos_h : algos);
    }
    // slice 2: p sweep at wrap = 1/2
    for (double p : {0.05, 0.95}) {
        for (auto n : {std::ptrdiff_t(64), std::ptrdiff_t(1024),
                       std::ptrdiff_t(65536), std::ptrdiff_t(1 << 20)})
            run_cell<T, Key, Comp, Proj>(tname, cname, "plain", n, 0.5, 0, p,
                                         algos);
    }
}

template <class T, class Key, class Comp, class Proj>
void offset_type(const char* tname, const char* cname) {
    const std::ptrdiff_t ns[] = {64, 256, 1024, 4096, 65536, 1 << 20};
    std::vector<int> algos = {O_FLAT, O_DISPATCH, O_GAP, O_WFILL, O_WSEG};
    // wrap x offset grid at p = 0.5
    for (double w : {0.125, 0.5, 0.875})
        for (double of : {0.0, 0.125, 0.5, 0.875})
            for (auto n : ns)
                run_cell<T, Key, Comp, Proj>(tname, cname, "offset", n, w, of,
                                             0.5, algos);
    // p sweep at wrap = 1/2, offset = 1/2
    for (double p : {0.05, 0.95})
        for (auto n : {std::ptrdiff_t(1024), std::ptrdiff_t(65536)})
            run_cell<T, Key, Comp, Proj>(tname, cname, "offset", n, 0.5, 0.5,
                                         p, algos);
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "plain";
    const std::string only = argc > 2 ? argv[2] : "";  // type filter
    auto want = [&](const char* t) { return only.empty() || only == t; };
    std::printf("mode,type,comp,n,wfrac,offfrac,p,algo,ns_per_elem\n");
    if (mode == "plain") {
        if (want("i64")) plain_type<i64, i64, std::less<>, std::identity>("i64", "lex");
        if (want("i32")) plain_type<i32, i32, std::less<>, std::identity>("i32", "lex");
        if (want("p32")) plain_type<p32, p32, std::less<>, std::identity>("p32", "lex");
        if (want("p64")) plain_type<p64, p64, std::less<>, std::identity>("p64", "lex");
        if (want("p32f")) plain_type<p32, i32, std::less<>, first_proj>("p32", "first");
        if (want("p64f")) plain_type<p64, i64, std::less<>, first_proj>("p64", "first");
        if (want("pf32")) plain_type<pf32, pf32, std::less<>, std::identity>("pf32", "lex");
        if (want("pd64")) plain_type<pd64, pd64, std::less<>, std::identity>("pd64", "lex");
        if (want("pf32f")) plain_type<pf32, float, std::less<>, first_proj>("pf32", "first");
        if (want("pd64f")) plain_type<pd64, double, std::less<>, first_proj>("pd64", "first");
        // explicit branchless-lex comparator: the kernel columns then measure
        // what the dispatcher's substitution runs; `flat` = the flat-port
        // estimate (the flat entry points do NOT substitute).
        if (want("p32bl")) plain_type<p32, p32, LL, std::identity>("p32", "lexbl");
        if (want("p64bl")) plain_type<p64, p64, LL, std::identity>("p64", "lexbl");
        if (want("pf32bl")) plain_type<pf32, pf32, LL, std::identity>("pf32", "lexbl");
        if (want("pd64bl")) plain_type<pd64, pd64, LL, std::identity>("pd64", "lexbl");
        // pinned scan comparators, explicit (direct-kernel tier analysis)
        if (want("p64sb")) plain_type<p64, p64, LS, std::identity>("p64", "lexsb");
        if (want("pd64sb")) plain_type<pd64, pd64, LS, std::identity>("pd64", "lexsb");
        if (want("pf32sb")) plain_type<pf32, pf32, LS, std::identity>("pf32", "lexsb");
        if (want("p32pk")) plain_type<p32, p32, LP, std::identity>("p32", "lexpk");
    } else if (mode == "offset") {
        if (want("i64")) offset_type<i64, i64, std::less<>, std::identity>("i64", "lex");
        if (want("i32")) offset_type<i32, i32, std::less<>, std::identity>("i32", "lex");
        if (want("p64")) offset_type<p64, p64, std::less<>, std::identity>("p64", "lex");
        if (want("p64f")) offset_type<p64, i64, std::less<>, first_proj>("p64", "first");
        if (want("pf32")) offset_type<pf32, pf32, std::less<>, std::identity>("pf32", "lex");
        if (want("pd64")) offset_type<pd64, pd64, std::less<>, std::identity>("pd64", "lex");
        if (want("pd64f")) offset_type<pd64, double, std::less<>, first_proj>("pd64", "first");
        if (want("p64bl")) offset_type<p64, p64, LL, std::identity>("p64", "lexbl");
        if (want("pd64bl")) offset_type<pd64, pd64, LL, std::identity>("pd64", "lexbl");
        if (want("p64sb")) offset_type<p64, p64, LS, std::identity>("p64", "lexsb");
        if (want("pd64sb")) offset_type<pd64, pd64, LS, std::identity>("pd64", "lexsb");
    } else if (mode == "min") {
        if (want("i32")) min_type<i32, i32, std::less<>, std::identity>("i32", "lex", false);
        if (want("i64")) min_type<i64, i64, std::less<>, std::identity>("i64", "lex", true);
        if (want("p32")) min_type<p32, p32, std::less<>, std::identity>("p32", "lex", false);
        if (want("p64")) min_type<p64, p64, std::less<>, std::identity>("p64", "lex", false);
        if (want("pf32")) min_type<pf32, pf32, std::less<>, std::identity>("pf32", "lex", false);
        if (want("pd64")) min_type<pd64, pd64, std::less<>, std::identity>("pd64", "lex", false);
        if (want("p64f")) min_type<p64, i64, std::less<>, first_proj>("p64", "first", false);
        if (want("pd64f")) min_type<pd64, double, std::less<>, first_proj>("pd64", "first", false);
    }
    std::fprintf(stderr, "sink=%lld\n", (long long)g_sink);
    return 0;
}
