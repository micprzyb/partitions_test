#ifndef PARTITIONS_QUICK_PARTITION_HPP
#define PARTITIONS_QUICK_PARTITION_HPP

// `quick_partition` -- one entry point that dispatches to the fastest partition
// for the (array size, element size, key/comparator cost, ISA) at hand.  It is a
// model of `PivotPartitioner` (concepts.hpp), so it plugs into the whole API and
// the registry like any other partitioner.
//
// The decision tree (raw throughput is the only criterion; thresholds tuned on
// Zen 3 / AVX2, see docs/zen3_quick_partition.md):
//
//   cheap-compare narrow integer key (i32/i64, identity proj, `<`), contiguous,
//   AVX2 available:
//       n <  64                  -> lomuto_branchless  (no setup; ~0.6 ns/elem)
//       n <= L2_bytes/sizeof(T)  -> block_compress_amd (single-pass AVX2
//                                   compaction; fewest instructions, but 2x store
//                                   traffic -> wins only while L2-resident)
//       else                     -> block_simd_amd     (AVX2 fill + few-write
//                                   swap; bandwidth-friendly once L2 spills)
//   everything else (wide/lexicographic/keyed/projected, or no AVX2):
//       -> algo::sized       (branchless Lomuto small / pdqsort block large --
//                             the portable Intel-tuned dispatcher, best for
//                             expensive-compare and wide elements)
//
// WHY THESE AXES
//   * element size  : the AVX2 compaction/fill paths are gated to 4- and 8-byte
//                     integer keys (8 and 4 lanes per vector); wider elements get
//                     too little SIMD parallelism and go to the block partition.
//   * key/comparator: a *cheap* fixed-pivot compare (integer `<`) is what lets
//                     the vectorised compare + permute win; an expensive
//                     comparator (pair64 lex) wants the block partition's 8x
//                     compare-ILP instead, so it routes to `sized`.  When the key
//                     is a small slice of a wider element (e.g. pair64f: 8-byte
//                     key in a 16-byte element) the cheap compare still favours
//                     the block partition's ILP over a 2-element-per-vector
//                     compaction -- `sized`/boost handles it well already.
//   * array size    : the cache-residency crossover above.
//   * ISA           : AVX2 is common to current Intel (Meteor Lake) and AMD
//                     (Zen 3); the SIMD paths help both.  FUTURE ARCHS plug in at
//                     the marked extension point below (e.g. AVX-512 vpcompressd
//                     would replace the permute-LUT compaction; NEON/SVE a
//                     different lane width) -- only the capability constant and
//                     the eligibility/threshold table change, not the dispatch.

#include <functional>
#include <iterator>
#include <type_traits>

#include "algorithms.hpp"
#include "algorithms_amd.hpp"

namespace partitions {
namespace detail_qp {

// ---- ISA capability (compile-time). Extend for future architectures. -------
inline constexpr bool kHasAvx2 =
#if defined(__AVX2__)
    true;
#else
    false;
#endif
// Future: kHasAvx512 (vpcompressd compaction), kHasNeon/kHasSve, ... -- add the
// flag here and a branch in the eligible block below; the rest is unchanged.

// Crossover thresholds for the cheap-compare narrow-int path.
//
// The compaction partition double-stores every element (its vectorised "swap"),
// so it pays ~2x the store traffic; that is free while the block is L2-resident
// and the dominant store-port-bound cost is the fewest-instruction kernel, but
// once the block spills L2 the extra writes hit the lower L3/DRAM bandwidth and
// the few-write fill+swap (block_simd_amd) wins.  So the crossover is exactly
// "does the block fit L2?"  -> kCompressMax = L2_bytes / sizeof(T).  This is
// both the measured Zen 3 crossover (i64 wins through 2^16 = 512 KiB = Zen 3 L2,
// cliffs at 2^18) AND a portable, future-proof rule: set kL2Bytes per target.
inline constexpr std::size_t kL2Bytes = 512 * 1024;  // Zen 3 per-core L2.
// Meteor Lake P-core L2 = 2 MiB -> compaction would win a 4x larger range there;
// a future build can pick kL2Bytes from the target (e.g. __cpuid / a -D flag).
template <class T>
inline constexpr std::ptrdiff_t kLomutoMax = 64;  // below: branchless Lomuto
template <class T>
inline constexpr std::ptrdiff_t kCompressMax =
    static_cast<std::ptrdiff_t>(kL2Bytes / sizeof(T));  // <=: compaction; else fill+swap

// The fast-path predicate: a contiguous block of a 4/8-byte signed integer key
// compared with the default `<` and an identity projection, on an AVX2 machine.
template <class I, class T, class Comp, class Proj>
inline constexpr bool kFastPath =
    kHasAvx2 && std::contiguous_iterator<I> &&
    algo::detail_amd::simd_eligible<T, Comp, Proj>;

}  // namespace detail_qp

struct quick_partition {
    static constexpr const char* name = "quick_partition";

    template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
              class Comp, class Proj>
    I operator()(I first, S last, K pivot, Comp comp, Proj proj) const {
        using T = std::iter_value_t<I>;
        I end = first + (last - first);
        const auto n = end - first;
        if constexpr (detail_qp::kFastPath<I, T, Comp, Proj>) {
            if (n < detail_qp::kLomutoMax<T>)
                return algo::lomuto_branchless{}(first, end, pivot, comp, proj);
            if (n <= detail_qp::kCompressMax<T>)
                return algo::block_compress_amd{}(first, end, pivot, comp, proj);
            return algo::block_simd_amd{}(first, end, pivot, comp, proj);
        } else {
            return algo::sized{}(first, end, pivot, comp, proj);
        }
    }

    // Position form.  For the wide/expensive path keep `sized`'s sentinel fast
    // path; for the SIMD path the partitioners are value-based, so read the key
    // and route by size (raw-throughput-equal to the value form on random data).
    template <std::random_access_iterator I, std::sentinel_for<I> S,
              class Comp = std::less<>, class Proj = std::identity>
    I at(I first, S last_s, I pivot, Comp comp = {}, Proj proj = {}) const {
        using T = std::iter_value_t<I>;
        I last = first + (last_s - first);
        if constexpr (detail_qp::kFastPath<I, T, Comp, Proj>) {
            auto key = std::invoke(proj, *pivot);
            const auto n = last - first;
            if (n < detail_qp::kLomutoMax<T>)
                return algo::lomuto_branchless{}(first, last, key, comp, proj);
            if (n <= detail_qp::kCompressMax<T>)
                return algo::block_compress_amd{}(first, last, key, comp, proj);
            return algo::block_simd_amd{}.at(first, last, pivot, comp, proj);
        } else {
            return algo::sized{}.at(first, last, pivot, comp, proj);
        }
    }
};

}  // namespace partitions

#endif  // PARTITIONS_QUICK_PARTITION_HPP
