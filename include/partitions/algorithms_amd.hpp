#ifndef PARTITIONS_ALGORITHMS_AMD_HPP
#define PARTITIONS_ALGORITHMS_AMD_HPP

// AMD Zen 3 (znver3) tuned block partitions -- NEW functions, kept separate
// from algorithms.hpp on purpose.  The shipped `algo::boost_block` is tuned for
// Intel Meteor Lake (see docs/zen3_benchmarks.md): on Zen 3 its scalar
// offset-fill loop (`xor; cmp mem; setge; add num; lea; mov-byte->offsets[num]`)
// runs ~20% slower per element than on Meteor Lake's wider front-end, so
// `fulcrum` (single-pass double-write) overtakes it for i64 across n=256..2^20.
//
// These variants attack exactly that loop WITHOUT touching the Intel-tuned
// originals (which must stay as-is so they remain optimal there):
//
//   * block_scalar_amd<BS> : the pdqsort branchless block partition, identical
//     algorithm to algo::boost_block but with the offset-block size `BS` exposed
//     as a template parameter, so the Zen-3-optimal block size can be measured
//     and pinned (the original's 128 was tuned on Meteor Lake).
//
//   * block_simd_amd : the same block partition, but the two full-block
//     offset-fill loops are VECTORISED with AVX2.  For a contiguous block of an
//     8-byte signed-integer key compared with the default `<` (the i64 case --
//     where the Zen 3 gap lives), it computes 4 predicates per `vpcmpgtq`,
//     extracts a 4-bit mask with `vmovmskpd`, and appends the packed indices via
//     a 16-entry shuffle LUT and one unaligned 4-byte store (vs the scalar
//     loop's per-element setcc + byte store).  Every element that does not match
//     this fast-path shape -- wider/lexicographic keys, a non-identity
//     projection, the reverse partition's negated comparator, a non-contiguous
//     iterator -- transparently falls back to the SAME scalar fill the original
//     uses, so correctness is unchanged across the whole type/form matrix.
//
// The swap phase and all boundary bookkeeping are byte-for-byte the original's
// (we reuse detail::swap_offsets / detail::align_cacheline), so any speedup is
// purely the vectorised fill.  Raw partition throughput is the only criterion.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <type_traits>

#include "algorithms.hpp"  // detail::swap_offsets, align_cacheline, hoare, boost_block fallback

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace partitions::algo {
namespace detail_amd {

// ---- 16-entry pack LUTs for 4-lane (i64) AVX2 compaction --------------------
// For a 4-bit lane mask, `pos[m]` packs the set lanes' local indices (0..3) in
// INCREASING order (matching the scalar left fill's scan order); `roff[m]` packs
// the right-fill offsets (4 - lane) in INCREASING offset-value order (lane 3->1,
// lane 0->4), matching the scalar right fill's scan order.  Order matters: the
// leftover-drain phase (`base - offsets_r[num_r]`, processed top-down) is only
// collision-free when offsets are monotonic, exactly as pdqsort produces them
// scalar-side.  The unused high bytes of each 4-byte store are overwritten by
// the next group, exactly like the scalar loop's unconditional byte write.
struct Lut4 {
    std::uint32_t pos[16];
    std::uint32_t roff[16];
    std::uint8_t cnt[16];
};
inline constexpr Lut4 make_lut4() {
    Lut4 t{};
    for (int m = 0; m < 16; ++m) {
        std::uint8_t bp[4] = {0, 0, 0, 0};
        std::uint8_t br[4] = {0, 0, 0, 0};
        int k = 0;
        for (int b = 0; b < 4; ++b)
            if (m & (1 << b)) bp[k++] = static_cast<std::uint8_t>(b);
        int j = 0;
        for (int b = 3; b >= 0; --b)  // descending lane -> ascending offset 1..4
            if (m & (1 << b)) br[j++] = static_cast<std::uint8_t>(4 - b);
        t.cnt[m] = static_cast<std::uint8_t>(k);
        t.pos[m] = std::uint32_t(bp[0]) | (std::uint32_t(bp[1]) << 8) |
                   (std::uint32_t(bp[2]) << 16) | (std::uint32_t(bp[3]) << 24);
        t.roff[m] = std::uint32_t(br[0]) | (std::uint32_t(br[1]) << 8) |
                    (std::uint32_t(br[2]) << 16) | (std::uint32_t(br[3]) << 24);
    }
    return t;
}
inline constexpr Lut4 kLut4 = make_lut4();

// Is the (type, comparator, projection) the i64-shaped fast path?  Then
// below(x) == (x < pivot) on a contiguous block of an 8-byte signed integer.
template <class T, class Comp, class Proj>
inline constexpr bool simd_eligible =
    std::is_same_v<Proj, std::identity> &&
    (std::is_same_v<Comp, std::less<>> || std::is_same_v<Comp, std::less<T>>) &&
    std::is_integral_v<T> && std::is_signed_v<T> && sizeof(T) == 8;

#if defined(__AVX2__)
// Append, to offsets_l[num_l..], the local indices in [0,BS) of elements that
// are NOT below the pivot (>= pivot -> belong right).  base points at the block.
template <class T>
[[gnu::always_inline]] inline void vfill_left(const T* base, T pivot,
                                              unsigned char* offl,
                                              std::size_t& num_l,
                                              std::size_t BS) {
    const __m256i vp = _mm256_set1_epi64x(static_cast<long long>(pivot));
    for (std::size_t i = 0; i < BS; i += 4) {
        __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base + i));
        // below = x < pivot  <=>  pivot > x  <=>  cmpgt(vp, x)
        __m256i lt = _mm256_cmpgt_epi64(vp, x);
        unsigned bm =
            static_cast<unsigned>(_mm256_movemask_pd(_mm256_castsi256_pd(lt)));
        unsigned ge = (~bm) & 0xF;  // !below -> goes right
        std::uint32_t packed =
            kLut4.pos[ge] + static_cast<std::uint32_t>(i) * 0x01010101u;
        std::memcpy(offl + num_l, &packed, 4);  // unconditional 4-byte store
        num_l += kLut4.cnt[ge];                  // advance by popcount only
    }
}

// Append, to offsets_r[num_r..], the 1-based offsets-from-`rlast` (1..BS) of the
// elements that ARE below the pivot (belong left), scanning the block
// descending -- mirrors the scalar `offsets_r[num_r]=++i; num_r += below(--last)`.
template <class T>
[[gnu::always_inline]] inline void vfill_right(const T* rlast, T pivot,
                                               unsigned char* offr,
                                               std::size_t& num_r,
                                               std::size_t BS) {
    const __m256i vp = _mm256_set1_epi64x(static_cast<long long>(pivot));
    for (std::size_t g = 0; g < BS; g += 4) {
        // group examines [rlast-g-4 .. rlast-g-1]; lane b at rlast-g-4+b has
        // offset-from-rlast = g + (4 - b)  -> roff lane value (4-b), base g.
        const T* p = rlast - g - 4;
        __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        __m256i lt = _mm256_cmpgt_epi64(vp, x);  // below = x < pivot
        unsigned bm =
            static_cast<unsigned>(_mm256_movemask_pd(_mm256_castsi256_pd(lt)));
        std::uint32_t packed =
            kLut4.roff[bm] + static_cast<std::uint32_t>(g) * 0x01010101u;
        std::memcpy(offr + num_r, &packed, 4);
        num_r += kLut4.cnt[bm];
    }
}
#endif  // __AVX2__

// Mirror of detail::branchless_partition, with the two FULL-block fills
// optionally vectorised.  `UseSimd` is the compile-time opt-in; the actual SIMD
// path is taken only when the type/comp/proj/iterator also qualify, else the
// identical scalar fill runs.  BS is the offset-block size (multiple of 8).
template <bool Guarded, bool UseSimd, std::size_t BS, class Iter, class K,
          class Compare, class Proj>
inline Iter branchless_partition_amd(Iter begin, Iter end, K pivot, Compare comp,
                                     Proj proj) {
    using T = std::iter_value_t<Iter>;
    auto below = [&](Iter it) {
        return static_cast<bool>(
            std::invoke(comp, std::invoke(proj, *it), pivot));
    };
    constexpr bool kSimd =
#if defined(__AVX2__)
        UseSimd && std::contiguous_iterator<Iter> && simd_eligible<T, Compare, Proj>;
#else
        false;
#endif

    Iter first = begin, last = end;
    if constexpr (Guarded) {
        while (first < last && below(first)) ++first;
    } else {
        while (below(first)) ++first;
    }
    if (first == begin) {
        while (begin < last && !below(--last)) {
        }
    } else {
        while (!below(--last)) {
        }
    }

    if (first < last) {
        std::iter_swap(first, last);
        ++first;

        alignas(detail::boost_cacheline_size)
            unsigned char off_l_store[BS + 2 * detail::boost_cacheline_size];
        alignas(detail::boost_cacheline_size)
            unsigned char off_r_store[BS + 2 * detail::boost_cacheline_size];
        unsigned char* offsets_l = detail::align_cacheline(off_l_store);
        unsigned char* offsets_r = detail::align_cacheline(off_r_store);

        Iter offsets_l_base = first;
        Iter offsets_r_base = last;
        std::size_t num_l = 0, num_r = 0, start_l = 0, start_r = 0;

        while (first < last) {
            std::size_t num_unknown = static_cast<std::size_t>(last - first);
            std::size_t left_split =
                num_l == 0 ? (num_r == 0 ? num_unknown / 2 : num_unknown) : 0;
            std::size_t right_split = num_r == 0 ? (num_unknown - left_split) : 0;

            if (left_split >= BS) {
                if constexpr (kSimd) {
                    vfill_left<T>(&*first, pivot, offsets_l, num_l, BS);
                    first += BS;
                } else {
                    for (std::size_t i = 0; i < BS;) {
                        offsets_l[num_l] = (unsigned char)i++; num_l += !below(first); ++first;
                        offsets_l[num_l] = (unsigned char)i++; num_l += !below(first); ++first;
                        offsets_l[num_l] = (unsigned char)i++; num_l += !below(first); ++first;
                        offsets_l[num_l] = (unsigned char)i++; num_l += !below(first); ++first;
                        offsets_l[num_l] = (unsigned char)i++; num_l += !below(first); ++first;
                        offsets_l[num_l] = (unsigned char)i++; num_l += !below(first); ++first;
                        offsets_l[num_l] = (unsigned char)i++; num_l += !below(first); ++first;
                        offsets_l[num_l] = (unsigned char)i++; num_l += !below(first); ++first;
                    }
                }
            } else {
                for (std::size_t i = 0; i < left_split;) {
                    offsets_l[num_l] = (unsigned char)i++; num_l += !below(first); ++first;
                }
            }

            if (right_split >= BS) {
                if constexpr (kSimd) {
                    vfill_right<T>(&*last, pivot, offsets_r, num_r, BS);
                    last -= BS;
                } else {
                    for (std::size_t i = 0; i < BS;) {
                        offsets_r[num_r] = (unsigned char)++i; num_r += below(--last);
                        offsets_r[num_r] = (unsigned char)++i; num_r += below(--last);
                        offsets_r[num_r] = (unsigned char)++i; num_r += below(--last);
                        offsets_r[num_r] = (unsigned char)++i; num_r += below(--last);
                        offsets_r[num_r] = (unsigned char)++i; num_r += below(--last);
                        offsets_r[num_r] = (unsigned char)++i; num_r += below(--last);
                        offsets_r[num_r] = (unsigned char)++i; num_r += below(--last);
                        offsets_r[num_r] = (unsigned char)++i; num_r += below(--last);
                    }
                }
            } else {
                for (std::size_t i = 0; i < right_split;) {
                    offsets_r[num_r] = (unsigned char)++i; num_r += below(--last);
                }
            }

            std::size_t num = std::min(num_l, num_r);
            detail::swap_offsets(offsets_l_base, offsets_r_base, offsets_l + start_l,
                                 offsets_r + start_r, num, num_l == num_r);
            num_l -= num;
            num_r -= num;
            start_l += num;
            start_r += num;
            if (num_l == 0) { start_l = 0; offsets_l_base = first; }
            if (num_r == 0) { start_r = 0; offsets_r_base = last; }
        }

        if (num_l) {
            offsets_l += start_l;
            while (num_l--)
                std::iter_swap(offsets_l_base + offsets_l[num_l], --last);
            first = last;
        }
        if (num_r) {
            offsets_r += start_r;
            while (num_r--) {
                std::iter_swap(offsets_r_base - offsets_r[num_r], first);
                ++first;
            }
            last = first;
        }
    }
    return first;
}

}  // namespace detail_amd

// Block partition, block size `BS` exposed (Zen-3 retune knob).  Scalar fill.
template <std::size_t BS = 128>
struct block_scalar_amd {
    static constexpr const char* name = "block_scalar_amd";
    static_assert(BS % 8 == 0 && BS < 256);

    template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
              class Comp, class Proj>
    I operator()(I first, S last, K pivot, Comp comp, Proj proj) const {
        I end = first + (last - first);
        if (end - first < detail::boost_scalar_cutoff)
            return hoare{}(first, end, pivot, comp, proj);
        return detail_amd::branchless_partition_amd<true, false, BS>(
            first, end, pivot, comp, proj);
    }

    template <std::random_access_iterator I, std::sentinel_for<I> S,
              class Comp = std::less<>, class Proj = std::identity>
    I at(I first, S last_s, I pivot, Comp comp = {}, Proj proj = {}) const {
        I last = first + (last_s - first);
        auto key = std::invoke(proj, *pivot);
        if (last - first < detail::boost_scalar_cutoff)
            return hoare{}(first, last, key, comp, proj);
        return detail_amd::branchless_partition_amd<false, false, BS>(
            first, last, key, comp, proj);
    }
};

// Block partition with AVX2-vectorised offset fill for the i64 fast path.
struct block_simd_amd {
    static constexpr const char* name = "block_simd_amd";
    static constexpr std::size_t BS = 128;

    template <std::random_access_iterator I, std::sentinel_for<I> S, class K,
              class Comp, class Proj>
    I operator()(I first, S last, K pivot, Comp comp, Proj proj) const {
        I end = first + (last - first);
        if (end - first < detail::boost_scalar_cutoff)
            return hoare{}(first, end, pivot, comp, proj);
        return detail_amd::branchless_partition_amd<true, true, BS>(
            first, end, pivot, comp, proj);
    }

    template <std::random_access_iterator I, std::sentinel_for<I> S,
              class Comp = std::less<>, class Proj = std::identity>
    I at(I first, S last_s, I pivot, Comp comp = {}, Proj proj = {}) const {
        I last = first + (last_s - first);
        auto key = std::invoke(proj, *pivot);
        if (last - first < detail::boost_scalar_cutoff)
            return hoare{}(first, last, key, comp, proj);
        return detail_amd::branchless_partition_amd<false, true, BS>(
            first, last, key, comp, proj);
    }
};

}  // namespace partitions::algo

#endif  // PARTITIONS_ALGORITHMS_AMD_HPP
