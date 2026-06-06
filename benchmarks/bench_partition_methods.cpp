// Focused study: three questions about HOW to drive the branchless block
// partition, judged purely on raw throughput (random_uniform, batched blocks,
// same harness as bench_partition; min ns/element reported).
//
//   Q1  sort-as-partition.  For small n, is it faster to fully SORT the block
//       with a branchless small-sort network (then take lower_bound of the
//       pivot) than to partition it?  -> only for a cheap-to-swap element
//       (i64) at n <= ~16; never for pair64 / pair64-by-first, whose
//       O(n log^2 n) 16-byte compare-exchanges dwarf a partition's few swaps.
//
//   Q2  value pivot vs position sentinel.  Does passing the pivot BY VALUE
//       (`const K&` + comp/proj, the library's primitive) cost anything versus
//       the original pdqsort scheme that swaps the pivot to an end and uses the
//       ELEMENT as a physical sentinel?  -> no: the unguarded scan keeps the
//       sentinel benefit; the two are within a few %, and `pos` does two extra
//       swaps for no consistent win.
//
//   Q3  predicate vs comp/proj.  Is the old `Pred keep_left` interface as fast
//       as threading comp/proj/pivot directly?  -> equal at small n, and the
//       comp/proj form is FASTER at large n (the opaque predicate inhibits
//       codegen).  The interface change is not a regression.
//
// Methods (all return the partition boundary m: [first,m) < pivot):
//   hoare  scalar two-pointer baseline
//   value  branchless block, local key copy + comp/proj      (current design)
//   pred   branchless block driven by an opaque keep_left     (old interface)
//   pos    branchless block, pivot swapped to an end as a physical sentinel
//          and placed at the boundary                         (faithful pdqsort)
//   sort   small_sort::sort + lower_bound                     (n <= 64 only)
//
// Usage: bench_partition_methods            full sweep, CSV to stdout
// CSV columns: type,method,n,ns_per_elem

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <vector>

#include "bench_harness.hpp"
#include "partitions/partitions.hpp"
#include "partitions/small_sort.hpp"

using namespace partitions;
namespace dbl = partitions::algo::detail;

namespace {

// Generic branchless block core parameterised by an opaque predicate `below`
// (a copy of detail::branchless_partition<false> so the value/predicate variants
// share identical structure and only the predicate differs).  Unguarded scan:
// requires an in-block element >= pivot to stop it (the median-of-3 pivot is).
template <class Iter, class Below>
Iter bl_core(Iter begin, Iter end, Below below) {
    Iter first = begin, last = end;
    while (below(first)) ++first;
    if (first == begin) {
        while (begin < last && !below(--last)) {}
    } else {
        while (!below(--last)) {}
    }
    if (first < last) {
        std::iter_swap(first, last);
        ++first;
        constexpr std::size_t BS = dbl::boost_block_size, CL = dbl::boost_cacheline_size;
        alignas(CL) unsigned char ls[BS + CL];
        alignas(CL) unsigned char rs[BS + CL];
        unsigned char* ol = dbl::align_cacheline(ls);
        unsigned char* orr = dbl::align_cacheline(rs);
        Iter olb = first, orb = last;
        std::size_t nl = 0, nr = 0, sl = 0, sr = 0;
        while (first < last) {
            std::size_t nu = last - first;
            std::size_t lsp = nl == 0 ? (nr == 0 ? nu / 2 : nu) : 0;
            std::size_t rsp = nr == 0 ? (nu - lsp) : 0;
            if (lsp >= BS) {
                for (std::size_t i = 0; i < BS;) {
                    ol[nl] = (unsigned char)i++; nl += !below(first); ++first;
                }
            } else {
                for (std::size_t i = 0; i < lsp;) {
                    ol[nl] = (unsigned char)i++; nl += !below(first); ++first;
                }
            }
            if (rsp >= BS) {
                for (std::size_t i = 0; i < BS;) {
                    orr[nr] = (unsigned char)++i; nr += below(--last);
                }
            } else {
                for (std::size_t i = 0; i < rsp;) {
                    orr[nr] = (unsigned char)++i; nr += below(--last);
                }
            }
            std::size_t num = std::min(nl, nr);
            dbl::swap_offsets(olb, orb, ol + sl, orr + sr, num, nl == nr);
            nl -= num; nr -= num; sl += num; sr += num;
            if (nl == 0) { sl = 0; olb = first; }
            if (nr == 0) { sr = 0; orb = last; }
        }
        if (nl) { ol += sl; while (nl--) std::iter_swap(olb + ol[nl], --last); first = last; }
        if (nr) { orr += sr; while (nr--) { std::iter_swap(orb - orr[nr], first); ++first; } last = first; }
    }
    return first;
}

template <class I, class Comp, class Proj>
I m_value(I f, I l, I piv, Comp comp, Proj proj) {
    auto key = std::invoke(proj, *piv);
    return bl_core(f, l, [&](I it) {
        return static_cast<bool>(std::invoke(comp, std::invoke(proj, *it), key));
    });
}

template <class I, class Comp, class Proj>
I m_pred(I f, I l, I piv, Comp comp, Proj proj) {
    auto key = std::invoke(proj, *piv);
    auto keep_left = [&](const auto& x) {
        return static_cast<bool>(std::invoke(comp, std::invoke(proj, x), key));
    };
    return bl_core(f, l, [&](I it) { return keep_left(*it); });
}

// Faithful pdqsort: pivot swapped to *f and used as a physical sentinel, then
// placed at the boundary.  (Compares against *f, the moved pivot, each step.)
template <class I, class Comp, class Proj>
I m_pos(I f, I l, I piv, Comp comp, Proj proj) {
    std::iter_swap(f, piv);
    auto below = [&](I it) {
        return static_cast<bool>(
            std::invoke(comp, std::invoke(proj, *it), std::invoke(proj, *f)));
    };
    I first = f + 1, last = l;
    while (below(first)) ++first;
    if (first == f + 1) {
        while (first < last && !below(--last)) {}
    } else {
        while (!below(--last)) {}
    }
    if (first < last) {
        std::iter_swap(first, last);
        ++first;
        constexpr std::size_t BS = dbl::boost_block_size, CL = dbl::boost_cacheline_size;
        alignas(CL) unsigned char ls[BS + CL];
        alignas(CL) unsigned char rs[BS + CL];
        unsigned char* ol = dbl::align_cacheline(ls);
        unsigned char* orr = dbl::align_cacheline(rs);
        I olb = first, orb = last;
        std::size_t nl = 0, nr = 0, sl = 0, sr = 0;
        while (first < last) {
            std::size_t nu = last - first;
            std::size_t lsp = nl == 0 ? (nr == 0 ? nu / 2 : nu) : 0;
            std::size_t rsp = nr == 0 ? (nu - lsp) : 0;
            if (lsp >= BS) {
                for (std::size_t i = 0; i < BS;) {
                    ol[nl] = (unsigned char)i++; nl += !below(first); ++first;
                }
            } else {
                for (std::size_t i = 0; i < lsp;) {
                    ol[nl] = (unsigned char)i++; nl += !below(first); ++first;
                }
            }
            if (rsp >= BS) {
                for (std::size_t i = 0; i < BS;) {
                    orr[nr] = (unsigned char)++i; nr += below(--last);
                }
            } else {
                for (std::size_t i = 0; i < rsp;) {
                    orr[nr] = (unsigned char)++i; nr += below(--last);
                }
            }
            std::size_t num = std::min(nl, nr);
            dbl::swap_offsets(olb, orb, ol + sl, orr + sr, num, nl == nr);
            nl -= num; nr -= num; sl += num; sr += num;
            if (nl == 0) { sl = 0; olb = first; }
            if (nr == 0) { sr = 0; orb = last; }
        }
        if (nl) { ol += sl; while (nl--) std::iter_swap(olb + ol[nl], --last); first = last; }
        if (nr) { orr += sr; while (nr--) { std::iter_swap(orb - orr[nr], first); ++first; } last = first; }
    }
    std::iter_swap(f, first - 1);  // place pivot at the boundary
    return first;
}

template <class I, class Comp, class Proj>
I m_sort(I f, I l, I piv, Comp comp, Proj proj) {
    auto key = std::invoke(proj, *piv);
    small_sort::sort(f, l, comp, proj);
    return std::lower_bound(f, l, key, [&](const auto& x, const auto& k) {
        return static_cast<bool>(std::invoke(comp, std::invoke(proj, x), k));
    });
}

template <class I, class Comp, class Proj>
I m_hoare(I f, I l, I piv, Comp comp, Proj proj) {
    auto key = std::invoke(proj, *piv);
    return algo::hoare{}(f, l, key, comp, proj);
}

constexpr std::size_t kSizes[] = {8,   12,  16,  21,   24,      32,      48, 64,
                                  128, 256, 1024, 4096, 1u << 16, 1u << 20, 1u << 22};
constexpr std::size_t kElemsPerIter = 1u << 16;

template <class T, class Proj, class Fn>
void run_one(const char* tn, const char* mn, Proj proj, Fn fn, std::size_t n,
             bool small_only) {
    if (small_only && n > 64) return;  // small_sort degrades to O(n^2) above 24
    auto comp = std::less<>{};
    const std::size_t batch = n < kElemsPerIter ? kElemsPerIter / n : 1;
    const std::size_t total = n * batch;
    std::vector<T> master;
    master.reserve(total);
    std::vector<std::size_t> pidx(batch);
    for (std::size_t b = 0; b < batch; ++b) {
        auto blk = dist::random_uniform{}.template operator()<T>(n, 0x1234u + b + n);
        auto it = pivot::median_of_3{}(blk.begin(), blk.end(), comp, proj);
        pidx[b] = static_cast<std::size_t>(it - blk.begin());
        master.insert(master.end(), blk.begin(), blk.end());
    }
    std::vector<T> work(total);
    std::uint64_t reps = static_cast<std::uint64_t>((1u << 22) / total);
    reps = std::min<std::uint64_t>(std::max<std::uint64_t>(reps, 5), 200);
    auto setup = [&] { work = master; };
    auto do_work = [&] {
        std::ptrdiff_t sink = 0;
        for (std::size_t b = 0; b < batch; ++b) {
            auto beg = work.begin() + static_cast<std::ptrdiff_t>(b * n);
            auto p = fn(beg, beg + static_cast<std::ptrdiff_t>(n),
                        beg + static_cast<std::ptrdiff_t>(pidx[b]), comp, proj);
            sink += p - beg;
        }
        bench::do_not_optimize(sink);
    };
    auto r = bench::measure(reps, setup, do_work);
    std::printf("%s,%s,%zu,%.4f\n", tn, mn, n,
                r.min_ns / static_cast<double>(total));
    std::fflush(stdout);
}

template <class T, class Proj>
void run_type(const char* tn, Proj proj) {
    for (std::size_t n : kSizes) {
        run_one<T>(tn, "hoare", proj, [](auto f, auto l, auto p, auto c, auto pr) { return m_hoare(f, l, p, c, pr); }, n, false);
        run_one<T>(tn, "value", proj, [](auto f, auto l, auto p, auto c, auto pr) { return m_value(f, l, p, c, pr); }, n, false);
        run_one<T>(tn, "pred",  proj, [](auto f, auto l, auto p, auto c, auto pr) { return m_pred(f, l, p, c, pr); }, n, false);
        run_one<T>(tn, "pos",   proj, [](auto f, auto l, auto p, auto c, auto pr) { return m_pos(f, l, p, c, pr); }, n, false);
        run_one<T>(tn, "sort",  proj, [](auto f, auto l, auto p, auto c, auto pr) { return m_sort(f, l, p, c, pr); }, n, true);
    }
}

struct first_key {
    template <class P>
    auto operator()(const P& p) const { return p.first; }
};

}  // namespace

int main() {
    std::printf("type,method,n,ns_per_elem\n");
    run_type<i64>("i64", std::identity{});
    run_type<pair64>("pair64", std::identity{});
    run_type<pair64>("pair64f", first_key{});
    return 0;
}
