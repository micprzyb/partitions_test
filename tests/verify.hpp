#ifndef PARTITIONS_TEST_VERIFY_HPP
#define PARTITIONS_TEST_VERIFY_HPP

// Postcondition checkers shared by the correctness tests.

#include <algorithm>
#include <functional>
#include <iterator>
#include <vector>

namespace ptv {

// Two ranges hold the same multiset of values (partitioning must never lose,
// duplicate or invent an element).
template <class T>
bool same_multiset(std::vector<T> a, std::vector<T> b) {
    std::sort(a.begin(), a.end(), [](const T& x, const T& y) {
        return std::less<>{}(x, y);
    });
    std::sort(b.begin(), b.end(), [](const T& x, const T& y) {
        return std::less<>{}(x, y);
    });
    return a == b;
}

// Forward partition: [begin, p) strictly < pivot, [p, end) >= pivot.
template <class It, class Key, class Comp, class Proj>
bool forward_ok(It begin, It p, It end, const Key& pivot, Comp comp, Proj proj) {
    for (It it = begin; it != p; ++it)
        if (!comp(std::invoke(proj, *it), pivot)) return false;  // not < pivot
    for (It it = p; it != end; ++it)
        if (comp(std::invoke(proj, *it), pivot)) return false;  // < pivot on right
    return true;
}

// Reverse partition: [begin, p) >= pivot, [p, end) strictly < pivot.
template <class It, class Key, class Comp, class Proj>
bool reverse_ok(It begin, It p, It end, const Key& pivot, Comp comp, Proj proj) {
    for (It it = begin; it != p; ++it)
        if (comp(std::invoke(proj, *it), pivot)) return false;  // < pivot on left
    for (It it = p; it != end; ++it)
        if (!comp(std::invoke(proj, *it), pivot)) return false;  // not < pivot on right
    return true;
}

}  // namespace ptv

#endif  // PARTITIONS_TEST_VERIFY_HPP
