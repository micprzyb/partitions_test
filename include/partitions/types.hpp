#ifndef PARTITIONS_TYPES_HPP
#define PARTITIONS_TYPES_HPP

// Element types used throughout the test and benchmark suites.
//
// The interface under test partitions a block of arbitrary elements given a
// comparison and a projection.  These types exercise three relevant regimes:
//
//   * `i32`      - small, cheap-to-move scalar (register sized).
//   * `i64`      - wider scalar.
//   * `pair64`   - 16-byte aggregate compared lexicographically; the move is
//                  twice as wide and the comparison is no longer a single
//                  machine instruction, which stresses branch-heavy schemes.
//   * `keyed`    - a record carrying a payload, used to exercise a non-trivial
//                  projection (we sort by `.key` but carry `.payload`).

#include <cstdint>
#include <functional>
#include <string>
#include <utility>

namespace partitions {

using i32 = std::int32_t;
using i64 = std::int64_t;

// Lexicographically ordered pair of 64-bit integers.  std::pair already
// provides the lexicographic operator<, but we wrap it so the type has a
// short, stable name in benchmark output and a known size.
struct pair64 {
    i64 first{};
    i64 second{};

    friend constexpr bool operator==(const pair64&, const pair64&) = default;
    friend constexpr auto operator<=>(const pair64&, const pair64&) = default;
};

static_assert(sizeof(pair64) == 16);

// A record with a separate sort key and payload, used to test that a
// non-identity projection is threaded correctly through every code path.
struct keyed {
    i64 key{};
    i64 payload{};

    // Full lexicographic (key, payload) order -- used only by the tests'
    // multiset check; the partition logic itself orders by the `key_of`
    // projection below.
    friend constexpr bool operator==(const keyed&, const keyed&) = default;
    friend constexpr auto operator<=>(const keyed&, const keyed&) = default;
};

// Projection that selects the sort key of a `keyed` record.
struct key_of {
    constexpr i64 operator()(const keyed& k) const noexcept { return k.key; }
};

// Human-readable type names for reporting.
template <class T>
constexpr const char* type_name() {
    if constexpr (std::is_same_v<T, i32>) return "i32";
    else if constexpr (std::is_same_v<T, i64>) return "i64";
    else if constexpr (std::is_same_v<T, pair64>) return "pair64";
    else if constexpr (std::is_same_v<T, keyed>) return "keyed";
    else return "unknown";
}

// Build a value of element type `T` from an integer rank.  Used by the
// distribution generators so they can be written once over integer "ranks"
// and materialised into any element type.
template <class T>
constexpr T make_value(i64 rank) {
    if constexpr (std::is_same_v<T, pair64>) {
        // Spread the rank across both fields so ordering still tracks `rank`
        // but the second field genuinely participates in comparisons.
        return pair64{rank >> 8, rank & 0xff};
    } else if constexpr (std::is_same_v<T, keyed>) {
        return keyed{rank, ~rank};
    } else {
        return static_cast<T>(rank);
    }
}

// The projection appropriate for element type `T`.
template <class T>
constexpr auto projection_for() {
    if constexpr (std::is_same_v<T, keyed>) return key_of{};
    else return std::identity{};
}

}  // namespace partitions

#endif  // PARTITIONS_TYPES_HPP
