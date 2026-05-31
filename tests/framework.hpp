#ifndef PARTITIONS_TEST_FRAMEWORK_HPP
#define PARTITIONS_TEST_FRAMEWORK_HPP

// A tiny, dependency-free test framework.
//
// Deliberately minimal: this project's tests are mostly templated loops over
// the algorithm / type / distribution matrix, which a macro-heavy framework
// makes awkward.  Usage:
//
//     TEST_CASE("name") { CHECK(cond); REQUIRE(cond); }
//
// CHECK records a failure and continues; REQUIRE aborts the current test.
// Link exactly one translation unit against framework_main.cpp (or define
// PT_TEST_MAIN) to get a runner.

#include <cstdio>
#include <exception>
#include <string>
#include <string_view>
#include <vector>

namespace pt_test {

struct Stats {
    long checks = 0;
    long failures = 0;
};
inline Stats& stats() {
    static Stats s;
    return s;
}

struct require_failed : std::exception {};

struct Case {
    std::string_view name;
    void (*fn)();
};

inline std::vector<Case>& registry() {
    static std::vector<Case> r;
    return r;
}

struct Registrar {
    Registrar(std::string_view name, void (*fn)()) { registry().push_back({name, fn}); }
};

inline void report_failure(const char* file, int line, const char* expr,
                           const std::string& msg) {
    ++stats().failures;
    std::printf("  [FAIL] %s:%d  %s%s%s\n", file, line, expr,
                msg.empty() ? "" : "  -- ", msg.c_str());
}

inline int run_all() {
    std::printf("Running %zu test cases\n", registry().size());
    int failed_cases = 0;
    for (const auto& c : registry()) {
        const long before = stats().failures;
        try {
            c.fn();
        } catch (const require_failed&) {
            // already reported
        } catch (const std::exception& e) {
            report_failure(__FILE__, __LINE__, "uncaught exception",
                           std::string("what(): ") + e.what());
        }
        const bool ok = stats().failures == before;
        std::printf("[%s] %.*s\n", ok ? " ok " : "FAIL",
                    static_cast<int>(c.name.size()), c.name.data());
        if (!ok) ++failed_cases;
    }
    std::printf("\n%ld checks, %ld failures across %d/%zu failing cases\n",
                stats().checks, stats().failures, failed_cases, registry().size());
    return failed_cases == 0 ? 0 : 1;
}

}  // namespace pt_test

#define PT_CONCAT_(a, b) a##b
#define PT_CONCAT(a, b) PT_CONCAT_(a, b)

#define TEST_CASE(NAME)                                                     \
    static void PT_CONCAT(pt_case_, __LINE__)();                            \
    static ::pt_test::Registrar PT_CONCAT(pt_reg_, __LINE__)(               \
        NAME, &PT_CONCAT(pt_case_, __LINE__));                             \
    static void PT_CONCAT(pt_case_, __LINE__)()

#define CHECK_MESSAGE(COND, MSG)                                            \
    do {                                                                    \
        ++::pt_test::stats().checks;                                        \
        if (!(COND))                                                        \
            ::pt_test::report_failure(__FILE__, __LINE__, #COND, (MSG));   \
    } while (0)

#define CHECK(COND) CHECK_MESSAGE(COND, std::string{})

#define REQUIRE(COND)                                                       \
    do {                                                                    \
        ++::pt_test::stats().checks;                                        \
        if (!(COND)) {                                                      \
            ::pt_test::report_failure(__FILE__, __LINE__, #COND,           \
                                      "REQUIRE failed");                    \
            throw ::pt_test::require_failed{};                              \
        }                                                                   \
    } while (0)

#endif  // PARTITIONS_TEST_FRAMEWORK_HPP
