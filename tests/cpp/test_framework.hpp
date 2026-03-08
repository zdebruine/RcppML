// Minimal C++ unit test framework — no external dependencies
// Provides assertion macros and a test runner.
// Can be replaced with Catch2 or doctest by swapping this header.

#ifndef RCPPML_TEST_FRAMEWORK_HPP
#define RCPPML_TEST_FRAMEWORK_HPP

#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <cmath>
#include <sstream>

namespace test {

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

struct TestState {
    int passes = 0;
    int failures = 0;
    std::vector<std::string> failure_messages;

    static TestState& instance() {
        static TestState s;
        return s;
    }

    static std::vector<TestCase>& tests() {
        static std::vector<TestCase> t;
        return t;
    }

    void pass() { passes++; }

    void fail(const std::string& msg, const char* file, int line) {
        failures++;
        std::ostringstream oss;
        oss << "  FAIL: " << file << ":" << line << " — " << msg;
        failure_messages.push_back(oss.str());
        std::cerr << failure_messages.back() << "\n";
    }

    int run_all() {
        int suite_failures = 0;
        for (auto& tc : tests()) {
            int before = failures;
            std::cout << "[TEST] " << tc.name << " ... ";
            try {
                tc.fn();
            } catch (const std::exception& e) {
                fail(std::string("exception: ") + e.what(), "?", 0);
            } catch (...) {
                fail("unknown exception", "?", 0);
            }
            if (failures == before) {
                std::cout << "OK\n";
            } else {
                std::cout << "FAILED\n";
                suite_failures++;
            }
        }
        std::cout << "\n=== " << passes << " passed, " << failures << " failed"
                  << " (" << tests().size() << " tests) ===\n";
        return failures > 0 ? 1 : 0;
    }
};

struct TestRegistrar {
    TestRegistrar(const char* name, std::function<void()> fn) {
        TestState::tests().push_back({name, fn});
    }
};

} // namespace test

// --- Assertion macros ---

// Helper for unique identifiers from __LINE__
#define TEST_CONCAT_(a, b) a##b
#define TEST_CONCAT(a, b) TEST_CONCAT_(a, b)

#define TEST_CASE(name) \
    static void TEST_CONCAT(test_fn_, __LINE__)(); \
    static test::TestRegistrar TEST_CONCAT(test_reg_, __LINE__)(name, TEST_CONCAT(test_fn_, __LINE__)); \
    static void TEST_CONCAT(test_fn_, __LINE__)()

#define CHECK(expr) \
    do { if (expr) { test::TestState::instance().pass(); } \
         else { test::TestState::instance().fail(#expr, __FILE__, __LINE__); } \
    } while(0)

#define CHECK_EQ(a, b) \
    do { if ((a) == (b)) { test::TestState::instance().pass(); } \
         else { std::ostringstream _oss; _oss << #a " == " #b " (" << (a) << " != " << (b) << ")"; \
                test::TestState::instance().fail(_oss.str(), __FILE__, __LINE__); } \
    } while(0)

#define CHECK_NEAR(a, b, tol) \
    do { if (std::abs((a) - (b)) < (tol)) { test::TestState::instance().pass(); } \
         else { std::ostringstream _oss; _oss << #a " ≈ " #b " (diff=" << std::abs((a)-(b)) << ", tol=" << (tol) << ")"; \
                test::TestState::instance().fail(_oss.str(), __FILE__, __LINE__); } \
    } while(0)

#define CHECK_GT(a, b) \
    do { if ((a) > (b)) { test::TestState::instance().pass(); } \
         else { std::ostringstream _oss; _oss << #a " > " #b " (" << (a) << " <= " << (b) << ")"; \
                test::TestState::instance().fail(_oss.str(), __FILE__, __LINE__); } \
    } while(0)

#define CHECK_LT(a, b) \
    do { if ((a) < (b)) { test::TestState::instance().pass(); } \
         else { std::ostringstream _oss; _oss << #a " < " #b " (" << (a) << " >= " << (b) << ")"; \
                test::TestState::instance().fail(_oss.str(), __FILE__, __LINE__); } \
    } while(0)

#define CHECK_GE(a, b) \
    do { if ((a) >= (b)) { test::TestState::instance().pass(); } \
         else { std::ostringstream _oss; _oss << #a " >= " #b " (" << (a) << " < " << (b) << ")"; \
                test::TestState::instance().fail(_oss.str(), __FILE__, __LINE__); } \
    } while(0)

#define CHECK_THROWS(expr) \
    do { bool _caught = false; \
         try { expr; } catch (...) { _caught = true; } \
         if (_caught) { test::TestState::instance().pass(); } \
         else { test::TestState::instance().fail(#expr " did not throw", __FILE__, __LINE__); } \
    } while(0)

// Main entry point — call in main()
#define RUN_ALL_TESTS() test::TestState::instance().run_all()

#endif // RCPPML_TEST_FRAMEWORK_HPP
