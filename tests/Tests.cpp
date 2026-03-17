
// ------------------------------------------------

#include "gtest/gtest.h"

// ------------------------------------------------

#include "basic_simd.hpp"

// ------------------------------------------------

#include <numbers>

// ------------------------------------------------

namespace kaixo::test {

    // ------------------------------------------------

    using simd_type = simd_256;
    using simdi_type = simd_256i;

    // ------------------------------------------------

    constexpr float Epsilon = 1e-9f;
    constexpr float MediumEpsilon = 1e-6f;
    constexpr float LargeEpsilon = 1e-3f;

    // ------------------------------------------------

    class BasicFloatOperationsTests : public ::testing::TestWithParam<std::tuple<float, float>> {};

    TEST_P(BasicFloatOperationsTests, BasicOperations) {
        auto [a, b] = GetParam();

        simd_type a_s = a;
        simd_type b_s = b;

        EXPECT_NEAR((a_s + b_s)[0], a + b, Epsilon);
        EXPECT_NEAR((a_s - b_s)[0], a - b, Epsilon);
        EXPECT_NEAR((a_s * b_s)[0], a * b, Epsilon);
        EXPECT_NEAR((a_s / b_s)[0], a / b, Epsilon);

        EXPECT_NEAR(simd_type::max(a_s, b_s)[0], std::max(a, b), Epsilon);
        EXPECT_NEAR(simd_type::min(a_s, b_s)[0], std::min(a, b), Epsilon);

        EXPECT_NEAR(simd_type::floor(a_s)[0], std::floor(a), Epsilon);
        EXPECT_NEAR(simd_type::trunc(a_s)[0], std::trunc(a), Epsilon);
        EXPECT_NEAR(simd_type::ceil(a_s)[0], std::ceil(a), Epsilon);
        EXPECT_NEAR(simd_type::round(a_s)[0], std::round(a), Epsilon);

        EXPECT_NEAR(simd_type::abs(a_s)[0], std::abs(a), Epsilon);
    }
    
    INSTANTIATE_TEST_CASE_P(Basic, BasicFloatOperationsTests, ::testing::Values(
          std::make_tuple(    1.f,     1.f)
        , std::make_tuple(   -1.f,     1.f)
        , std::make_tuple(   -1.f,     9.f)
        , std::make_tuple(   -9.f,     1.f)
        , std::make_tuple(   -9.f,     9.f)
        , std::make_tuple(    1.f,    -1.f)
        , std::make_tuple(    1.f,    -9.f)
        , std::make_tuple(    9.f,    -9.f)
        , std::make_tuple(   -1.f,    -1.f)
        , std::make_tuple(   -1.f,    -9.f)
        , std::make_tuple(   -9.f,    -9.f)
        , std::make_tuple(   1e9f,    1e9f)
        , std::make_tuple(  -1e9f,    1e9f)
        , std::make_tuple(   1e9f,   -1e9f)
        , std::make_tuple(  -1e9f,   -1e9f)
    ));

    // ------------------------------------------------

    class MathFloatOperationsTests : public ::testing::TestWithParam<std::tuple<float, float>> {};

    TEST_P(MathFloatOperationsTests, MathOperations) {
        auto [a, b] = GetParam();

        simd_type a_s = a;
        simd_type b_s = b;

        EXPECT_NEAR(simd_type::rcp(a_s)[0], 1 / a, LargeEpsilon);
        EXPECT_NEAR(simd_type::log(a_s)[0], std::log(a), MediumEpsilon);
        EXPECT_NEAR(simd_type::log2(a_s)[0], std::log2(a), MediumEpsilon);
        EXPECT_NEAR(simd_type::log10(a_s)[0], std::log10(a), MediumEpsilon);
        EXPECT_NEAR(simd_type::sqrt(a_s)[0], std::sqrt(a), MediumEpsilon);
        EXPECT_NEAR(simd_type::cbrt(a_s)[0], std::cbrt(a), MediumEpsilon);
        EXPECT_NEAR(simd_type::exp(a_s)[0], std::exp(a), MediumEpsilon);
        EXPECT_NEAR(simd_type::exp2(a_s)[0], std::exp2(a), MediumEpsilon);
        EXPECT_NEAR(simd_type::tanh(a_s)[0], std::tanh(a), MediumEpsilon);
        EXPECT_NEAR(simd_type::cos(a_s)[0], std::cos(a), MediumEpsilon);
        EXPECT_NEAR(simd_type::cosh(a_s)[0], std::cosh(a), MediumEpsilon);
        EXPECT_NEAR(simd_type::sin(a_s)[0], std::sin(a), MediumEpsilon);
        EXPECT_NEAR(simd_type::sinh(a_s)[0], std::sinh(a), MediumEpsilon);
        EXPECT_NEAR(simd_type::pow(a_s, b_s)[0], std::pow(a, b), MediumEpsilon);
    }
    
    INSTANTIATE_TEST_CASE_P(Basic, MathFloatOperationsTests, ::testing::Values(
          std::make_tuple(    1.f,     1.f)
        , std::make_tuple( 0.001f,  0.001f)
        , std::make_tuple(   0.1f,    0.1f)
        , std::make_tuple(    9.f,     1.f)
        , std::make_tuple(    1.f,     9.f)
        , std::make_tuple(    9.f,     9.f)
    ));

    // ------------------------------------------------

    class BasicIntOperationsTests : public ::testing::TestWithParam<std::tuple<int, int>> {};

    TEST_P(BasicIntOperationsTests, BasicOperations) {
        auto [a, b] = GetParam();

        simdi_type a_s = a;
        simdi_type b_s = b;

        ASSERT_EQ((a_s + b_s)[0], a + b);
        ASSERT_EQ((a_s - b_s)[0], a - b);
        ASSERT_EQ((a_s * b_s)[0], a * b);
        ASSERT_EQ((a_s / b_s)[0], a / b);

        EXPECT_NEAR(simdi_type::max(a_s, b_s)[0], std::max(a, b), Epsilon);
        EXPECT_NEAR(simdi_type::min(a_s, b_s)[0], std::min(a, b), Epsilon);
    }
    
    INSTANTIATE_TEST_CASE_P(Basic, BasicIntOperationsTests, ::testing::Values(
          std::make_tuple(     0,      1)
        , std::make_tuple(     1,      1)
        , std::make_tuple(    -1,      1)
        , std::make_tuple(    -1,      9)
        , std::make_tuple(    -9,      1)
        , std::make_tuple(    -9,      9)
        , std::make_tuple(     1,     -1)
        , std::make_tuple(     0,     -1)
        , std::make_tuple(     1,     -9)
        , std::make_tuple(     9,     -9)
        , std::make_tuple(    -1,     -1)
        , std::make_tuple(    -0,     -1)
        , std::make_tuple(    -1,     -9)
        , std::make_tuple(    -9,     -9)
        , std::make_tuple(   1e9,    1e9)
        , std::make_tuple(  -1e9,    1e9)
        , std::make_tuple(   1e9,   -1e9)
        , std::make_tuple(  -1e9,   -1e9)
    ));

    // ------------------------------------------------

}

// ------------------------------------------------
