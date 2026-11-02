
// ------------------------------------------------

#include "gtest/gtest.h"

// ------------------------------------------------

#include "basic_simd.hpp"

// ------------------------------------------------

#include <numbers>
#include <limits>
#include <random>
#include <ranges>

// ------------------------------------------------

namespace kaixo::test {

    // ------------------------------------------------

    constexpr float Epsilon = 1e-9f;
    constexpr float MediumEpsilon = 1e-6f;
    constexpr float LargeEpsilon = 5e-3f;

    void expect_math_near(float actual, float expected) {
        EXPECT_NEAR(actual, expected,
            std::max(MediumEpsilon, std::abs(expected) * MediumEpsilon));
    }

    // ------------------------------------------------

    TEST(BasicFloatTests, Initialization) {
        simd_256 val = simd_256::setzero();
        for (std::size_t i = 0; i < val.elements; ++i) {
            EXPECT_EQ(val[i], 0);
        }

        val = simd_256::setincr();
        for (std::size_t i = 0; i < val.elements; ++i) {
            EXPECT_EQ(val[i], i);
        }

        val = simd_256::set1(123.456f);
        for (std::size_t i = 0; i < val.elements; ++i) {
            EXPECT_EQ(val[i], 123.456f);
        }

        simd_mask_256 mask = simd_256::true_mask();
        for (std::size_t i = 0; i < mask.elements; ++i) {
            EXPECT_EQ(std::popcount(std::bit_cast<std::uint32_t>(mask[i])), mask.bytes_per_element * 8);
        }

        mask = simd_256::false_mask();
        for (std::size_t i = 0; i < mask.elements; ++i) {
            EXPECT_EQ(mask[i], 0);
        }

        alignas(simd_256::alignment) constexpr float data[]{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };

        val = simd_256::load(data);
        for (std::size_t i = 0; i < val.elements; ++i) {
            EXPECT_EQ(val[i], data[i]);
        }

        val = simd_256::load(data).reverse();
        for (std::size_t i = 0; i < val.elements; ++i) {
            EXPECT_EQ(val[i], data[val.elements - i - 1]);
        }

        val = simd_256::loadu(data);
        for (std::size_t i = 0; i < val.elements; ++i) {
            EXPECT_EQ(val[i], data[i]);
        }

        val = simd_256::loadr(data);
        for (std::size_t i = 0; i < val.elements; ++i) {
            EXPECT_EQ(val[i], data[val.elements - i - 1]);
        }

        alignas(simd_256::alignment) float buffer[16]{};

        val = simd_256::load(data);
        val.store(buffer);
        for (std::size_t i = 0; i < val.elements; ++i) {
            EXPECT_EQ(buffer[i], data[i]);
        }

        val = simd_256::load(data);
        val.storeu(buffer);
        for (std::size_t i = 0; i < val.elements; ++i) {
            EXPECT_EQ(buffer[i], data[i]);
        }

        val = simd_256::load(data);
        val.stream(buffer);
        for (std::size_t i = 0; i < val.elements; ++i) {
            EXPECT_EQ(buffer[i], data[i]);
        }
    }

    TEST(BasicFloatTests, ScalarFallback) {
        using scalar_simd = basic_simd<float, simd_scalar_fallback_abi<float>>;

        alignas(scalar_simd::alignment) float input[scalar_simd::elements];
        alignas(scalar_simd::alignment) float exponents[scalar_simd::elements];
        for (std::size_t i = 0; i < scalar_simd::elements; ++i) {
            input[i] = static_cast<float>(i + 1);
            exponents[i] = static_cast<float>(i + 2);
        }

        scalar_simd values = scalar_simd::load(input);
        scalar_simd powers = scalar_simd::load(exponents);
        const scalar_simd expValues = values.exp();
        const scalar_simd powValues = values.pow(powers);

        for (std::size_t i = 0; i < scalar_simd::elements; ++i) {
            const float expectedExp = std::exp(input[i]);
            const float expectedPow = std::pow(input[i], exponents[i]);
            EXPECT_NEAR(expValues[i], expectedExp, expectedExp * 1e-6f);
            EXPECT_NEAR(powValues[i], expectedPow, expectedPow * 1e-6f);
        }
    }

    TEST(BasicFloatTests, Convert) {
        EXPECT_EQ(simd_256::set1(-0.9f).to_int()[0], 0);
        EXPECT_EQ(simd_256::set1(-1.f).to_int()[0], -1);
        EXPECT_EQ(simd_256::set1(-1.5f).to_int()[0], -1);
        EXPECT_EQ(simd_256::set1(1.f).to_int()[0], 1);
        EXPECT_EQ(simd_256::set1(1.5f).to_int()[0], 1);
        EXPECT_EQ(simd_256::set1(1.9f).to_int()[0], 1);
        EXPECT_EQ(simd_256::set1(1.9999999f).to_int()[0], 1);
        EXPECT_EQ(simd_256::set1(2.0f).to_int()[0], 2);

        EXPECT_EQ(std::bit_cast<std::uint32_t>(simd_256::set1(std::bit_cast<float>(0xFFFFFFFF)).as_int()[0]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
        EXPECT_EQ(std::bit_cast<std::uint32_t>(simd_256::set1(std::bit_cast<float>(0x01010101)).as_int()[0]), std::bit_cast<std::uint32_t>(0x01010101));
    }

    TEST(BasicFloatTests, Bit) {
        simd_256 a = simd_256::set1(std::bit_cast<float>(0xFFFFFFFF));
        simd_256 b = simd_256::setzero();

        simd_256 or1 = a | b;
        simd_256 or2 = a | a;
        simd_256 or3 = b | b;
        simd_256 and1 = a & b;
        simd_256 and2 = a & a;
        simd_256 and3 = b & b;
        simd_256 xor1 = a ^ b;
        simd_256 xor2 = a ^ a;
        simd_256 xor3 = b ^ b;
        simd_256 not1 = ~a;
        simd_256 not2 = ~b;
        for (std::size_t i = 0; i < simd_256::elements; ++i) {
            EXPECT_EQ(std::bit_cast<std::uint32_t>(or1[i]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(or2[i]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(or3[i]), std::bit_cast<std::uint32_t>(0x0));

            EXPECT_EQ(std::bit_cast<std::uint32_t>(and1[i]), std::bit_cast<std::uint32_t>(0x0));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(and2[i]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(and3[i]), std::bit_cast<std::uint32_t>(0x0));

            EXPECT_EQ(std::bit_cast<std::uint32_t>(xor1[i]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(xor2[i]), std::bit_cast<std::uint32_t>(0x0));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(xor3[i]), std::bit_cast<std::uint32_t>(0x0));

            EXPECT_EQ(std::bit_cast<std::uint32_t>(not1[i]), std::bit_cast<std::uint32_t>(0x0));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(not2[i]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
        }
    }

    TEST(BasicFloatTests, Boolean) {
        simd_mask_256 a = simd_256::true_mask();
        simd_mask_256 b = simd_256::false_mask();

        simd_mask_256 or1 = a || b;
        simd_mask_256 or2 = a || a;
        simd_mask_256 or3 = b || b;
        simd_mask_256 and1 = a && b;
        simd_mask_256 and2 = a && a;
        simd_mask_256 and3 = b && b;
        simd_mask_256 not1 = !a;
        simd_mask_256 not2 = !b;
        for (std::size_t i = 0; i < simd_256::elements; ++i) {
            EXPECT_EQ(std::bit_cast<std::uint32_t>(or1[i]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(or2[i]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(or3[i]), std::bit_cast<std::uint32_t>(0x0));

            EXPECT_EQ(std::bit_cast<std::uint32_t>(and1[i]), std::bit_cast<std::uint32_t>(0x0));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(and2[i]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(and3[i]), std::bit_cast<std::uint32_t>(0x0));

            EXPECT_EQ(std::bit_cast<std::uint32_t>(not1[i]), std::bit_cast<std::uint32_t>(0x0));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(not2[i]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
        }
    }

    TEST(BasicFloatTests, Equality) {
        simd_256 a = 1.f;

        EXPECT_EQ(std::bit_cast<std::uint32_t>((a == 1)[0]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
        EXPECT_EQ(std::bit_cast<std::uint32_t>((a == 1.0000001f)[0]), std::bit_cast<std::uint32_t>(0x0));
        EXPECT_EQ(std::bit_cast<std::uint32_t>((a == 0.9999999f)[0]), std::bit_cast<std::uint32_t>(0x0));
        EXPECT_EQ(std::bit_cast<std::uint32_t>((a <= 1)[0]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
        EXPECT_EQ(std::bit_cast<std::uint32_t>((a <= 0.9999999f)[0]), std::bit_cast<std::uint32_t>(0x0));
        EXPECT_EQ(std::bit_cast<std::uint32_t>((a < 1)[0]), std::bit_cast<std::uint32_t>(0x0));
        EXPECT_EQ(std::bit_cast<std::uint32_t>((a < 1.0000001f)[0]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
        EXPECT_EQ(std::bit_cast<std::uint32_t>((a >= 1)[0]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
        EXPECT_EQ(std::bit_cast<std::uint32_t>((a >= 1.0000001f)[0]), std::bit_cast<std::uint32_t>(0x0));
        EXPECT_EQ(std::bit_cast<std::uint32_t>((a > 1)[0]), std::bit_cast<std::uint32_t>(0x0));
        EXPECT_EQ(std::bit_cast<std::uint32_t>((a > 0.9999999f)[0]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
        EXPECT_EQ(std::bit_cast<std::uint32_t>((a != 1)[0]), std::bit_cast<std::uint32_t>(0x0));
        EXPECT_EQ(std::bit_cast<std::uint32_t>((a != 1.0000001f)[0]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
        EXPECT_EQ(std::bit_cast<std::uint32_t>((a != 0.9999999f)[0]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
        EXPECT_EQ(std::bit_cast<std::uint32_t>((a.is_negative())[0]), std::bit_cast<std::uint32_t>(0x0));
        EXPECT_TRUE(std::bit_cast<std::uint32_t>(((-a).is_negative())[0]) != 0);
    }

    TEST(BasicFloatTests, Sum) {
        simd_256 a = simd_256::setincr();
        float result = a.sum();
        ASSERT_EQ(result, std::ranges::fold_left(std::views::iota(0ull, simd_256::elements), 0, std::plus{}));
    }

    TEST(BasicFloatTests, Sign) {
        simd_256 a = -1;
        simd_256 b = 1;

        EXPECT_EQ(simd_256::copysign(a, b)[0], a[0]);
        EXPECT_EQ(simd_256::copysign(b, a)[0], b[0]);
        EXPECT_EQ(simd_256::copysign(a, a)[0], a[0]);
        EXPECT_EQ(simd_256::copysign(b, b)[0], b[0]);
        EXPECT_EQ(simd_256::xorsign(a, b)[0], a[0]);
        EXPECT_EQ(simd_256::xorsign(b, a)[0], a[0]);
        EXPECT_EQ(simd_256::xorsign(a, a)[0], b[0]);
        EXPECT_EQ(simd_256::xorsign(b, b)[0], b[0]);
        EXPECT_EQ(simd_256::orsign(a, b)[0], a[0]);
        EXPECT_EQ(simd_256::orsign(b, a)[0], a[0]);
        EXPECT_EQ(simd_256::orsign(a, a)[0], a[0]);
        EXPECT_EQ(simd_256::orsign(b, b)[0], b[0]);
        EXPECT_EQ(simd_256::sign(a)[0], a[0]);
        EXPECT_EQ(simd_256::sign(b)[0], b[0]);
    }

    // ------------------------------------------------

    class BasicFloatOperationsTests : public ::testing::TestWithParam<std::tuple<float, float>> {};

    TEST_P(BasicFloatOperationsTests, BasicOperations) {
        auto [a, b] = GetParam();

        simd_256 a_s = a;
        simd_256 b_s = b;

        EXPECT_NEAR((a_s + b_s)[0], a + b, Epsilon);
        EXPECT_NEAR((a_s - b_s)[0], a - b, Epsilon);
        EXPECT_NEAR((a_s * b_s)[0], a * b, Epsilon);
        EXPECT_NEAR((a_s / b_s)[0], a / b, Epsilon);
        EXPECT_NEAR(simd_256::fmadd(a_s, b_s, a_s)[0], a * b + a, Epsilon);
        EXPECT_NEAR(simd_256::fmsub(a_s, b_s, a_s)[0], a * b - a, Epsilon);
        EXPECT_NEAR((-b_s)[0], -b, Epsilon);

        EXPECT_NEAR(simd_256::max(a_s, b_s)[0], std::max(a, b), Epsilon);
        EXPECT_NEAR(simd_256::min(a_s, b_s)[0], std::min(a, b), Epsilon);

        EXPECT_NEAR(simd_256::floor(a_s)[0], std::floor(a), Epsilon);
        EXPECT_NEAR(simd_256::trunc(a_s)[0], std::trunc(a), Epsilon);
        EXPECT_NEAR(simd_256::ceil(a_s)[0], std::ceil(a), Epsilon);
        EXPECT_NEAR(simd_256::round(a_s)[0], std::round(a), Epsilon);

        EXPECT_NEAR(simd_256::abs(a_s)[0], std::abs(a), Epsilon);
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

        simd_256 a_s = a;
        simd_256 b_s = b;

        EXPECT_NEAR(simd_256::rcp(a_s)[0], 1 / a, LargeEpsilon);
        expect_math_near(simd_256::log(a_s)[0], std::log(a));
        expect_math_near(simd_256::log2(a_s)[0], std::log2(a));
        expect_math_near(simd_256::log10(a_s)[0], std::log10(a));
        expect_math_near(simd_256::sqrt(a_s)[0], std::sqrt(a));
        expect_math_near(simd_256::cbrt(a_s)[0], std::cbrt(a));
        expect_math_near(simd_256::exp(a_s)[0], std::exp(a));
        expect_math_near(simd_256::exp2(a_s)[0], std::exp2(a));
        expect_math_near(a_s.exp10()[0], std::pow(10.f, a));
        expect_math_near(simd_256::tanh(a_s)[0], std::tanh(a));
        expect_math_near(simd_256::cos(a_s)[0], std::cos(a));
        expect_math_near(simd_256::cosh(a_s)[0], std::cosh(a));
        expect_math_near(simd_256::sin(a_s)[0], std::sin(a));
        expect_math_near(simd_256::sinh(a_s)[0], std::sinh(a));
        expect_math_near(simd_256::pow(a_s, b_s)[0], std::pow(a, b));
    }
    
    INSTANTIATE_TEST_CASE_P(Basic, MathFloatOperationsTests, ::testing::Values(
          std::make_tuple(    1.f,     1.f)
        , std::make_tuple( 0.001f,  0.001f)
        , std::make_tuple(   0.1f,    0.1f)
        , std::make_tuple(    9.f,     1.f)
        , std::make_tuple(    1.f,     9.f)
        , std::make_tuple(    9.f,     9.f)
    ));

    TEST(BasicFloatTests, MathOperationsFuzzy) {
        std::mt19937 random(0x5eed);
        std::uniform_real_distribution<float> positive(0.01f, 100.f);
        std::uniform_real_distribution<float> power_base(0.1f, 10.f);
        std::uniform_real_distribution<float> general(-5.f, 5.f);
        std::uniform_real_distribution<float> power_exponent(-3.f, 3.f);
        std::uniform_real_distribution<float> trig(-16.f, 16.f);

        alignas(simd_256::alignment) float inputs[simd_256::elements];
        alignas(simd_256::alignment) float root_inputs[simd_256::elements];
        alignas(simd_256::alignment) float power_bases[simd_256::elements];
        alignas(simd_256::alignment) float exponents[simd_256::elements];
        for (int iteration = 0; iteration < 256; ++iteration) {
            for (std::size_t lane = 0; lane < simd_256::elements; ++lane) {
                inputs[lane] = positive(random);
                root_inputs[lane] = std::copysign(inputs[lane], lane % 2 ? -1.f : 1.f);
                power_bases[lane] = power_base(random);
                exponents[lane] = power_exponent(random);
            }

            auto values = simd_256::load(inputs);
            const auto root_values = simd_256::load(root_inputs);
            auto power_values = simd_256::load(power_bases);
            const auto powers_input = simd_256::load(exponents);
            const auto logs = values.log();
            const auto logs2 = values.log2();
            const auto logs10 = values.log10();
            const auto roots3 = root_values.cbrt();
            const auto powered = power_values.pow(powers_input);

            for (std::size_t lane = 0; lane < simd_256::elements; ++lane) {
                expect_math_near(logs[lane], std::log(inputs[lane]));
                expect_math_near(logs2[lane], std::log2(inputs[lane]));
                expect_math_near(logs10[lane], std::log10(inputs[lane]));
                expect_math_near(roots3[lane], std::cbrt(root_inputs[lane]));
                expect_math_near(powered[lane], std::pow(power_bases[lane], exponents[lane]));
                inputs[lane] = general(random);
            }

            const auto exponent_values = simd_256::load(inputs);
            const auto exponentials = exponent_values.exp();
            const auto exponentials2 = exponent_values.exp2();
            const auto exponentials10 = exponent_values.exp10();
            const auto hyperbolic_tangents = exponent_values.tanh();
            const auto hyperbolic_sines = exponent_values.sinh();
            const auto hyperbolic_cosines = exponent_values.cosh();

            for (std::size_t lane = 0; lane < simd_256::elements; ++lane) {
                expect_math_near(exponentials[lane], std::exp(inputs[lane]));
                expect_math_near(exponentials2[lane], std::exp2(inputs[lane]));
                expect_math_near(exponentials10[lane], std::pow(10.f, inputs[lane]));
                expect_math_near(hyperbolic_tangents[lane], std::tanh(inputs[lane]));
                expect_math_near(hyperbolic_sines[lane], std::sinh(inputs[lane]));
                expect_math_near(hyperbolic_cosines[lane], std::cosh(inputs[lane]));

                inputs[lane] = trig(random);
            }

            const auto trig_values = simd_256::load(inputs);
            const auto sines = trig_values.sin();
            const auto cosines = trig_values.cos();
            for (std::size_t lane = 0; lane < simd_256::elements; ++lane) {
                expect_math_near(sines[lane], std::sin(inputs[lane]));
                expect_math_near(cosines[lane], std::cos(inputs[lane]));
            }

            expect_math_near(simd_256::setzero().cbrt()[0], 0.f);
            expect_math_near(simd_256::set1(-2.f).pow(simd_256::set1(3.f))[0], -8.f);
            expect_math_near(simd_256::set1(-2.f).pow(simd_256::set1(2.f))[0], 4.f);
            expect_math_near(simd_256::set1(-2.f).pow(simd_256::set1(-3.f))[0], -0.125f);
            expect_math_near(simd_256::setzero().pow(simd_256::set1(2.f))[0], 0.f);
        }
    }

#if defined(_MSC_VER) && _MSC_VER >= 1933
    TEST(BasicFloatTests, MathOperationsMatchSvml) {
        alignas(simd_256::alignment) float input[]{ 0.125f, 0.5f, 0.75f, 1.f, 1.5f, 2.f, 4.f, 8.f };
        alignas(simd_256::alignment) float exponent[]{ -2.f, -1.f, -0.5f, 0.f, 0.5f, 1.f, 2.f, 3.f };
        alignas(simd_256::alignment) float baseline[simd_256::elements];

        auto values = simd_256::load(input);
        const auto powers_input = simd_256::load(exponent);
        const auto input_vector = _mm256_load_ps(input);
        const auto exponent_vector = _mm256_load_ps(exponent);
        const auto compare = [&](const simd_256& actual, __m256 expected) {
            _mm256_store_ps(baseline, expected);
            for (std::size_t lane = 0; lane < simd_256::elements; ++lane) {
                expect_math_near(actual[lane], baseline[lane]);
            }
        };

        compare(values.log(), _mm256_log_ps(input_vector));
        compare(values.log2(), _mm256_log2_ps(input_vector));
        compare(values.log10(), _mm256_log10_ps(input_vector));
        compare(values.cbrt(), _mm256_cbrt_ps(input_vector));
        compare(values.exp(), _mm256_exp_ps(input_vector));
        compare(values.exp2(), _mm256_exp2_ps(input_vector));
        compare(values.exp10(), _mm256_pow_ps(_mm256_set1_ps(10.f), input_vector));
        compare(values.tanh(), _mm256_tanh_ps(input_vector));
        compare(values.cos(), _mm256_cos_ps(input_vector));
        compare(values.cosh(), _mm256_cosh_ps(input_vector));
        compare(values.sin(), _mm256_sin_ps(input_vector));
        compare(values.sinh(), _mm256_sinh_ps(input_vector));
        compare(values.pow(powers_input), _mm256_pow_ps(input_vector, exponent_vector));
    }

    TEST(BasicFloatTests, SvmlDocumentedRanges) {
        std::mt19937 random(0x51A1);
        std::uniform_real_distribution<float> trig_wide(0.f, 30000.f);
        std::uniform_real_distribution<float> exponential(-100.f, 100.f);
        std::uniform_real_distribution<float> pow_input(-30.f, 30.f);
        alignas(simd_256::alignment) float inputs[simd_256::elements];
        alignas(simd_256::alignment) float exponents[simd_256::elements];
        alignas(simd_256::alignment) float baseline[simd_256::elements];

        bool comparisons_passed = true;
        const char* failing_function = "";
        std::size_t failing_lane = 0;
        float failing_input = 0.f;
        float failing_actual = 0.f;
        float failing_expected = 0.f;
        constexpr float RangeRelativeEpsilon = 1e-5f;
        const auto compare = [&](simd_256 actual, __m256 expected,
                                 const char* function, float input,
                                 bool skip_nonfinite = false) {
            _mm256_store_ps(baseline, expected);
            for (std::size_t lane = 0; lane < simd_256::elements; ++lane) {
                if (!comparisons_passed) return;
                if (skip_nonfinite && !std::isfinite(baseline[lane])) continue;
                if (std::isinf(baseline[lane]) || std::isnan(baseline[lane])) {
                    if (actual[lane] != baseline[lane]) {
                        comparisons_passed = false;
                    }
                } else {
                    const float tolerance = std::max(
                        MediumEpsilon, std::abs(baseline[lane]) * RangeRelativeEpsilon);
                    if (!std::isfinite(actual[lane])
                        || std::abs(actual[lane] - baseline[lane]) > tolerance) {
                        comparisons_passed = false;
                    }
                }
                if (!comparisons_passed) {
                    failing_function = function;
                    failing_lane = lane;
                    failing_input = input;
                    failing_actual = actual[lane];
                    failing_expected = baseline[lane];
                }
            }
        };
        const auto compare_trigonometry = [&](float input) {
            std::fill(std::begin(inputs), std::end(inputs), input);
            const auto values = simd_256::load(inputs);
            const auto input_vector = _mm256_load_ps(inputs);
            compare(values.sin(), _mm256_sin_ps(input_vector), "sin", input);
            compare(values.cos(), _mm256_cos_ps(input_vector), "cos", input);
        };
        const auto compare_logarithm = [&](float input) {
            std::fill(std::begin(inputs), std::end(inputs), input);
            const auto values = simd_256::load(inputs);
            compare(values.log(), _mm256_log_ps(_mm256_load_ps(inputs)), "log", input);
            compare(values.log2(), _mm256_log2_ps(_mm256_load_ps(inputs)), "log2", input);
            compare(values.log10(), _mm256_log10_ps(_mm256_load_ps(inputs)), "log10", input);
        };
        const auto compare_exponential = [&](float input) {
            std::fill(std::begin(inputs), std::end(inputs), input);
            const auto values = simd_256::load(inputs);
            const auto input_vector = _mm256_load_ps(inputs);
            compare(values.exp(), _mm256_exp_ps(input_vector), "exp", input, true);
        };

        for (int sample = 0; sample <= 4096; ++sample) {
            const float angle = 6.28f * static_cast<float>(sample) / 4096.f;
            compare_trigonometry(angle);
            if (!comparisons_passed) break;
            compare_trigonometry(30000.f * static_cast<float>(sample) / 4096.f);
            if (!comparisons_passed) break;

            const float exponent = -100.f + 200.f * static_cast<float>(sample) / 4096.f;
            compare_exponential(exponent);
            if (!comparisons_passed) break;

            const float logarithm_input = 1e38f * (static_cast<float>(sample) / 4096.f);
            compare_logarithm(logarithm_input);
            if (!comparisons_passed) break;
        }

        if (comparisons_passed) {
            compare_logarithm(0.f);
        }
        compare_logarithm(std::numeric_limits<float>::denorm_min());
        if (comparisons_passed) {
            compare_logarithm(std::numeric_limits<float>::min());
        }
        for (int exponent = -45; comparisons_passed && exponent <= 38; ++exponent) {
            compare_logarithm(std::pow(10.f, static_cast<float>(exponent)));
        }

        constexpr float power_boundaries[]{ -30.f, -1.f, 0.f, 1.f, 30.f };
        for (const float base : power_boundaries) {
            for (const float exponent : power_boundaries) {
                std::fill(std::begin(inputs), std::end(inputs), base);
                std::fill(std::begin(exponents), std::end(exponents), exponent);
                auto bases = simd_256::load(inputs);
                const auto powers_input = simd_256::load(exponents);
                compare(bases.pow(powers_input),
                    _mm256_pow_ps(_mm256_load_ps(inputs), _mm256_load_ps(exponents)),
                    "pow", base, true);
                if (!comparisons_passed) break;
            }
            if (!comparisons_passed) break;
        }

        for (int sample = 0; comparisons_passed && sample < 4096; ++sample) {
            for (std::size_t lane = 0; lane < simd_256::elements; ++lane) {
                inputs[lane] = pow_input(random);
                exponents[lane] = pow_input(random);
                if (inputs[lane] < 0.f) {
                    exponents[lane] = std::round(exponents[lane]);
                } else if (inputs[lane] == 0.f) {
                    inputs[lane] = 1.f;
                }
            }

            auto bases = simd_256::load(inputs);
            const auto powers_input = simd_256::load(exponents);
            compare(bases.pow(powers_input),
                _mm256_pow_ps(_mm256_load_ps(inputs), _mm256_load_ps(exponents)),
                "pow", inputs[0], true);
        }

        EXPECT_TRUE(comparisons_passed)
            << failing_function << " mismatch at input " << failing_input
            << ", lane " << failing_lane << ": got " << failing_actual
            << ", expected " << failing_expected;
    }
#endif

    // ------------------------------------------------
    
    TEST(BasicIntTests, Initialization) {
        simd_256i val = simd_256i::setzero();
        for (std::size_t i = 0; i < val.elements; ++i) {
            EXPECT_EQ(val[i], 0);
        }

        val = simd_256i::setincr();
        for (std::size_t i = 0; i < val.elements; ++i) {
            EXPECT_EQ(val[i], i);
        }

        val = simd_256i::set1(123456);
        for (std::size_t i = 0; i < val.elements; ++i) {
            EXPECT_EQ(val[i], 123456);
        }

        simd_mask_256i mask = simd_256i::true_mask();
        for (std::size_t i = 0; i < mask.elements; ++i) {
            EXPECT_EQ(std::popcount(std::bit_cast<std::uint32_t>(mask[i])), mask.bytes_per_element * 8);
        }

        mask = simd_256i::false_mask();
        for (std::size_t i = 0; i < mask.elements; ++i) {
            EXPECT_EQ(mask[i], 0);
        }

        alignas(simd_256i::alignment) constexpr int data[]{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };

        val = simd_256i::load(data);
        for (std::size_t i = 0; i < val.elements; ++i) {
            EXPECT_EQ(val[i], data[i]);
        }

        val = simd_256i::load(data).reverse();
        for (std::size_t i = 0; i < val.elements; ++i) {
            EXPECT_EQ(val[i], data[val.elements - i - 1]);
        }
        
        val = simd_256i::loadu(data);
        for (std::size_t i = 0; i < val.elements; ++i) {
            EXPECT_EQ(val[i], data[i]);
        }

        val = simd_256i::loadr(data);
        for (std::size_t i = 0; i < val.elements; ++i) {
            EXPECT_EQ(val[i], data[val.elements - i - 1]);
        }

        alignas(simd_256i::alignment) int buffer[16]{};

        val = simd_256i::load(data);
        val.store(buffer);
        for (std::size_t i = 0; i < val.elements; ++i) {
            EXPECT_EQ(buffer[i], data[i]);
        }

        val = simd_256i::load(data);
        val.storeu(buffer);
        for (std::size_t i = 0; i < val.elements; ++i) {
            EXPECT_EQ(buffer[i], data[i]);
        }

        val = simd_256i::load(data);
        val.stream(buffer);
        for (std::size_t i = 0; i < val.elements; ++i) {
            EXPECT_EQ(buffer[i], data[i]);
        }
    }
    
    TEST(BasicIntTests, Gather) {
        alignas(simd_256i::alignment) int indices[]{ 1, 4, 5, 1, 2, 3, 7, 8, 9, 1, 4, 3, 2, 1, 6, 8 };

        simd_256i index = simd_256i::load(indices);

        float fdata[]{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        int idata[]{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

        auto fgather = index.gather(fdata);
        auto igather = index.gather(idata);

        for (std::size_t i = 0; i < index.elements; ++i) {
            EXPECT_EQ(fgather[i], fdata[indices[i]]);
            EXPECT_EQ(igather[i], idata[indices[i]]);
        }
    }

    TEST(BasicIntTests, Convert) {
        EXPECT_EQ(simd_256i::set1(0).to_float()[0], 0);
        EXPECT_EQ(simd_256i::set1(2).to_float()[0], 2);
        EXPECT_EQ(simd_256i::set1(-10).to_float()[0], -10);

        EXPECT_EQ(std::bit_cast<std::uint32_t>(simd_256i::set1(std::bit_cast<int>(0xFFFFFFFF)).as_float()[0]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
        EXPECT_EQ(std::bit_cast<std::uint32_t>(simd_256i::set1(std::bit_cast<int>(0x01010101)).as_float()[0]), std::bit_cast<std::uint32_t>(0x01010101));
    }

    TEST(BasicIntTests, Bit) {
        simd_256i a = simd_256i::set1(std::bit_cast<int>(0xFFFFFFFF));
        simd_256i b = simd_256i::setzero();

        simd_256i or1 = a | b;
        simd_256i or2 = a | a;
        simd_256i or3 = b | b;
        simd_256i and1 = a & b;
        simd_256i and2 = a & a;
        simd_256i and3 = b & b;
        simd_256i xor1 = a ^ b;
        simd_256i xor2 = a ^ a;
        simd_256i xor3 = b ^ b;
        simd_256i not1 = ~a;
        simd_256i not2 = ~b;
        for (std::size_t i = 0; i < simd_256i::elements; ++i) {
            EXPECT_EQ(std::bit_cast<std::uint32_t>(or1[i]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(or2[i]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(or3[i]), std::bit_cast<std::uint32_t>(0x0));

            EXPECT_EQ(std::bit_cast<std::uint32_t>(and1[i]), std::bit_cast<std::uint32_t>(0x0));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(and2[i]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(and3[i]), std::bit_cast<std::uint32_t>(0x0));

            EXPECT_EQ(std::bit_cast<std::uint32_t>(xor1[i]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(xor2[i]), std::bit_cast<std::uint32_t>(0x0));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(xor3[i]), std::bit_cast<std::uint32_t>(0x0));

            EXPECT_EQ(std::bit_cast<std::uint32_t>(not1[i]), std::bit_cast<std::uint32_t>(0x0));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(not2[i]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
        }
    }

    TEST(BasicIntTests, Boolean) {
        simd_mask_256i a = simd_256i::true_mask();
        simd_mask_256i b = simd_256i::false_mask();

        simd_mask_256i or1 = a || b;
        simd_mask_256i or2 = a || a;
        simd_mask_256i or3 = b || b;
        simd_mask_256i and1 = a && b;
        simd_mask_256i and2 = a && a;
        simd_mask_256i and3 = b && b;
        simd_mask_256i not1 = !a;
        simd_mask_256i not2 = !b;
        for (std::size_t i = 0; i < simd_256i::elements; ++i) {
            EXPECT_EQ(std::bit_cast<std::uint32_t>(or1[i]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(or2[i]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(or3[i]), std::bit_cast<std::uint32_t>(0x0));

            EXPECT_EQ(std::bit_cast<std::uint32_t>(and1[i]), std::bit_cast<std::uint32_t>(0x0));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(and2[i]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(and3[i]), std::bit_cast<std::uint32_t>(0x0));

            EXPECT_EQ(std::bit_cast<std::uint32_t>(not1[i]), std::bit_cast<std::uint32_t>(0x0));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(not2[i]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
        }
    }

    TEST(BasicIntTests, Shift) {
        simd_256i val = 128;
        
        EXPECT_EQ((val << 1)[0], 256);
        EXPECT_EQ((val << 2)[0], 512);
        EXPECT_EQ((val << 3)[0], 1024);
        
        EXPECT_EQ((val >> 1)[0], 64);
        EXPECT_EQ((val >> 2)[0], 32);
        EXPECT_EQ((val >> 3)[0], 16);
        
        simd_256i shift = simd_256i::setincr();

        simd_256i result = val >> shift;
        for (std::size_t i = 0; i < simd_256i::elements; ++i) {
            EXPECT_EQ(result[i], 128 >> i);
        }

        result = val << shift;
        for (std::size_t i = 0; i < simd_256i::elements; ++i) {
            EXPECT_EQ(result[i], 128 << i);
        }
    }

    TEST(BasicIntTests, Equality) {
        simd_256i a = 1;

        EXPECT_EQ(std::bit_cast<std::uint32_t>((a == 1)[0]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
        EXPECT_EQ(std::bit_cast<std::uint32_t>((a <= 1)[0]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
        EXPECT_EQ(std::bit_cast<std::uint32_t>((a < 1)[0]), std::bit_cast<std::uint32_t>(0x0));
        EXPECT_EQ(std::bit_cast<std::uint32_t>((a >= 1)[0]), std::bit_cast<std::uint32_t>(0xFFFFFFFF));
        EXPECT_EQ(std::bit_cast<std::uint32_t>((a > 1)[0]), std::bit_cast<std::uint32_t>(0x0));
        EXPECT_EQ(std::bit_cast<std::uint32_t>((a != 1)[0]), std::bit_cast<std::uint32_t>(0x0));
    }

    // ------------------------------------------------

    class BasicIntOperationsTests : public ::testing::TestWithParam<std::tuple<int, int>> {};

    TEST_P(BasicIntOperationsTests, BasicOperations) {
        auto [a, b] = GetParam();

        simd_256i a_s = a;
        simd_256i b_s = b;

        ASSERT_EQ((a_s + b_s)[0], a + b);
        ASSERT_EQ((a_s - b_s)[0], a - b);
        ASSERT_EQ((a_s * b_s)[0], a * b);
        ASSERT_EQ((a_s / b_s)[0], a / b);

        EXPECT_NEAR(simd_256i::max(a_s, b_s)[0], std::max(a, b), Epsilon);
        EXPECT_NEAR(simd_256i::min(a_s, b_s)[0], std::min(a, b), Epsilon);
    }

    TEST(BasicIntTests, ScalarFallback) {
        using scalar_simd = basic_simd<int, simd_scalar_fallback_abi<int>>;

        alignas(scalar_simd::alignment) int numerators[scalar_simd::elements];
        alignas(scalar_simd::alignment) int denominators[scalar_simd::elements];
        for (std::size_t i = 0; i < scalar_simd::elements; ++i) {
            numerators[i] = static_cast<int>(100 + i);
            denominators[i] = static_cast<int>(i + 1);
        }

        const scalar_simd quotients = scalar_simd::load(numerators) / scalar_simd::load(denominators);

        for (std::size_t i = 0; i < scalar_simd::elements; ++i) {
            EXPECT_EQ(quotients[i], numerators[i] / denominators[i]);
        }
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
