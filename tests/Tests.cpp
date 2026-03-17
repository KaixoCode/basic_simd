
// ------------------------------------------------

#include "gtest/gtest.h"

// ------------------------------------------------

#include "basic_simd.hpp"

// ------------------------------------------------

#include <numbers>
#include <ranges>

// ------------------------------------------------

namespace kaixo::test {

    // ------------------------------------------------

    constexpr float Epsilon = 1e-9f;
    constexpr float MediumEpsilon = 1e-6f;
    constexpr float LargeEpsilon = 1e-3f;

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

        float buffer[16]{};

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
        EXPECT_NEAR(simd_256::log(a_s)[0], std::log(a), MediumEpsilon);
        EXPECT_NEAR(simd_256::log2(a_s)[0], std::log2(a), MediumEpsilon);
        EXPECT_NEAR(simd_256::log10(a_s)[0], std::log10(a), MediumEpsilon);
        EXPECT_NEAR(simd_256::sqrt(a_s)[0], std::sqrt(a), MediumEpsilon);
        EXPECT_NEAR(simd_256::cbrt(a_s)[0], std::cbrt(a), MediumEpsilon);
        EXPECT_NEAR(simd_256::exp(a_s)[0], std::exp(a), MediumEpsilon);
        EXPECT_NEAR(simd_256::exp2(a_s)[0], std::exp2(a), MediumEpsilon);
        EXPECT_NEAR(simd_256::tanh(a_s)[0], std::tanh(a), MediumEpsilon);
        EXPECT_NEAR(simd_256::cos(a_s)[0], std::cos(a), MediumEpsilon);
        EXPECT_NEAR(simd_256::cosh(a_s)[0], std::cosh(a), MediumEpsilon);
        EXPECT_NEAR(simd_256::sin(a_s)[0], std::sin(a), MediumEpsilon);
        EXPECT_NEAR(simd_256::sinh(a_s)[0], std::sinh(a), MediumEpsilon);
        EXPECT_NEAR(simd_256::pow(a_s, b_s)[0], std::pow(a, b), MediumEpsilon);
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

        int buffer[16]{};

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
