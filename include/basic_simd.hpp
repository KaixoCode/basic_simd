#pragma once

// ------------------------------------------------

#include <algorithm>
#include <bit>
#include <concepts>
#include <type_traits>
#include <numeric>
#include <cmath>
#include <cstdint>

// ------------------------------------------------

#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
#define KAIXO_SIMD_ARM 1
#elif defined(__x86_64__) || defined(_M_AMD64)
#define KAIXO_SIMD_X86_64 1
#include <immintrin.h>
#endif

#if defined(_MSC_VER)
#if KAIXO_SIMD_X86_64
#include <intrin.h>
#elif KAIXO_SIMD_ARM
#include <arm_neon.h>
#endif
#define KAIXO_VECTORCALL __vectorcall
#define KAIXO_INLINE __forceinline
#elif defined(__GNUC__) || defined(__GNUG__) || defined(__clang__)
#define KAIXO_VECTORCALL
#define KAIXO_INLINE
#endif

// ------------------------------------------------

namespace kaixo {

    // ------------------------------------------------

    template<std::size_t Bytes> struct uint_type;
    template<> struct uint_type<1> : std::type_identity<std::uint8_t> {};
    template<> struct uint_type<2> : std::type_identity<std::uint16_t> {};
    template<> struct uint_type<4> : std::type_identity<std::uint32_t> {};
    template<> struct uint_type<8> : std::type_identity<std::uint64_t> {};
    template<std::size_t Bytes> using uint_t = typename uint_type<Bytes>::type;

    template<std::size_t Bytes> struct float_type;
    template<> struct float_type<4> : std::type_identity<float> {};
    template<> struct float_type<8> : std::type_identity<long double> {};
    template<std::size_t Bytes> using float_t = typename float_type<Bytes>::type;

    // ------------------------------------------------

    template<class T>
    struct simd_scalar_fallback_abi {

        // ------------------------------------------------

        using value_type = T;
        using mask_type = bool;
        using simd_type = T;
        using buffer_type = value_type*;
        using const_buffer_type = const value_type*;

        // ------------------------------------------------

        constexpr static std::size_t elements = 1;
        constexpr static std::size_t bytes = sizeof(T);
        constexpr static std::size_t alignment = alignof(T);

        // ------------------------------------------------

        using int_alias = uint_t<bytes>;
        using float_alias = float_t<bytes>;

        // ------------------------------------------------

        KAIXO_INLINE static value_type KAIXO_VECTORCALL index(simd_type a, std::size_t i) { return a; }

        // ------------------------------------------------
        
        KAIXO_INLINE static simd_type setzero() { return static_cast<simd_type>(0); }
        KAIXO_INLINE static simd_type setincr() { return static_cast<simd_type>(0); }
        KAIXO_INLINE static simd_type set1(value_type val) { return val; }

        KAIXO_INLINE static mask_type true_mask() { return true; }
        KAIXO_INLINE static mask_type false_mask() { return false; }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type load(const_buffer_type data) { return *data; };
        KAIXO_INLINE static simd_type loadr(const_buffer_type data) { return *data; };
        KAIXO_INLINE static simd_type loadu(const_buffer_type data) { return *data; };

        KAIXO_INLINE static void store(buffer_type to, simd_type from) { *to = from; };
        KAIXO_INLINE static void storeu(buffer_type to, simd_type from) { *to = from; };
        KAIXO_INLINE static void stream(buffer_type to, simd_type from) { *to = from; };

        KAIXO_INLINE static float gather(const float* data, simd_type index) requires std::integral<simd_type> { return data[index]; };
        KAIXO_INLINE static int gather(const int* data, simd_type index) requires std::integral<simd_type> { return data[index]; };

        // ------------------------------------------------

        KAIXO_INLINE static int_alias to_int(simd_type a) { return static_cast<int_alias>(a); }
        KAIXO_INLINE static int_alias as_int(simd_type a) { return std::bit_cast<int_alias>(a); }
        KAIXO_INLINE static float_alias to_float(simd_type a) { return static_cast<float_alias>(a); }
        KAIXO_INLINE static float_alias as_float(simd_type a) { return std::bit_cast<float_alias>(a); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type add(simd_type a, simd_type b) { return a + b; }
        KAIXO_INLINE static simd_type sub(simd_type a, simd_type b) { return a - b; }
        KAIXO_INLINE static simd_type mul(simd_type a, simd_type b) { return a * b; }
        KAIXO_INLINE static simd_type div(simd_type a, simd_type b) { return a / b; }

        KAIXO_INLINE static simd_type negate(simd_type a) { return -a; }

        KAIXO_INLINE static simd_type fmadd(simd_type a, simd_type b, simd_type c) { return a * b + c; }
        KAIXO_INLINE static simd_type fmsub(simd_type a, simd_type b, simd_type c) { return a * b - c; }

        // ------------------------------------------------
        
        KAIXO_INLINE static simd_type bit_and(simd_type a, simd_type b) { return std::bit_cast<simd_type>(std::bit_cast<int_alias>(a) & std::bit_cast<int_alias>(b)); }
        KAIXO_INLINE static simd_type bit_or(simd_type a, simd_type b) { return std::bit_cast<simd_type>(std::bit_cast<int_alias>(a) | std::bit_cast<int_alias>(b)); }
        KAIXO_INLINE static simd_type bit_xor(simd_type a, simd_type b) { return std::bit_cast<simd_type>(std::bit_cast<int_alias>(a) ^ std::bit_cast<int_alias>(b)); }
        KAIXO_INLINE static simd_type bit_not(simd_type a) { return std::bit_cast<simd_type>(std::bit_cast<int_alias>(a) ^ static_cast<int_alias>(0xFFFFFFFFFFFFFFFF)); }

        KAIXO_INLINE static simd_type bit_shift_left(simd_type a, simd_type v) requires std::integral<simd_type> { return a << v; }
        KAIXO_INLINE static simd_type bit_shift_right(simd_type a, simd_type v) requires std::integral<simd_type> { return a << v; }

        // ------------------------------------------------

        KAIXO_INLINE static bool eq(simd_type a, simd_type b) { return a == b; }
        KAIXO_INLINE static bool neq(simd_type a, simd_type b) { return a != b; }
        KAIXO_INLINE static bool gt(simd_type a, simd_type b) { return a > b; }
        KAIXO_INLINE static bool gteq(simd_type a, simd_type b) { return a >= b; }
        KAIXO_INLINE static bool lt(simd_type a, simd_type b) { return a < b; }
        KAIXO_INLINE static bool lteq(simd_type a, simd_type b) { return a <= b; }

        KAIXO_INLINE static bool lt0(simd_type a) { return a < 0; }

        // ------------------------------------------------
        
        KAIXO_INLINE static simd_type min(simd_type a, simd_type b) { return a < b ? a : b; }
        KAIXO_INLINE static simd_type max(simd_type a, simd_type b) { return a < b ? b : a; }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type sum(simd_type a) { return a; }
        
        // ------------------------------------------------

        KAIXO_INLINE static simd_type reverse(simd_type a) { return a; }
        
        // ------------------------------------------------

        KAIXO_INLINE static simd_type trunc(simd_type a) requires std::floating_point<simd_type> { return static_cast<simd_type>(static_cast<int_alias>(a)); }
        KAIXO_INLINE static simd_type floor(simd_type a) requires std::floating_point<simd_type> { return static_cast<simd_type>(static_cast<int_alias>(a)) - (static_cast<int_alias>(a) > a); }
        KAIXO_INLINE static simd_type ceil(simd_type a) requires std::floating_point<simd_type> { return static_cast<simd_type>(static_cast<int_alias>(a)) + (static_cast<int_alias>(a) > a); }
        KAIXO_INLINE static simd_type round(simd_type a) requires std::floating_point<simd_type> { return std::round(a); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type sign(simd_type a) requires std::floating_point<simd_type> { return a < 0 ? -1 : 1; }
        KAIXO_INLINE static simd_type copysign(simd_type from, simd_type to) requires std::floating_point<simd_type> { return sign(from) < sign(to) ? -to : to; }
        KAIXO_INLINE static simd_type xorsign(simd_type from, simd_type to) requires std::floating_point<simd_type> { return sign(from) != sign(to) ? -to : to; }
        KAIXO_INLINE static simd_type abs(simd_type a) requires std::floating_point<simd_type> { return a < 0 ? -a : a; }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type rcp(simd_type a) requires std::floating_point<simd_type> { return 1 / a; }
        KAIXO_INLINE static simd_type log(simd_type a) requires std::floating_point<simd_type> { return std::log(a); }
        KAIXO_INLINE static simd_type log2(simd_type a) requires std::floating_point<simd_type> { return std::log2(a); }
        KAIXO_INLINE static simd_type log10(simd_type a) requires std::floating_point<simd_type> { return std::log10(a); }
        KAIXO_INLINE static simd_type sqrt(simd_type a) requires std::floating_point<simd_type> { return std::sqrt(a); }
        KAIXO_INLINE static simd_type cbrt(simd_type a) requires std::floating_point<simd_type> { return std::cbrt(a); }
        KAIXO_INLINE static simd_type exp(simd_type a) requires std::floating_point<simd_type> { return std::exp(a); }
        KAIXO_INLINE static simd_type exp2(simd_type a) requires std::floating_point<simd_type> { return std::exp2(a); }
        KAIXO_INLINE static simd_type exp10(simd_type a) requires std::floating_point<simd_type> { return std::pow(10, a); }
        KAIXO_INLINE static simd_type tanh(simd_type a) requires std::floating_point<simd_type> { return std::tanh(a); }
        KAIXO_INLINE static simd_type cos(simd_type a) requires std::floating_point<simd_type> { return std::cos(a); }
        KAIXO_INLINE static simd_type cosh(simd_type a) requires std::floating_point<simd_type> { return std::cosh(a); }
        KAIXO_INLINE static simd_type sin(simd_type a) requires std::floating_point<simd_type> { return std::sin(a); }
        KAIXO_INLINE static simd_type sinh(simd_type a) requires std::floating_point<simd_type> { return std::sinh(a); }
        KAIXO_INLINE static simd_type pow(simd_type a, simd_type b) requires std::floating_point<simd_type> { return std::pow(a, b); }
        KAIXO_INLINE static std::pair<simd_type, simd_type> sincos(simd_type a) requires std::floating_point<simd_type> { return std::make_pair(std::sin(a), std::cos(a)); }
        
        // ------------------------------------------------

        KAIXO_INLINE static simd_type fast_nsin(simd_type a) requires std::floating_point<simd_type> {
            auto inter = a * (-16 * abs(a) + 8);
            return inter * (static_cast<simd_type>(0.224) * abs(inter) + static_cast<simd_type>(0.776));
        }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type blend(mask_type mask, simd_type a, simd_type b) { return mask ? a : b; }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type noise() {
            thread_local struct Random {
                uint64_t state[2]{ 0x84242f96eca9c41d, 0xa3c65b8776f96855 };
                Random() { state[0] *= rand(); state[1] *= rand(); }
                constexpr uint64_t rotl(const uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
            } random{};
            const uint64_t s0 = random.state[0];
            uint64_t s1 = random.state[1];
            const uint64_t result = s0 + s1;
            s1 ^= s0;
            random.state[0] = random.rotl(s0, 24) ^ s1 ^ (s1 << 16);
            random.state[1] = random.rotl(s1, 37);
            return static_cast<float>(static_cast<long double>(result) / std::numeric_limits<std::uint64_t>::max());
        }

        KAIXO_INLINE static simd_type noise(int& part1, int& part2) {
            std::uint32_t s1 = std::bit_cast<std::uint32_t>(part1);
            const std::uint32_t s0 = std::bit_cast<std::uint32_t>(part2);
            part1 = part2;
            s1 = std::bit_cast<std::uint32_t>(part2) ^ (std::bit_cast<std::uint32_t>(part2) << 23);
            part2 = std::bit_cast<int>(((s1 ^ s0) ^ (s1 >> 18)) ^ (s0 >> 5));
            return static_cast<float>(std::bit_cast<std::uint32_t>(part2) + s0) / static_cast<float>(std::numeric_limits<std::int32_t>::max());
        }

        // ------------------------------------------------

    };

    // ------------------------------------------------

    template<class T, std::size_t Bits>
    struct simd_find_abi : std::type_identity<simd_scalar_fallback_abi<T>> {};

    template<class T> struct simd_find_abi<T, 512> : simd_find_abi<T, 256> {};
    template<class T> struct simd_find_abi<T, 256> : simd_find_abi<T, 128> {};
    template<class T> struct simd_find_abi<T, 128> : simd_find_abi<T, 0> {};

    // ------------------------------------------------

#if KAIXO_SIMD_ARM

    // ------------------------------------------------

    struct simd_float_128_neon {

        // ------------------------------------------------

        using value_type = float;
        using mask_type = float32x4_t;
        using simd_type = float32x4_t;
        using buffer_type = value_type*;
        using const_buffer_type = const value_type*;
        using int_alias = uint32x4_t;

        // ------------------------------------------------

        constexpr static std::size_t elements  = sizeof(simd_type) / sizeof(value_type);
        constexpr static std::size_t bytes     = sizeof(simd_type);
        constexpr static std::size_t alignment = alignof(simd_type);

        // ------------------------------------------------
        
        template<float(*fun)(float)> 
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL scalar_fallback(simd_type a) {
            float vals[elements];
            vst1q_f32(vals, a);
            for (int i = 0; i < 4; i++) vals[i] = fun(vals[i]);
            return vld1q_f32(vals);
		}

        template<float(*fun)(float, float)> 
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL scalar_fallback(simd_type a, simd_type b) {
            float valsa[elements];
            float valsb[elements];
            vst1q_f32(valsa, a);
            vst1q_f32(valsb, b);
            for (int i = 0; i < 4; i++) valsa[i] = fun(valsa[i], valsb[b]);
            return vld1q_f32(valsa);
		}

        // ------------------------------------------------

        KAIXO_INLINE static value_type KAIXO_VECTORCALL index(simd_type a, std::size_t i) { return vgetq_lane_f32(a, i); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL setzero() { return vdupq_n_f32(0.f); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL set1(value_type val) { return vdupq_n_f32(val); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL setincr() {
            constexpr float vals[4] = { 0.f,1.f,2.f,3.f };
            return vld1q_f32(vals);
        }

        KAIXO_INLINE static mask_type KAIXO_VECTORCALL true_mask() { return vdupq_n_f32(std::bit_cast<value_type>(0xFFFFFFFF)); }
        KAIXO_INLINE static mask_type KAIXO_VECTORCALL false_mask() { return vdupq_n_f32(0.f); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL load(const_buffer_type data) { return vld1q_f32(data); };
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL loadu(const_buffer_type data) { return vld1q_f32(data); };
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL loadr(const_buffer_type data) {
            float vals[4] = { data[3], data[2], data[1], data[0] };
            return vld1q_f32(vals);
        };

        KAIXO_INLINE static void KAIXO_VECTORCALL store(buffer_type to, simd_type from) { vst1q_f32(to, from); };
        KAIXO_INLINE static void KAIXO_VECTORCALL storeu(buffer_type to, simd_type from) { vst1q_f32(to, from); };
        KAIXO_INLINE static void KAIXO_VECTORCALL stream(buffer_type to, simd_type from) { vst1q_f32(to, from); };

        // ------------------------------------------------

        KAIXO_INLINE static int_alias KAIXO_VECTORCALL to_int(simd_type a) { return vcvtq_u32_f32(vrndzq_f32(a)); }
        KAIXO_INLINE static int_alias KAIXO_VECTORCALL as_int(simd_type a) { return vreinterpretq_u32_f32(a); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL add(simd_type a, simd_type b) { return vaddq_f32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL sub(simd_type a, simd_type b) { return vsubq_f32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL mul(simd_type a, simd_type b) { return vmulq_f32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL div(simd_type a, simd_type b) { return vdivq_f32(a, b); }

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL negate(simd_type a) { return vnegq_f32(a); }

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL fmadd(simd_type a, simd_type b, simd_type c) { return vfmaq_f32(a, b, c); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL fmsub(simd_type a, simd_type b, simd_type c) { return vfmsq_f32(a, b, c); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL bit_and(simd_type a, simd_type b) { return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b))); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL bit_or(simd_type a, simd_type b) { return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b))); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL bit_xor(simd_type a, simd_type b) { return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b))); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL bit_not(simd_type a) { return vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(a))); }
        
        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL eq(simd_type a, simd_type b) { return vceqq_f32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL neq(simd_type a, simd_type b) { return vmvnq_u32(vceqq_f32(a, b)); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL gt(simd_type a, simd_type b) { return vcgtq_f32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL gteq(simd_type a, simd_type b) { return vcgeq_f32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL lt(simd_type a, simd_type b) { return vcltq_f32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL lteq(simd_type a, simd_type b) { return vcleq_f32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL lt0(simd_type a) { return vcltq_f32(a, vdupq_n_f32(0.f)); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL min(simd_type a, simd_type b) { return vminq_f32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL max(simd_type a, simd_type b) { return vmaxq_f32(a, b); }

        // ------------------------------------------------

        KAIXO_INLINE static float KAIXO_VECTORCALL sum(simd_type a) {
            float32x2_t t = vadd_f32(vget_low_f32(a), vget_high_f32(a));
            t = vpadd_f32(t, t);
            return vget_lane_f32(t, 0);
        }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL reverse(simd_type a) {
            return vrev64q_f32(vcombine_f32(vget_high_f32(a), vget_low_f32(a)));
        }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL trunc(simd_type a) { return vrndq_f32(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL floor(simd_type a) { return vrndmq_f32(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL ceil(simd_type a) { return vrndpq_f32(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL round(simd_type a) { return vrndnq_f32(a); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL sign(simd_type a) { return copysign(a, set1(1)); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL copysign(simd_type from, simd_type to) { return vreinterpretq_f32_u32(vorrq_u32(vandq_u32(vreinterpretq_u32_f32(to), vdupq_n_u32(0xFFFFFFFF - 0x80000000)), vandq_u32(vreinterpretq_u32_f32(from), vdupq_n_u32(0x80000000)))); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL xorsign(simd_type from, simd_type to) { return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(to), vandq_u32(vreinterpretq_u32_f32(from), vdupq_n_u32(0x80000000)))); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL orsign(simd_type from, simd_type to) { return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(to), vandq_u32(vreinterpretq_u32_f32(from), vdupq_n_u32(0x80000000)))); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL abs(simd_type a) { return vabsq_f32(a); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL rcp(simd_type a) { return vrecpeq_f32(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL log(simd_type a) { return scalar_fallback<&std::log>(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL log2(simd_type a) { return scalar_fallback<&std::log2>(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL log10(simd_type a) { return scalar_fallback<&std::log10>(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL sqrt(simd_type a) { return vsqrtq_f32(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL cbrt(simd_type a) { return scalar_fallback<&std::cbrt>(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL exp(simd_type a) { return scalar_fallback<&std::exp>(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL exp2(simd_type a) { return scalar_fallback<&std::exp2>(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL tanh(simd_type a) { return scalar_fallback<&std::tanh>(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL cos(simd_type a) { return scalar_fallback<&std::cos>(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL cosh(simd_type a) { return scalar_fallback<&std::cosh>(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL sin(simd_type a) { return scalar_fallback<&std::sin>(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL sinh(simd_type a) { return scalar_fallback<&std::sinh>(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL pow(simd_type a, simd_type b) { return scalar_fallback<&std::pow>(a, b); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL fast_nsin(simd_type a) {
            const simd_type c1 = vdupq_n_f32(-16.0f);
            const simd_type c2 = vdupq_n_f32(8.0f);
            const simd_type c3 = vdupq_n_f32(0.224f);
            const simd_type c4 = vdupq_n_f32(0.776f);

            // (-16 * abs(a)) + 8
            simd_type inter1 = vfmaq_f32(c2, c1, vabsq_f32(a));

            // a * inter1
            simd_type inter2 = vmulq_f32(inter1, a);

            // (0.224 * abs(inter2)) + 0.776
            simd_type inter3 = vfmaq_f32(c4, c3, vabsq_f32(inter2));

            // inter2 * inter3
            return vmulq_f32(inter3, inter2);
        }

        // ------------------------------------------------
        
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL blend(mask_type mask, simd_type a, simd_type b) {
            return vbslq_s32(mask, a, b);
        }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL noise() { 
            thread_local struct {
                uint64x2_t  part1{ 0xdb899e0994c4b301, 0x3f5abe6af2efd66e };
                uint64x2_t  part2{ 0x891627810c57c0dc, 0x8793773053862b5f };
            } state;

            uint64x2_t s1 = state.part1;
            const uint64x2_t s0 = state.part2;
            state.part1 = state.part2;
            s1 = veorq_u64(state.part2, vshlq_n_u64(state.part2, 23));
            state.part2 = veorq_u64(
                veorq_u64(veorq_u64(s1, s0),
                    vshrq_n_u64(s1, 18)), vshrq_n_u64(s0, 5));

            return vdivq_f32(vcvtq_f32_u32(vaddq_u64(state.part2, s0)),
                vdupq_n_f32(static_cast<float>(std::numeric_limits<std::int32_t>::max())));
        }

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL noise(uint64x2_t& part1, uint64x2_t& part2) {
            uint64x2_t s1 = part1;
            const uint64x2_t s0 = part2;
            part1 = part2;
            s1 = veorq_u64(part2, vshlq_n_u64(part2, 23));
            part2 = veorq_u64(
                veorq_u64(veorq_u64(s1, s0),
                    vshrq_n_u64(s1, 18)), vshrq_n_u64(s0, 5));

            return vdivq_f32(vcvtq_f32_u32(vaddq_u64(part2, s0)),
                vdupq_n_f32(static_cast<float>(std::numeric_limits<std::int32_t>::max())));
        }

        // ------------------------------------------------

    };
    
    struct simd_int_128_neon {
       
       // ------------------------------------------------

        using value_type = int;
        using mask_type = int32x4_t;
        using simd_type = int32x4_t;
        using buffer_type = value_type*;
        using const_buffer_type = const value_type*;

        // ------------------------------------------------

        constexpr static std::size_t elements  = sizeof(simd_type) / sizeof(value_type);
        constexpr static std::size_t bytes     = sizeof(simd_type);
        constexpr static std::size_t alignment = alignof(simd_type);

        // ------------------------------------------------
        
        template<int(*fun)(int)>
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL scalar_fallback(simd_type a) {
            int vals[elements];
            vst1q_s32(vals, a);
            for (int i = 0; i < 4; i++) vals[i] = fun(vals[i]);
            return vld1q_s32(vals);
		}

        template<int(*fun)(int, int)>
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL scalar_fallback(simd_type a, simd_type b) {
            int valsa[elements];
            int valsb[elements];
            vst1q_s32(valsa, a);
            vst1q_s32(valsb, b);
            for (int i = 0; i < 4; i++) valsa[i] = fun(valsa[i], valsb[b]);
            return vld1q_s32(valsa);
		}

        // ------------------------------------------------

        KAIXO_INLINE static value_type KAIXO_VECTORCALL index(simd_type a, std::size_t i) { return vgetq_lane_s32(a, i); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL setzero() { return vdupq_n_s32(0.f); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL set1(value_type val) { return vdupq_n_s32(val); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL setincr() {
            constexpr int vals[4] = { 0, 1, 2, 3 };
            return vld1q_s32(vals);
        }

        KAIXO_INLINE static mask_type KAIXO_VECTORCALL true_mask() { return vdupq_n_s32(std::bit_cast<value_type>(0xFFFFFFFF)); }
        KAIXO_INLINE static mask_type KAIXO_VECTORCALL false_mask() { return vdupq_n_s32(0); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL load(const_buffer_type data) { return vld1q_s32(data); };
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL loadu(const_buffer_type data) { return vld1q_s32(data); };
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL loadr(const_buffer_type data) {
            int vals[4] = { data[3], data[2], data[1], data[0] };
            return vld1q_s32(vals);
        };

        KAIXO_INLINE static void KAIXO_VECTORCALL store(buffer_type to, simd_type from) { vst1q_s32(to, from); };
        KAIXO_INLINE static void KAIXO_VECTORCALL storeu(buffer_type to, simd_type from) { vst1q_s32(to, from); };
        KAIXO_INLINE static void KAIXO_VECTORCALL stream(buffer_type to, simd_type from) { vst1q_s32(to, from); };

        KAIXO_INLINE static float32x4_t KAIXO_VECTORCALL gather(const float* data, simd_type index) {
            float vals[4] = {
                data[vgetq_lane_s32(index, 0)],
                data[vgetq_lane_s32(index, 1)],
                data[vgetq_lane_s32(index, 2)],
                data[vgetq_lane_s32(index, 3)],
            };

            return vld1q_f32(vals);
        };

        KAIXO_INLINE static int32x4_t KAIXO_VECTORCALL gather(const int* data, simd_type index) {
            int vals[4] = {
                data[vgetq_lane_s32(index, 0)],
                data[vgetq_lane_s32(index, 1)],
                data[vgetq_lane_s32(index, 2)],
                data[vgetq_lane_s32(index, 3)],
            };

            return vld1q_s32(vals);
        };

        // ------------------------------------------------

        KAIXO_INLINE static float32x4_t KAIXO_VECTORCALL to_float(simd_type a) { return vcvtq_f32_s32(a); }
        KAIXO_INLINE static float32x4_t KAIXO_VECTORCALL as_float(simd_type a) { return vreinterpretq_f32_s32(a); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL add(simd_type a, simd_type b) { return vaddq_s32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL sub(simd_type a, simd_type b) { return vsubq_s32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL mul(simd_type a, simd_type b) { return vmulq_s32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL div(simd_type a, simd_type b) { return scalar_fallback<[](int a, int b){ return a / b; }>(a, b); }

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL negate(simd_type a) { return vnegq_s32(a); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL bit_and(simd_type a, simd_type b) { return vandq_u32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL bit_or(simd_type a, simd_type b) { return vorrq_u32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL bit_xor(simd_type a, simd_type b) { return veorq_u32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL bit_not(simd_type a) { return vmvnq_u32(a); }
        
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL bit_shift_left(simd_type a, int v) { return vshlq_n_s32(a, v); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL bit_shift_right(simd_type a, int v) { return vshrq_n_s32(a, v); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL bit_shift_left(simd_type a, simd_type v) { return vshlq_s32(a, v); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL bit_shift_right(simd_type a, simd_type v) { return vshlq_s32(a, vnegq_s32(v)); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL eq(simd_type a, simd_type b) { return vceqq_s32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL neq(simd_type a, simd_type b) { return bit_not(vceqq_s32(a, b)); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL gt(simd_type a, simd_type b) { return vcgtq_s32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL gteq(simd_type a, simd_type b) { return vcgeq_s32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL lt(simd_type a, simd_type b) { return vcltq_s32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL lteq(simd_type a, simd_type b) { return vcleq_s32(a, b); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL min(simd_type a, simd_type b) { return vminq_s32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL max(simd_type a, simd_type b) { return vmaxq_s32(a, b); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL reverse(simd_type a) {
            return vrev64q_s32(vcombine_s32(vget_high_s32(a), vget_low_s32(a)));
        }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL blend(mask_type mask, simd_type a, simd_type b) {
            return vbslq_s32(mask, a, b);
        }

        // ------------------------------------------------

    };

    // ------------------------------------------------

    template<> struct simd_find_abi<float, 128> : std::type_identity<simd_float_128_neon> {};
    template<> struct simd_find_abi<int, 128> : std::type_identity<simd_int_128_neon> {};

    // ------------------------------------------------

#elif KAIXO_SIMD_X86_64

    // ------------------------------------------------

    struct simd_float_256_x86_64 {

        // ------------------------------------------------

        using value_type = float;
        using mask_type = __m256;
        using simd_type = __m256;
        using buffer_type = value_type*;
        using const_buffer_type = const value_type*;

        // ------------------------------------------------

        constexpr static std::size_t elements  = sizeof(simd_type) / sizeof(value_type);
        constexpr static std::size_t bytes     = sizeof(simd_type);
        constexpr static std::size_t alignment = alignof(simd_type);

        // ------------------------------------------------

        KAIXO_INLINE static value_type KAIXO_VECTORCALL index(simd_type a, std::size_t i) { return a.m256_f32[i]; }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL setzero() { return _mm256_setzero_ps(); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL setincr() { return _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL set1(value_type val) { return _mm256_set1_ps(val); }

        KAIXO_INLINE static mask_type KAIXO_VECTORCALL true_mask() { return _mm256_set1_ps(std::bit_cast<value_type>(0xFFFFFFFF)); }
        KAIXO_INLINE static mask_type KAIXO_VECTORCALL false_mask() { return _mm256_setzero_ps(); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL load(const_buffer_type data) { return _mm256_load_ps(data); };
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL loadr(const_buffer_type data) { return _mm256_set_ps(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]); };
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL loadu(const_buffer_type data) { return _mm256_loadu_ps(data); };

        KAIXO_INLINE static void KAIXO_VECTORCALL store(buffer_type to, simd_type from) { _mm256_store_ps(to, from); };
        KAIXO_INLINE static void KAIXO_VECTORCALL storeu(buffer_type to, simd_type from) { _mm256_storeu_ps(to, from); };
        KAIXO_INLINE static void KAIXO_VECTORCALL stream(buffer_type to, simd_type from) { _mm256_stream_ps(to, from); };

        // ------------------------------------------------

        KAIXO_INLINE static __m256i KAIXO_VECTORCALL to_int(simd_type a) { return _mm256_cvttps_epi32(a); }
        KAIXO_INLINE static __m256i KAIXO_VECTORCALL as_int(simd_type a) { return _mm256_castps_si256(a); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL add(simd_type a, simd_type b) { return _mm256_add_ps(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL sub(simd_type a, simd_type b) { return _mm256_sub_ps(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL mul(simd_type a, simd_type b) { return _mm256_mul_ps(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL div(simd_type a, simd_type b) { return _mm256_div_ps(a, b); }

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL negate(simd_type a) { return _mm256_xor_ps(a, _mm256_set1_ps(-0.f)); }

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL fmadd(simd_type a, simd_type b, simd_type c) { return _mm256_fmadd_ps(a, b, c); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL fmsub(simd_type a, simd_type b, simd_type c) { return _mm256_fmsub_ps(a, b, c); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL bit_and(simd_type a, simd_type b) { return _mm256_and_ps(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL bit_or(simd_type a, simd_type b) { return _mm256_or_ps(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL bit_xor(simd_type a, simd_type b) { return _mm256_xor_ps(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL bit_not(simd_type a) { return _mm256_xor_ps(a, _mm256_set1_ps(std::bit_cast<float>(0xFFFFFFFF))); }
        
        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL eq(simd_type a, simd_type b) { return _mm256_cmp_ps(a, b, _CMP_EQ_OQ); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL neq(simd_type a, simd_type b) { return _mm256_cmp_ps(a, b, _CMP_NEQ_OQ); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL gt(simd_type a, simd_type b) { return _mm256_cmp_ps(a, b, _CMP_GT_OQ); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL gteq(simd_type a, simd_type b) { return _mm256_cmp_ps(a, b, _CMP_GE_OQ); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL lt(simd_type a, simd_type b) { return _mm256_cmp_ps(a, b, _CMP_LT_OQ); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL lteq(simd_type a, simd_type b) { return _mm256_cmp_ps(a, b, _CMP_LE_OQ); }

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL lt0(simd_type a) { return _mm256_and_ps(a, _mm256_set1_ps(-0.0f)); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL min(simd_type a, simd_type b) { return _mm256_min_ps(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL max(simd_type a, simd_type b) { return _mm256_max_ps(a, b); }

        // ------------------------------------------------

        KAIXO_INLINE static float KAIXO_VECTORCALL sum(simd_type a) { 
            __m128 hiQuad = _mm256_extractf128_ps(a, 1);
            __m128 loQuad = _mm256_castps256_ps128(a);
            __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
            __m128 loDual = sumQuad;
            __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
            __m128 sumDual = _mm_add_ps(loDual, hiDual);
            __m128 lo = sumDual;
            __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
            __m128 sum = _mm_add_ss(lo, hi);
            return _mm_cvtss_f32(sum);
        }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL reverse(simd_type a) {
            const auto v = _mm256_permute2f128_ps(a, a, 0x01);
            return _mm256_shuffle_ps(v, v, _MM_SHUFFLE(0, 1, 2, 3));
        }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL trunc(simd_type a) { return _mm256_trunc_ps(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL floor(simd_type a) { return _mm256_round_ps(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL ceil(simd_type a) { return _mm256_round_ps(a, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL round(simd_type a) { return _mm256_round_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL sign(simd_type a) { return copysign(a, set1(1)); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL copysign(simd_type from, simd_type to) { return _mm256_or_ps(_mm256_and_ps(to, _mm256_set1_ps(std::bit_cast<float>(0xFFFFFFFF - 0x80000000))), _mm256_and_ps(from, _mm256_set1_ps(-0.f))); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL xorsign(simd_type from, simd_type to) { return _mm256_xor_ps(to, _mm256_and_ps(from, _mm256_set1_ps(-0.f))); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL orsign(simd_type from, simd_type to) { return _mm256_or_ps(to, _mm256_and_ps(from, _mm256_set1_ps(-0.f))); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL abs(simd_type a) { return _mm256_andnot_ps(_mm256_set1_ps(-0.0), a); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL rcp(simd_type a) { return _mm256_rcp_ps(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL log(simd_type a) { return _mm256_log_ps(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL log2(simd_type a) { return _mm256_log2_ps(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL log10(simd_type a) { return _mm256_log10_ps(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL sqrt(simd_type a) { return _mm256_sqrt_ps(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL cbrt(simd_type a) { return _mm256_cbrt_ps(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL exp(simd_type a) { return _mm256_exp_ps(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL exp2(simd_type a) { return _mm256_exp2_ps(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL tanh(simd_type a) { return _mm256_tanh_ps(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL cos(simd_type a) { return _mm256_cos_ps(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL cosh(simd_type a) { return _mm256_cosh_ps(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL sin(simd_type a) { return _mm256_sin_ps(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL sinh(simd_type a) { return _mm256_sinh_ps(a); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL pow(simd_type a, simd_type b) { return _mm256_pow_ps(a, b); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL fast_nsin(simd_type a) {
            // value * ((-16.f * abs(value)) + 8.f)
            auto _inter1 = _mm256_fmadd_ps(_mm256_set1_ps(-16.f), _mm256_andnot_ps(_mm256_set1_ps(-0.0), a), _mm256_set1_ps(8.f));
            auto _inter2 = _mm256_mul_ps(_inter1, a);
            // _inter2 * ((0.224f * abs(_inter2)) + 0.776f)
            auto _inter3 = _mm256_fmadd_ps(_mm256_set1_ps(0.224f), _mm256_andnot_ps(_mm256_set1_ps(-0.0), _inter2), _mm256_set1_ps(0.776f));
            return _mm256_mul_ps(_inter3, _inter2);
        }

        // ------------------------------------------------
        
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL blend(mask_type mask, simd_type a, simd_type b) { return _mm256_blendv_ps(b, a, mask); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL noise() { 
            thread_local struct {
                __m256i part1{ .m256i_u64 { 0xdb899e0994c4b301, 0x3f5abe6af2efd66e, 0x225316feba2bd4eb, 0xbe5ba0327ff5a462, } };
                __m256i part2{ .m256i_u64 { 0x891627810c57c0dc, 0x8793773053862b5f, 0x6e041e1b9b54605a, 0x19d9edbb34011806, } };
            } state;

            __m256i s1 = state.part1;
            const __m256i s0 = state.part2;
            state.part1 = state.part2;
            s1 = _mm256_xor_si256(state.part2, _mm256_slli_epi64(state.part2, 23));
            state.part2 = _mm256_xor_si256(
                _mm256_xor_si256(_mm256_xor_si256(s1, s0),
                    _mm256_srli_epi64(s1, 18)), _mm256_srli_epi64(s0, 5));
            return _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_add_epi64(state.part2, s0)),
                _mm256_set1_ps(static_cast<float>(std::numeric_limits<std::int32_t>::max())));
        }

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL noise(__m256i& part1, __m256i& part2) {
            __m256i s1 = part1;
            const __m256i s0 = part2;
            part1 = part2;
            s1 = _mm256_xor_si256(part2, _mm256_slli_epi64(part2, 23));
            part2 = _mm256_xor_si256(
                _mm256_xor_si256(_mm256_xor_si256(s1, s0),
                    _mm256_srli_epi64(s1, 18)), _mm256_srli_epi64(s0, 5));
            return _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_add_epi64(part2, s0)),
                _mm256_set1_ps(static_cast<float>(std::numeric_limits<std::int32_t>::max())));
        }

        // ------------------------------------------------

    };

    // ------------------------------------------------

    struct simd_int_256_x86_64 {
       
       // ------------------------------------------------

        using value_type = int;
        using mask_type = __m256i;
        using simd_type = __m256i;
        using buffer_type = value_type*;
        using const_buffer_type = const value_type*;

        // ------------------------------------------------

        constexpr static std::size_t elements  = sizeof(simd_type) / sizeof(value_type);
        constexpr static std::size_t bytes     = sizeof(simd_type);
        constexpr static std::size_t alignment = alignof(simd_type);

        // ------------------------------------------------

        KAIXO_INLINE static value_type KAIXO_VECTORCALL index(simd_type a, std::size_t i) { return a.m256i_i32[i]; }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL setzero() { return _mm256_setzero_si256(); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL setincr() { return _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL set1(value_type val) { return _mm256_set1_epi32(val); }

        KAIXO_INLINE static mask_type KAIXO_VECTORCALL true_mask() { return _mm256_set1_epi32(std::bit_cast<value_type>(0xFFFFFFFF)); }
        KAIXO_INLINE static mask_type KAIXO_VECTORCALL false_mask() { return _mm256_setzero_si256(); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL load(const_buffer_type data) { return _mm256_load_si256(reinterpret_cast<__m256i const*>(data)); };
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL loadr(const_buffer_type data) { return _mm256_set_epi32(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]); };
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL loadu(const_buffer_type data) { return _mm256_loadu_si256(reinterpret_cast<__m256i const*>(data)); };

        KAIXO_INLINE static void KAIXO_VECTORCALL store(buffer_type to, simd_type from) { _mm256_store_si256(reinterpret_cast<__m256i*>(to), from); };
        KAIXO_INLINE static void KAIXO_VECTORCALL storeu(buffer_type to, simd_type from) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(to), from); };
        KAIXO_INLINE static void KAIXO_VECTORCALL stream(buffer_type to, simd_type from) { _mm256_stream_si256(reinterpret_cast<__m256i*>(to), from); };

        KAIXO_INLINE static __m256 KAIXO_VECTORCALL gather(const float* data, simd_type index) { return _mm256_i32gather_ps(data, index, sizeof(float)); };
        KAIXO_INLINE static __m256i KAIXO_VECTORCALL gather(const int* data, simd_type index) { return _mm256_i32gather_epi32(data, index, sizeof(int)); };

        // ------------------------------------------------

        KAIXO_INLINE static __m256 KAIXO_VECTORCALL to_float(simd_type a) { return _mm256_cvtepi32_ps(a); }
        KAIXO_INLINE static __m256 KAIXO_VECTORCALL as_float(simd_type a) { return _mm256_castsi256_ps(a); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL add(simd_type a, simd_type b) { return _mm256_add_epi32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL sub(simd_type a, simd_type b) { return _mm256_sub_epi32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL mul(simd_type a, simd_type b) { return _mm256_mullo_epi32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL div(simd_type a, simd_type b) { return _mm256_div_epi32(a, b); }

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL negate(simd_type a) { return sub(set1(0), a); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL bit_and(simd_type a, simd_type b) { return _mm256_and_epi32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL bit_or(simd_type a, simd_type b) { return _mm256_or_epi32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL bit_xor(simd_type a, simd_type b) { return _mm256_xor_epi32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL bit_not(simd_type a) { return _mm256_xor_si256(a, _mm256_set1_epi32(std::bit_cast<int>(0xFFFFFFFF))); }
        
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL bit_shift_left(simd_type a, int v) { return _mm256_slli_epi32(a, v); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL bit_shift_right(simd_type a, int v) { return _mm256_srli_epi32(a, v); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL bit_shift_left(simd_type a, simd_type v) { return _mm256_sllv_epi32(a, v); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL bit_shift_right(simd_type a, simd_type v) { return _mm256_srlv_epi32(a, v); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL eq(simd_type a, simd_type b) { return _mm256_cmpeq_epi32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL neq(simd_type a, simd_type b) { return bit_not(_mm256_cmpeq_epi32(a, b)); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL gt(simd_type a, simd_type b) { return _mm256_cmpgt_epi32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL gteq(simd_type a, simd_type b) { return bit_or(gt(a, b), eq(a, b)); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL lt(simd_type a, simd_type b) { return bit_not(gteq(a, b)); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL lteq(simd_type a, simd_type b) { return bit_not(gt(a, b)); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL min(simd_type a, simd_type b) { return _mm256_min_epi32(a, b); }
        KAIXO_INLINE static simd_type KAIXO_VECTORCALL max(simd_type a, simd_type b) { return _mm256_max_epi32(a, b); }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL reverse(simd_type a) {
            const auto v = _mm256_permute2f128_si256(a, a, 0x01);
            return _mm256_shuffle_epi32(v, _MM_SHUFFLE(0, 1, 2, 3));
        }

        // ------------------------------------------------

        KAIXO_INLINE static simd_type KAIXO_VECTORCALL blend(mask_type mask, simd_type a, simd_type b) { return _mm256_blendv_epi8(b, a, mask); }

        // ------------------------------------------------

    };

    // ------------------------------------------------

    template<> struct simd_find_abi<float, 256> : std::type_identity<simd_float_256_x86_64> {};
    template<> struct simd_find_abi<int, 256> : std::type_identity<simd_int_256_x86_64> {};

    // ------------------------------------------------

#endif

    // ------------------------------------------------

#define KAIXO_FROM_ABI(x) requires requires { Abi::x; } { return Abi::x; }

    // ------------------------------------------------

    template<class T, class Abi>
    struct basic_simd_mask {

        // ------------------------------------------------

        using mask_type = typename Abi::mask_type;
        using value_type = typename Abi::value_type;
        using buffer_type = typename Abi::buffer_type;
        using const_buffer_type = typename Abi::const_buffer_type;

        // ------------------------------------------------

        constexpr static std::size_t elements = Abi::elements;
        constexpr static std::size_t bits = Abi::bytes * 8;
        constexpr static std::size_t bytes = Abi::bytes;
        constexpr static std::size_t bytes_per_element = bytes / elements;
        constexpr static std::size_t alignment = Abi::alignment;

        // ------------------------------------------------

        mask_type value{};

        // ------------------------------------------------

        KAIXO_INLINE basic_simd_mask() : value() {}
        KAIXO_INLINE basic_simd_mask(mask_type val) : value(val) {}

        // ------------------------------------------------

        KAIXO_INLINE value_type operator[](std::size_t i) const KAIXO_FROM_ABI(index(value, i));

        // ------------------------------------------------

        KAIXO_INLINE static mask_type true_mask() KAIXO_FROM_ABI(true_mask());
        KAIXO_INLINE static mask_type false_mask() KAIXO_FROM_ABI(false_mask());

        // ------------------------------------------------

        KAIXO_INLINE friend basic_simd_mask operator&&(const basic_simd_mask& a, const basic_simd_mask& b) KAIXO_FROM_ABI(bit_and(a.value, b.value));
        KAIXO_INLINE friend basic_simd_mask operator||(const basic_simd_mask& a, const basic_simd_mask& b) KAIXO_FROM_ABI(bit_or(a.value, b.value));
        KAIXO_INLINE friend basic_simd_mask operator~(const basic_simd_mask& a) KAIXO_FROM_ABI(bit_not(a.value));
        KAIXO_INLINE friend basic_simd_mask operator!(const basic_simd_mask& a) KAIXO_FROM_ABI(bit_not(a.value));

        // ------------------------------------------------

    };

    // ------------------------------------------------

    template<class T, class Abi>
    struct basic_simd {

        // ------------------------------------------------

        using simd_type = typename Abi::simd_type;
        using value_type = typename Abi::value_type;
        using buffer_type = typename Abi::buffer_type;
        using const_buffer_type = typename Abi::const_buffer_type;
        using mask_type = basic_simd_mask<T, Abi>;

        // ------------------------------------------------

        constexpr static std::size_t elements = Abi::elements;
        constexpr static std::size_t bits = Abi::bytes * 8;
        constexpr static std::size_t bytes = Abi::bytes;
        constexpr static std::size_t bytes_per_element = bytes / elements;
        constexpr static std::size_t alignment = Abi::alignment;

        // ------------------------------------------------

        simd_type value{};

        // ------------------------------------------------

        KAIXO_INLINE basic_simd() : value() {}
        KAIXO_INLINE basic_simd(simd_type val) : value(val) {}

        template<class Ty> requires (std::convertible_to<Ty, value_type> && !std::same_as<Ty, simd_type>)
        KAIXO_INLINE basic_simd(Ty val) : value(Abi::set1(static_cast<value_type>(val))) {}

        // ------------------------------------------------

        KAIXO_INLINE operator simd_type() const { return value; }

        // ------------------------------------------------

        KAIXO_INLINE value_type operator[](std::size_t i) const KAIXO_FROM_ABI(index(value, i));

        // ------------------------------------------------

        KAIXO_INLINE static basic_simd setzero() KAIXO_FROM_ABI(setzero());
        KAIXO_INLINE static basic_simd setincr() KAIXO_FROM_ABI(setincr());
        KAIXO_INLINE static basic_simd set1(value_type val) KAIXO_FROM_ABI(set1(val));

        KAIXO_INLINE static mask_type true_mask() KAIXO_FROM_ABI(true_mask());
        KAIXO_INLINE static mask_type false_mask() KAIXO_FROM_ABI(false_mask());

        // ------------------------------------------------

        KAIXO_INLINE static basic_simd load(const_buffer_type data) KAIXO_FROM_ABI(load(data));
        KAIXO_INLINE static basic_simd loadr(const_buffer_type data) KAIXO_FROM_ABI(loadr(data));
        KAIXO_INLINE static basic_simd loadu(const_buffer_type data) KAIXO_FROM_ABI(loadu(data));

        KAIXO_INLINE void store(buffer_type to) const KAIXO_FROM_ABI(store(to, value));
        KAIXO_INLINE void storeu(buffer_type to) const KAIXO_FROM_ABI(storeu(to, value));
        KAIXO_INLINE void stream(buffer_type to) const KAIXO_FROM_ABI(stream(to, value));

        KAIXO_INLINE static void store(buffer_type to, const basic_simd& from) KAIXO_FROM_ABI(store(to, from));
        KAIXO_INLINE static void storeu(buffer_type to, const basic_simd& from) KAIXO_FROM_ABI(storeu(to, from));
        KAIXO_INLINE static void stream(buffer_type to, const basic_simd& from) KAIXO_FROM_ABI(stream(to, from));

        KAIXO_INLINE static void store(buffer_type to, value_type from) { store(to, set1(from)); }
        KAIXO_INLINE static void storeu(buffer_type to, value_type from) { store(to, set1(from)); }

        KAIXO_INLINE auto gather(const float* data) const -> basic_simd<float, typename simd_find_abi<float, bits>::type> KAIXO_FROM_ABI(gather(data, value));
        KAIXO_INLINE auto gather(const int* data) const -> basic_simd<int, typename simd_find_abi<int, bits>::type> KAIXO_FROM_ABI(gather(data, value));

        KAIXO_INLINE static auto gather(const float* data, const basic_simd& index) -> basic_simd<float, typename simd_find_abi<float, bits>::type> KAIXO_FROM_ABI(gather(data, index));
        KAIXO_INLINE static auto gather(const int* data, const basic_simd& index) -> basic_simd<int, typename simd_find_abi<int, bits>::type> KAIXO_FROM_ABI(gather(data, index));

        // ------------------------------------------------

        KAIXO_INLINE auto to_int() const -> basic_simd<int, typename simd_find_abi<int, bits>::type> KAIXO_FROM_ABI(to_int(value));
        KAIXO_INLINE auto as_int() const -> basic_simd<int, typename simd_find_abi<int, bits>::type> KAIXO_FROM_ABI(as_int(value));
        KAIXO_INLINE auto to_float() const -> basic_simd<float, typename simd_find_abi<float, bits>::type> KAIXO_FROM_ABI(to_float(value));
        KAIXO_INLINE auto as_float() const -> basic_simd<float, typename simd_find_abi<float, bits>::type> KAIXO_FROM_ABI(as_float(value));

        KAIXO_INLINE static auto to_int(const basic_simd& a) -> basic_simd<int, typename simd_find_abi<int, bits>::type> KAIXO_FROM_ABI(to_int(a));
        KAIXO_INLINE static auto as_int(const basic_simd& a) -> basic_simd<int, typename simd_find_abi<int, bits>::type> KAIXO_FROM_ABI(as_int(a));
        KAIXO_INLINE static auto to_float(const basic_simd& a) -> basic_simd<float, typename simd_find_abi<float, bits>::type> KAIXO_FROM_ABI(to_float(a));
        KAIXO_INLINE static auto as_float(const basic_simd& a) -> basic_simd<float, typename simd_find_abi<float, bits>::type> KAIXO_FROM_ABI(as_float(a));
            
        // ------------------------------------------------

        KAIXO_INLINE friend basic_simd operator+(const basic_simd& a, const basic_simd& b) KAIXO_FROM_ABI(add(a, b));
        KAIXO_INLINE friend basic_simd operator-(const basic_simd& a, const basic_simd& b) KAIXO_FROM_ABI(sub(a, b));
        KAIXO_INLINE friend basic_simd operator*(const basic_simd& a, const basic_simd& b) KAIXO_FROM_ABI(mul(a, b));
        KAIXO_INLINE friend basic_simd operator/(const basic_simd& a, const basic_simd& b) KAIXO_FROM_ABI(div(a, b));

        KAIXO_INLINE basic_simd& operator+=(const basic_simd& b) requires requires { Abi::add(value, b); } { value = Abi::add(value, b); return *this; }
        KAIXO_INLINE basic_simd& operator-=(const basic_simd& b) requires requires { Abi::sub(value, b); } { value = Abi::sub(value, b); return *this; }
        KAIXO_INLINE basic_simd& operator*=(const basic_simd& b) requires requires { Abi::mul(value, b); } { value = Abi::mul(value, b); return *this; }
        KAIXO_INLINE basic_simd& operator/=(const basic_simd& b) requires requires { Abi::div(value, b); } { value = Abi::div(value, b); return *this; }

        KAIXO_INLINE friend basic_simd operator-(const basic_simd& a) KAIXO_FROM_ABI(negate(a));

        KAIXO_INLINE static basic_simd fmadd(const basic_simd& a, const basic_simd& b, const basic_simd& c) KAIXO_FROM_ABI(fmadd(a, b, c));
        KAIXO_INLINE static basic_simd fmsub(const basic_simd& a, const basic_simd& b, const basic_simd& c) KAIXO_FROM_ABI(fmsub(a, b, c));

        // ------------------------------------------------

        KAIXO_INLINE friend basic_simd operator&(const basic_simd& a, const mask_type& b) KAIXO_FROM_ABI(bit_and(a, b.value));
        KAIXO_INLINE friend basic_simd operator|(const basic_simd& a, const mask_type& b) KAIXO_FROM_ABI(bit_or(a, b.value));
        KAIXO_INLINE friend basic_simd operator^(const basic_simd& a, const mask_type& b) KAIXO_FROM_ABI(bit_xor(a, b.value));
        KAIXO_INLINE friend basic_simd operator&(const mask_type& a, const basic_simd& b) KAIXO_FROM_ABI(bit_and(a.value, b));
        KAIXO_INLINE friend basic_simd operator|(const mask_type& a, const basic_simd& b) KAIXO_FROM_ABI(bit_or(a.value, b));
        KAIXO_INLINE friend basic_simd operator^(const mask_type& a, const basic_simd& b) KAIXO_FROM_ABI(bit_xor(a.value, b));

        KAIXO_INLINE friend basic_simd operator&(const basic_simd& a, const basic_simd& b) KAIXO_FROM_ABI(bit_and(a, b));
        KAIXO_INLINE friend basic_simd operator|(const basic_simd& a, const basic_simd& b) KAIXO_FROM_ABI(bit_or(a, b));
        KAIXO_INLINE friend basic_simd operator^(const basic_simd& a, const basic_simd& b) KAIXO_FROM_ABI(bit_xor(a, b));
        KAIXO_INLINE friend basic_simd operator~(const basic_simd& a) KAIXO_FROM_ABI(bit_not(a));

        KAIXO_INLINE basic_simd& operator&=(const mask_type& b) requires requires { Abi::bit_and(value, b.value); } { value = Abi::bit_and(value, b.value); return *this; }
        KAIXO_INLINE basic_simd& operator|=(const mask_type& b) requires requires { Abi::bit_or(value, b.value); } { value = Abi::bit_or(value, b.value); return *this; }
        KAIXO_INLINE basic_simd& operator^=(const mask_type& b) requires requires { Abi::bit_xor(value, b.value); } { value = Abi::bit_xor(value, b.value); return *this; }

        KAIXO_INLINE basic_simd& operator&=(const basic_simd& b) requires requires { Abi::bit_and(value, b); } { value = Abi::bit_and(value, b); return *this; }
        KAIXO_INLINE basic_simd& operator|=(const basic_simd& b) requires requires { Abi::bit_or(value, b); } { value = Abi::bit_or(value, b); return *this; }
        KAIXO_INLINE basic_simd& operator^=(const basic_simd& b) requires requires { Abi::bit_xor(value, b); } { value = Abi::bit_xor(value, b); return *this; }

        KAIXO_INLINE friend basic_simd operator<<(const basic_simd& a, int b) KAIXO_FROM_ABI(bit_shift_left(a, b));
        KAIXO_INLINE friend basic_simd operator>>(const basic_simd& a, int b) KAIXO_FROM_ABI(bit_shift_right(a, b));
        KAIXO_INLINE friend basic_simd operator<<(const basic_simd& a, const basic_simd& b) KAIXO_FROM_ABI(bit_shift_left(a, b));
        KAIXO_INLINE friend basic_simd operator>>(const basic_simd& a, const basic_simd& b) KAIXO_FROM_ABI(bit_shift_right(a, b));

        KAIXO_INLINE basic_simd operator<<(int b) KAIXO_FROM_ABI(bit_shift_left(value, b));
        KAIXO_INLINE basic_simd operator>>(int b) KAIXO_FROM_ABI(bit_shift_right(value, b));
        KAIXO_INLINE basic_simd operator<<(const basic_simd& b) KAIXO_FROM_ABI(bit_shift_left(value, b));
        KAIXO_INLINE basic_simd operator>>(const basic_simd& b) KAIXO_FROM_ABI(bit_shift_right(value, b));

        // ------------------------------------------------

        KAIXO_INLINE friend mask_type operator==(const basic_simd& a, const basic_simd& b) KAIXO_FROM_ABI(eq(a, b));
        KAIXO_INLINE friend mask_type operator!=(const basic_simd& a, const basic_simd& b) KAIXO_FROM_ABI(neq(a, b));
        KAIXO_INLINE friend mask_type operator> (const basic_simd& a, const basic_simd& b) KAIXO_FROM_ABI(gt(a, b));
        KAIXO_INLINE friend mask_type operator>=(const basic_simd& a, const basic_simd& b) KAIXO_FROM_ABI(gteq(a, b));
        KAIXO_INLINE friend mask_type operator< (const basic_simd& a, const basic_simd& b) KAIXO_FROM_ABI(lt(a, b));
        KAIXO_INLINE friend mask_type operator<=(const basic_simd& a, const basic_simd& b) KAIXO_FROM_ABI(lteq(a, b));
        
        KAIXO_INLINE mask_type is_negative() const KAIXO_FROM_ABI(lt0(value));
        KAIXO_INLINE static mask_type is_negative(const basic_simd& a) KAIXO_FROM_ABI(lt0(a));

        // ------------------------------------------------
        
        KAIXO_INLINE static basic_simd min(const basic_simd& a, const basic_simd& b) KAIXO_FROM_ABI(min(a, b));
        KAIXO_INLINE static basic_simd max(const basic_simd& a, const basic_simd& b) KAIXO_FROM_ABI(max(a, b));

        // ------------------------------------------------

        KAIXO_INLINE value_type sum() const KAIXO_FROM_ABI(sum(value));
        KAIXO_INLINE static value_type sum(const basic_simd& a) KAIXO_FROM_ABI(sum(a));

        // ------------------------------------------------

        KAIXO_INLINE basic_simd reverse() const KAIXO_FROM_ABI(reverse(value));
        KAIXO_INLINE static basic_simd reverse(const basic_simd& a) KAIXO_FROM_ABI(reverse(a));

        // ------------------------------------------------

        KAIXO_INLINE basic_simd trunc() const KAIXO_FROM_ABI(trunc(value));
        KAIXO_INLINE basic_simd floor() const KAIXO_FROM_ABI(floor(value));
        KAIXO_INLINE basic_simd ceil() const KAIXO_FROM_ABI(ceil(value));
        KAIXO_INLINE basic_simd round() const KAIXO_FROM_ABI(round(value));
        KAIXO_INLINE static basic_simd trunc(const basic_simd& a) KAIXO_FROM_ABI(trunc(a));
        KAIXO_INLINE static basic_simd floor(const basic_simd& a) KAIXO_FROM_ABI(floor(a));
        KAIXO_INLINE static basic_simd ceil(const basic_simd& a) KAIXO_FROM_ABI(ceil(a));
        KAIXO_INLINE static basic_simd round(const basic_simd& a) KAIXO_FROM_ABI(round(a));

        // ------------------------------------------------

        KAIXO_INLINE basic_simd sign() const KAIXO_FROM_ABI(sign(value));
        KAIXO_INLINE basic_simd abs() const KAIXO_FROM_ABI(abs(value));

        KAIXO_INLINE static basic_simd sign(const basic_simd& a) KAIXO_FROM_ABI(sign(a));
        KAIXO_INLINE static basic_simd copysign(const basic_simd& from, const basic_simd& to) KAIXO_FROM_ABI(copysign(from, to));
        KAIXO_INLINE static basic_simd xorsign(const basic_simd& from, const basic_simd& to) KAIXO_FROM_ABI(xorsign(from, to));
        KAIXO_INLINE static basic_simd orsign(const basic_simd& from, const basic_simd& to) KAIXO_FROM_ABI(orsign(from, to));
        KAIXO_INLINE static basic_simd abs(const basic_simd& a) KAIXO_FROM_ABI(abs(a));

        // ------------------------------------------------

        KAIXO_INLINE basic_simd rcp() const KAIXO_FROM_ABI(rcp(value));
        KAIXO_INLINE basic_simd log() const KAIXO_FROM_ABI(log(value));
        KAIXO_INLINE basic_simd log2() const KAIXO_FROM_ABI(log2(value));
        KAIXO_INLINE basic_simd log10() const KAIXO_FROM_ABI(log10(value));
        KAIXO_INLINE basic_simd sqrt() const KAIXO_FROM_ABI(sqrt(value));
        KAIXO_INLINE basic_simd cbrt() const KAIXO_FROM_ABI(cbrt(value));
        KAIXO_INLINE basic_simd exp() const KAIXO_FROM_ABI(exp(value));
        KAIXO_INLINE basic_simd exp2() const KAIXO_FROM_ABI(exp2(value));
        KAIXO_INLINE basic_simd exp10() const KAIXO_FROM_ABI(exp10(value));
        KAIXO_INLINE basic_simd tanh() const KAIXO_FROM_ABI(tanh(value));
        KAIXO_INLINE basic_simd cos() const KAIXO_FROM_ABI(cos(value));
        KAIXO_INLINE basic_simd cosh() const KAIXO_FROM_ABI(cosh(value));
        KAIXO_INLINE basic_simd sin() const KAIXO_FROM_ABI(sin(value));
        KAIXO_INLINE basic_simd sinh() const KAIXO_FROM_ABI(sinh(value));
        KAIXO_INLINE basic_simd pow(const basic_simd& b) KAIXO_FROM_ABI(pow(value, b));

        KAIXO_INLINE static basic_simd rcp(const basic_simd& a) KAIXO_FROM_ABI(rcp(a));
        KAIXO_INLINE static basic_simd log(const basic_simd& a) KAIXO_FROM_ABI(log(a));
        KAIXO_INLINE static basic_simd log2(const basic_simd& a) KAIXO_FROM_ABI(log2(a));
        KAIXO_INLINE static basic_simd log10(const basic_simd& a) KAIXO_FROM_ABI(log10(a));
        KAIXO_INLINE static basic_simd sqrt(const basic_simd& a) KAIXO_FROM_ABI(sqrt(a));
        KAIXO_INLINE static basic_simd cbrt(const basic_simd& a) KAIXO_FROM_ABI(cbrt(a));
        KAIXO_INLINE static basic_simd exp(const basic_simd& a) KAIXO_FROM_ABI(exp(a));
        KAIXO_INLINE static basic_simd exp2(const basic_simd& a) KAIXO_FROM_ABI(exp2(a));
        KAIXO_INLINE static basic_simd tanh(const basic_simd& a) KAIXO_FROM_ABI(tanh(a));
        KAIXO_INLINE static basic_simd cos(const basic_simd& a) KAIXO_FROM_ABI(cos(a));
        KAIXO_INLINE static basic_simd cosh(const basic_simd& a) KAIXO_FROM_ABI(cosh(a));
        KAIXO_INLINE static basic_simd sin(const basic_simd& a) KAIXO_FROM_ABI(sin(a));
        KAIXO_INLINE static basic_simd sinh(const basic_simd& a) KAIXO_FROM_ABI(sinh(a));
        KAIXO_INLINE static basic_simd pow(const basic_simd& a, const basic_simd& b) KAIXO_FROM_ABI(pow(a, b));

        KAIXO_INLINE basic_simd fast_nsin() const KAIXO_FROM_ABI(fast_nsin(value));
        KAIXO_INLINE static basic_simd fast_nsin(const basic_simd& a) KAIXO_FROM_ABI(fast_nsin(a));

        // ------------------------------------------------

        KAIXO_INLINE static basic_simd blend(const mask_type& mask, const basic_simd& a, const basic_simd& b) KAIXO_FROM_ABI(blend(mask.value, a, b));

        // ------------------------------------------------

        KAIXO_INLINE static basic_simd noise() KAIXO_FROM_ABI(noise());

        using _int_type = basic_simd<int, typename simd_find_abi<int, bits>::type>;
        KAIXO_INLINE static basic_simd noise(_int_type& part1, _int_type& part2) KAIXO_FROM_ABI(noise(part1.value, part2.value));

        // ------------------------------------------------

    };

    // ------------------------------------------------

    template<class T, std::size_t Bits>
    using simd = basic_simd<T, typename simd_find_abi<T, Bits>::type>;

    template<class T, std::size_t Bits>
    using simd_mask = basic_simd_mask<T, typename simd_find_abi<T, Bits>::type>;

    // ------------------------------------------------

    using simd_256 = simd<float, 256>;
    using simd_256i = simd<int, 256>;
    using simd_mask_256 = simd_mask<float, 256>;
    using simd_mask_256i = simd_mask<int, 256>;

    // ------------------------------------------------

    template<class Simd>
    concept is_simd = requires() {
        typename Simd::value_type;
        typename Simd::simd_type;
        std::same_as<decltype(Simd::elements), std::size_t>;
        std::same_as<decltype(Simd::bits), std::size_t>;
        std::same_as<decltype(Simd::bytes), std::size_t>;
        std::same_as<decltype(Simd::alignment), std::size_t>;
    };
    
    // ------------------------------------------------

}