#pragma once

// ------------------------------------------------

#include <algorithm>
#include <bit>
#include <concepts>
#include <type_traits>
#include <numeric>
#include <cmath>
#include <concepts>

// ------------------------------------------------

#include <intrin.h>
#include <immintrin.h>
#include <stdint.h>

// ------------------------------------------------

#define KAIXO_VECTORCALL __vectorcall
#define KAIXO_INLINE __forceinline

// ------------------------------------------------

namespace kaixo {

    // ------------------------------------------------
    
    enum class instruction_set : std::size_t {
        SSE      = 0b00000000'00000000'00000000'00000001ull,
        SSE2     = 0b00000000'00000000'00000000'00000010ull,
        SSE3     = 0b00000000'00000000'00000000'00000100ull,
        SSE4_1   = 0b00000000'00000000'00000000'00010000ull,
        AVX      = 0b00000000'00000000'00000001'00000000ull,
        AVX2     = 0b00000000'00000000'00000010'00000000ull,
        AVX512F  = 0b00000000'00000001'00000000'00000000ull,
        AVX512DQ = 0b00000000'00000010'00000000'00000000ull,
        AVX512VL = 0b00000000'00000100'00000000'00000000ull,
    };

    using instruction_sets = std::size_t;

    constexpr instruction_sets& operator|=(instruction_sets& x, instruction_set y) { return x |= static_cast<instruction_sets>(y); }
    constexpr instruction_sets operator|(instruction_set x, instruction_set y) { return static_cast<instruction_sets>(x) | static_cast<instruction_sets>(y); }
    constexpr instruction_sets operator|(instruction_set x, instruction_sets y) { return static_cast<instruction_sets>(x) | y; }
    constexpr instruction_sets operator|(instruction_sets x, instruction_set y) { return x | static_cast<instruction_sets>(y); }
    constexpr instruction_sets& operator&=(instruction_sets& x, instruction_set y) { return x &= static_cast<instruction_sets>(y); }
    constexpr instruction_sets operator&(instruction_set x, instruction_set y) { return static_cast<instruction_sets>(x) & static_cast<instruction_sets>(y); }
    constexpr instruction_sets operator&(instruction_set x, instruction_sets y) { return static_cast<instruction_sets>(x) & y; }
    constexpr instruction_sets operator&(instruction_sets x, instruction_set y) { return x & static_cast<instruction_sets>(y); }

    // ------------------------------------------------
    
    KAIXO_INLINE instruction_sets from_flags(instruction_sets result, std::uint32_t ecx, std::uint32_t edx) {
        if ((edx & (1 << 25)) != 0) result |= instruction_set::SSE;
        if ((edx & (1 << 26)) != 0) result |= instruction_set::SSE2;
        if ((ecx & (1 <<  0)) != 0) result |= instruction_set::SSE3;
        if ((ecx & (1 << 19)) != 0) result |= instruction_set::SSE4_1;
        if ((ecx & (1 << 28)) != 0) result |= instruction_set::AVX;
        return result;
    }
    
    KAIXO_INLINE instruction_sets from_extended_flags(instruction_sets result, std::uint32_t ebx, std::uint32_t ecx, std::uint32_t edx) {
        if ((ebx & (1 <<  5)) != 0) result |= instruction_set::AVX2;
        if ((ebx & (1 << 16)) != 0) result |= instruction_set::AVX512F;
        if ((ebx & (1 << 17)) != 0) result |= instruction_set::AVX512DQ;
        if ((ebx & (1 << 31)) != 0) result |= instruction_set::AVX512VL;
        return result;
    }

    // see https://en.wikipedia.org/wiki/CPUID
    KAIXO_INLINE instruction_sets from_cpuid() {
        instruction_sets  _result{};
        int _registers[4];
        __cpuid(_registers, 0);
        std::uint32_t _max = _registers[0];
        if (_max >= 1u) {
            __cpuid(_registers, 1);
            _result = from_flags(_result, _registers[2], _registers[3]);
        }

        if (_max >= 7u) {
            __cpuidex(_registers, 7, 0);
            _result = from_extended_flags(_result, _registers[1], _registers[2], _registers[3]);
        }

        return _result;
    }

    // ------------------------------------------------
    
    KAIXO_INLINE instruction_sets find_supported_instruction_sets() {
        return from_cpuid();
    }

    inline const instruction_sets supported_instruction_sets = find_supported_instruction_sets();

    // ------------------------------------------------

    template<class Ty, std::size_t Bits> 
    struct underlying_simd;

    template<> 
    struct underlying_simd<int, 0> : std::type_identity<int> {};
    
    template<> 
    struct underlying_simd<int, 128> : std::type_identity<__m128i> {};

    template<>
    struct underlying_simd<int, 256> : std::type_identity<__m256i> {};

    template<> 
    struct underlying_simd<int, 512> : std::type_identity<__m512i> {};

    template<> 
    struct underlying_simd<float, 0> : std::type_identity<float> {};
    
    template<> 
    struct underlying_simd<float, 128> : std::type_identity<__m128> {};

    template<> 
    struct underlying_simd<float, 256> : std::type_identity<__m256> {};

    template<> 
    struct underlying_simd<float, 512> : std::type_identity<__m512> {};

    template<class Ty, std::size_t Bits> 
    using underlying_simd_t = typename underlying_simd<Ty, Bits>::type;

    // ------------------------------------------------
    
#define KAIXO_FROM_MASK512(NAME, TYPE, MASK, FUN, UNDERLYING)\
    KAIXO_INLINE TYPE KAIXO_VECTORCALL NAME(MASK mask) noexcept{                            \
        constexpr UNDERLYING all_mask = std::bit_cast<UNDERLYING>(0xffffffff);              \
        return FUN(mask & (1ull <<  0) ? all_mask : 0, mask & (1ull <<  1) ? all_mask : 0,  \
                   mask & (1ull <<  2) ? all_mask : 0, mask & (1ull <<  3) ? all_mask : 0,  \
                   mask & (1ull <<  4) ? all_mask : 0, mask & (1ull <<  5) ? all_mask : 0,  \
                   mask & (1ull <<  6) ? all_mask : 0, mask & (1ull <<  7) ? all_mask : 0,  \
                   mask & (1ull <<  8) ? all_mask : 0, mask & (1ull <<  9) ? all_mask : 0,  \
                   mask & (1ull << 10) ? all_mask : 0, mask & (1ull << 11) ? all_mask : 0,  \
                   mask & (1ull << 12) ? all_mask : 0, mask & (1ull << 13) ? all_mask : 0,  \
                   mask & (1ull << 14) ? all_mask : 0, mask & (1ull << 15) ? all_mask : 0); \
    }
#define KAIXO_FROM_MASK256(NAME, TYPE, MASK, FUN, UNDERLYING)\
    KAIXO_INLINE TYPE KAIXO_VECTORCALL NAME(MASK mask) noexcept {                           \
        constexpr UNDERLYING all_mask = std::bit_cast<UNDERLYING>(0xffffffff);              \
        return FUN(mask & (1ull <<  0) ? all_mask : 0, mask & (1ull <<  1) ? all_mask : 0,  \
                   mask & (1ull <<  2) ? all_mask : 0, mask & (1ull <<  3) ? all_mask : 0,  \
                   mask & (1ull <<  4) ? all_mask : 0, mask & (1ull <<  5) ? all_mask : 0,  \
                   mask & (1ull <<  6) ? all_mask : 0, mask & (1ull <<  7) ? all_mask : 0); \
    }
#define KAIXO_FROM_MASK128(NAME, TYPE, MASK, FUN, UNDERLYING)\
    KAIXO_INLINE TYPE KAIXO_VECTORCALL NAME(MASK mask) noexcept {                           \
        constexpr UNDERLYING all_mask = std::bit_cast<UNDERLYING>(0xffffffff);              \
        return FUN(mask & (1ull <<  0) ? all_mask : 0, mask & (1ull <<  1) ? all_mask : 0,  \
                   mask & (1ull <<  2) ? all_mask : 0, mask & (1ull <<  3) ? all_mask : 0); \
    }

    KAIXO_FROM_MASK128(_mm_from_mask_epi32, __m128i, __mmask8, _mm_setr_epi32, int);
    KAIXO_FROM_MASK256(_mm256_from_mask_epi32, __m256i, __mmask8, _mm256_setr_epi32, int);
    KAIXO_FROM_MASK512(_mm512_from_mask_epi32, __m512i, __mmask16, _mm512_setr_epi32, int)
    KAIXO_FROM_MASK128(_mm_from_mask_ps, __m128, __mmask8, _mm_setr_ps, float);
    KAIXO_FROM_MASK256(_mm256_from_mask_ps, __m256, __mmask8, _mm256_setr_ps, float);
    KAIXO_FROM_MASK512(_mm512_from_mask_ps, __m512, __mmask16, _mm512_setr_ps, float);

    // ------------------------------------------------
    
// Defines a single implementation case, checks the FOR_INSTRUCTION_SETS, and checks BITS and TYPE of implementation.
#define KAIXO_SIMD_CASE(FOR_INSTRUCTION_SETS, BITS, TYPE) if constexpr ((Instructions & (FOR_INSTRUCTION_SETS)) && std::same_as<TYPE, Ty> && BITS == Bits)
#define KAIXO_SIMD_CASE_ALL(BITS, TYPE) if constexpr (std::same_as<TYPE, Ty> && BITS == Bits)
#define KAIXO_SIMD_BASE if constexpr (Bits == 0)
#define KAIXO_SIMD_BASE_TYPE(TYPE) if constexpr (Bits == 0 && std::same_as<TYPE, Ty>)

    // ------------------------------------------------

    template<class Ty, std::size_t Bits, instruction_sets Instructions>
    struct basic_simd {

        // ------------------------------------------------

        using enum instruction_set;

        // ------------------------------------------------

        constexpr static std::size_t bits = Bits;
        constexpr static std::size_t bytes = Bits / 8;
        constexpr static std::size_t elements = bytes / sizeof(Ty);
        constexpr static instruction_sets instructions = Instructions;

        // ------------------------------------------------

        using base = Ty;
        using simd_type = underlying_simd_t<Ty, Bits>;

        // ------------------------------------------------

        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL setzero() noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_setzero_ps();
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_setzero_ps();
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_setzero_ps();
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_setzero_si128();
            KAIXO_SIMD_CASE(AVX, 256, int) return _mm256_setzero_si256();
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_setzero_si512();
            KAIXO_SIMD_BASE return 0;
        }
        
        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL setone() noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_set1_ps(std::bit_cast<float>(0xFFFFFFFF));
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_set1_ps(std::bit_cast<float>(0xFFFFFFFF));
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_set1_ps(std::bit_cast<float>(0xFFFFFFFF));
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_set1_epi32(std::bit_cast<int>(0xFFFFFFFF));
            KAIXO_SIMD_CASE(AVX, 256, int) return _mm256_set1_epi32(std::bit_cast<int>(0xFFFFFFFF));
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_set1_epi32(std::bit_cast<int>(0xFFFFFFFF));
            KAIXO_SIMD_BASE return 1;
        }

        // ------------------------------------------------

        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL setr(base a1) noexcept {
            KAIXO_SIMD_BASE return a1;
        }
        
        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL setr(base a1, base a2, base a3, base a4) noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_setr_ps(a1, a2, a3, a4); 
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_setr_epi32(a1, a2, a3, a4);
        }

        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL setr(base a1, base a2, base a3, base a4, base a5, base a6, base a7, base a8) noexcept {
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_setr_ps(a1, a2, a3, a4, a5, a6, a7, a8); 
            KAIXO_SIMD_CASE(AVX, 256, int) return _mm256_setr_epi32(a1, a2, a3, a4, a5, a6, a7, a8);
        }
            
        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL setr(
                base a1, base  a2, base  a3, base  a4, base  a5, base  a6, base  a7, base  a8,
                base a9, base a10, base a11, base a12, base a13, base a14, base a15, base a16) noexcept {
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_setr_ps(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16); 
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_setr_epi32(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16); 
        }

        // ------------------------------------------------

        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL set1(base val) noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_set1_ps(val);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_set1_ps(val);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_set1_ps(val);
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_set1_epi32(val);
            KAIXO_SIMD_CASE(AVX, 256, int) return _mm256_set1_epi32(val);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_set1_epi32(val);
            KAIXO_SIMD_BASE return val;
        }

        // ------------------------------------------------

        // Must be aligned!
        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL load(base const* addr) noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_load_ps(addr);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_load_ps(addr);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_load_ps(addr);
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_load_si128(addr);
            KAIXO_SIMD_CASE(AVX, 256, int) return _mm256_load_si256(addr);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_load_si512(addr);
            KAIXO_SIMD_BASE return addr[0];
        }

        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL loadu(base const* addr) noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_loadu_ps(addr);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_loadu_ps(addr);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_loadu_ps(addr);
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_loadu_si128(addr);
            KAIXO_SIMD_CASE(AVX, 256, int) return _mm256_loadu_si256(addr);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_loadu_si512(addr);
            KAIXO_SIMD_BASE return addr[0];
        }

        // ------------------------------------------------

        KAIXO_INLINE void KAIXO_VECTORCALL store(base* addr) const noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_store_ps(addr, value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_store_ps(addr, value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_store_ps(addr, value);
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_store_si128(addr, value);
            KAIXO_SIMD_CASE(AVX, 256, int) return _mm256_store_si256(addr, value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_store_si512(addr, value);
            KAIXO_SIMD_BASE *addr = value;
        }

        KAIXO_INLINE void KAIXO_VECTORCALL storeu(base* addr) const noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_storeu_ps(addr, value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_storeu_ps(addr, value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_storeu_ps(addr, value);
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_storeu_si128(addr, value);
            KAIXO_SIMD_CASE(AVX, 256, int) return _mm256_storeu_si256(addr, value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_storeu_si512(addr, value);
            KAIXO_SIMD_BASE *addr = value;
        }

        // ------------------------------------------------

        simd_type value{};

        // ------------------------------------------------

        KAIXO_INLINE basic_simd() : value() {}
        KAIXO_INLINE basic_simd(base const* addr) : value(loadu(addr).value) {}
        KAIXO_INLINE basic_simd(base val) requires (!std::same_as<simd_type, base>) : value(set1(val).value) {}
        KAIXO_INLINE basic_simd(simd_type val) : value(val) {}
        KAIXO_INLINE explicit basic_simd(bool val) : value(val ? setone().value : setzero().value) {}

        template<class ...Args> requires (sizeof...(Args) == elements && (std::same_as<base, Args> && ...))
        KAIXO_INLINE basic_simd(Args ... args) : value(setr(args...)) {}

        // ------------------------------------------------

        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator+(const basic_simd& a, const basic_simd& b) noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_add_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_add_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_add_ps(a.value, b.value);
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_add_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_add_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_add_epi32(a.value, b.value);
            KAIXO_SIMD_BASE return a.value + b.value;
        }

        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator-(const basic_simd& a, const basic_simd& b) noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_sub_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_sub_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_sub_ps(a.value, b.value);
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_sub_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_sub_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_sub_epi32(a.value, b.value);
            KAIXO_SIMD_BASE return a.value - b.value;
        }
            
        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator*(const basic_simd& a, const basic_simd& b) noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_mul_ps(a. value, b.value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_mul_ps(a. value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_mul_ps(a. value, b.value);
            KAIXO_SIMD_CASE(SSE4_1, 128, int) return _mm_mullo_epi32(a. value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_mullo_epi32(a. value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_mullo_epi32(a. value, b.value);
            KAIXO_SIMD_BASE return a.value * b.value;
        }
            
        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator/(const basic_simd& a, const basic_simd& b) noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_div_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_div_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_div_ps(a.value, b.value);
            KAIXO_SIMD_CASE(SSE, 128, int) return _mm_div_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 256, int) return _mm256_div_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_div_epi32(a.value, b.value);
            KAIXO_SIMD_BASE return a.value / b.value;
        }
            
        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator&(const basic_simd& a, const basic_simd& b) noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_and_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_and_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512DQ, 512, float) return _mm512_and_ps(a.value, b.value);
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_and_si128(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_and_si256(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_and_si512(a.value, b.value);
            KAIXO_SIMD_BASE return a.value * b.value; // & is used with masks, so in base case multiply (0/1 bool)
        }
            
        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator|(const basic_simd& a, const basic_simd& b) noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_or_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_or_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512DQ, 512, float) return _mm512_or_ps(a.value, b.value);
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_or_si128(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_or_si256(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_or_si512(a.value, b.value);
            KAIXO_SIMD_BASE return a.value + b.value; // | is used with masks, so in base case add (either one should be 0)
        }
            
        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator^(const basic_simd& a, const basic_simd& b) noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_xor_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_xor_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512DQ, 512, float) return _mm512_xor_ps(a.value, b.value);
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_xor_si128(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_xor_si256(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_xor_si512(a.value, b.value);
            KAIXO_SIMD_BASE_TYPE(int) return a.value ^ b.value;
        }

        // ------------------------------------------------

        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator==(const basic_simd& a, const basic_simd& b) noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_cmpeq_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 128, float) return _mm_cmp_ps(a.value, b.value, _CMP_EQ_OS);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_cmp_ps(a.value, b.value, _CMP_EQ_OS);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_from_mask_ps(_mm512_cmpeq_ps_mask(a.value, b.value));
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_cmpeq_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_cmpeq_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_cmpeq_epi32(a.value, b.value);
            KAIXO_SIMD_BASE return a.value == b.value;
        }
            
        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator!=(const basic_simd& a, const basic_simd& b) noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_cmpneq_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 128, float) return _mm_cmp_ps(a.value, b.value, _CMP_NEQ_OS);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_cmp_ps(a.value, b.value, _CMP_NEQ_OS);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_from_mask_ps(_mm512_cmpneq_ps_mask(a.value, b.value));
            KAIXO_SIMD_CASE_ALL(128, int) return ~(a == b);
            KAIXO_SIMD_CASE_ALL(256, int) return ~(a == b);
            KAIXO_SIMD_CASE_ALL(512, int) return ~(a == b);
            KAIXO_SIMD_BASE return a.value != b.value;
        }
            
        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator>(const basic_simd& a, const basic_simd& b) noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_cmpgt_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 128, float) return _mm_cmp_ps(a.value, b.value, _CMP_GT_OS);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_cmp_ps(a.value, b.value, _CMP_GT_OS);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_from_mask_ps(_mm512_cmp_ps_mask(a.value, b.value, _CMP_GT_OS));
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_cmpgt_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_cmpgt_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_cmpgt_epi32(a.value, b.value);
            KAIXO_SIMD_BASE return a.value > b.value;
        }
            
        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator<(const basic_simd& a, const basic_simd& b) noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_cmplt_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 128, float) return _mm_cmp_ps(a.value, b.value, _CMP_LT_OS);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_cmp_ps(a.value, b.value, _CMP_LT_OS);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_from_mask_ps(_mm512_cmp_ps_mask(a.value, b.value, _CMP_LT_OS));
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_cmplt_epi32(a.value, b.value);
            KAIXO_SIMD_CASE_ALL(256, int) return (b > a) | (b > a);
            KAIXO_SIMD_CASE_ALL(512, int) return (b > a) | (b > a);
            KAIXO_SIMD_BASE return a.value < b.value;
        }
            
        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator>=(const basic_simd& a, const basic_simd& b) noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_cmpge_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 128, float) return _mm_cmp_ps(a.value, b.value, _CMP_GE_OS);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_cmp_ps(a.value, b.value, _CMP_GE_OS);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_from_mask_ps(_mm512_cmp_ps_mask(a.value, b.value, _CMP_GE_OS));
            KAIXO_SIMD_CASE_ALL(128, int) return (a > b) | (a == b);
            KAIXO_SIMD_CASE_ALL(256, int) return (a > b) | (a == b);
            KAIXO_SIMD_CASE_ALL(512, int) return (a > b) | (a == b);
            KAIXO_SIMD_BASE return a.value >= b.value;
        }
            
        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator<=(const basic_simd& a, const basic_simd& b) noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_cmple_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 128, float) return _mm_cmp_ps(a.value, b.value, _CMP_LE_OS);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_cmp_ps(a.value, b.value, _CMP_LE_OS);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_from_mask_ps(_mm512_cmp_ps_mask(a.value, b.value, _CMP_LE_OS));
            KAIXO_SIMD_CASE_ALL(128, int) return (a < b) | (a == b);
            KAIXO_SIMD_CASE_ALL(256, int) return (a < b) | (a == b);
            KAIXO_SIMD_CASE_ALL(512, int) return (a < b) | (a == b);
            KAIXO_SIMD_BASE return a.value <= b.value;
        }

        // ------------------------------------------------

        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator<<(const basic_simd& a, const basic_simd& b) noexcept {
            KAIXO_SIMD_CASE(AVX2, 128, int) return _mm_sllv_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_sllv_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_sllv_epi32(a.value, b.value);
            KAIXO_SIMD_BASE_TYPE(int) return a.value << b.value;
        }

        template<std::integral Arg>
        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator<<(const basic_simd& a, Arg b) noexcept {
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_slli_epi32(a.value, static_cast<int>(b));
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_slli_epi32(a.value, static_cast<int>(b));
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_slli_epi32(a.value, static_cast<unsigned int>(b));
            KAIXO_SIMD_BASE_TYPE(int) return a.value << b;
        }

        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator>>(const basic_simd& a, const basic_simd& b) noexcept {
            KAIXO_SIMD_CASE(AVX2, 128, int) return _mm_srlv_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_srlv_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_srlv_epi32(a.value, b.value);
            KAIXO_SIMD_BASE_TYPE(int) return a.value >> b.value;
        }

        template<std::integral Arg>
        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator>>(const basic_simd& a, Arg b) noexcept {
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_srli_epi32(a.value, static_cast<int>(b));
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_srli_epi32(a.value, static_cast<int>(b));
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_srli_epi32(a.value, static_cast<unsigned int>(b));
            KAIXO_SIMD_BASE_TYPE(int) return a.value >> b;
        }

        // ------------------------------------------------

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator~() const noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_xor_ps(value, _mm_set1_ps(std::bit_cast<float>(0xFFFFFFFF)));
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_xor_ps(value, _mm256_set1_ps(std::bit_cast<float>(0xFFFFFFFF)));
            KAIXO_SIMD_CASE(AVX512DQ, 512, float) return _mm512_xor_ps(value, _mm512_set1_ps(std::bit_cast<float>(0xFFFFFFFF)));
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_xor_si128(value, _mm_set1_epi32(std::bit_cast<int>(0xFFFFFFFF)));
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_xor_si256(value, _mm256_set1_epi32(std::bit_cast<int>(0xFFFFFFFF)));
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_xor_si512(value, _mm512_set1_epi32(std::bit_cast<int>(0xFFFFFFFF)));
            KAIXO_SIMD_BASE return !value;
        }

        // ------------------------------------------------

        // SSE3 | SSE
        KAIXO_INLINE static base KAIXO_VECTORCALL _sum_general(simd_type a) noexcept {
            base vals[elements];
            basic_simd{ a }.store(vals);
            return std::accumulate(vals, vals + elements, base{});
        }
            
        // SSE3 | SSE
        KAIXO_INLINE static base KAIXO_VECTORCALL _sum_128_f(__m128 a) noexcept {
            __m128 shuf = _mm_movehdup_ps(a);
            __m128 sums = _mm_add_ps(a, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            return _mm_cvtss_f32(sums);
        }

        // AVX | SSE
        KAIXO_INLINE static base KAIXO_VECTORCALL _sum_256_f(__m256 a) noexcept {
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

        // AVX512F | AVX512DQ | AVX | SSE
        KAIXO_INLINE static base KAIXO_VECTORCALL _sum_512_f(__m512 a) noexcept {
            __m256 v0 = _mm512_castps512_ps256(a);
            __m256 v1 = _mm512_extractf32x8_ps(a, 1);
            __m256 x0 = _mm256_add_ps(v0, v1);
            return _sum_256_f(x0);
        }

        KAIXO_INLINE base KAIXO_VECTORCALL sum() const noexcept {
            KAIXO_SIMD_CASE(SSE3 | SSE, 128, float) return _sum_128_f(value);
            KAIXO_SIMD_CASE(AVX | SSE, 256, float) return _sum_256_f(value);
            KAIXO_SIMD_CASE(AVX512F | AVX512DQ | AVX | SSE, 512, float) return _sum_512_f(value);
            KAIXO_SIMD_CASE_ALL(128, float) return _sum_general(value);
            KAIXO_SIMD_CASE_ALL(256, float) return _sum_general(value);
            KAIXO_SIMD_CASE_ALL(512, float) return _sum_general(value);
            KAIXO_SIMD_CASE_ALL(128, int) return _sum_general(value);
            KAIXO_SIMD_CASE_ALL(256, int) return _sum_general(value);
            KAIXO_SIMD_CASE_ALL(512, int) return _sum_general(value);
            KAIXO_SIMD_BASE return value;
        }

        // ------------------------------------------------

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL trunc() const noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_trunc_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_trunc_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_trunc_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::trunc(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL floor() const noexcept {
            KAIXO_SIMD_CASE(SSE4_1, 128, float) return _mm_floor_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_floor_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_floor_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::floor(value);
        }
        
        KAIXO_INLINE basic_simd KAIXO_VECTORCALL fmod1() const noexcept {
            KAIXO_SIMD_CASE(SSE | SSE4_1, 128, float) return _mm_sub_ps(value, _mm_floor_ps(value));
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_sub_ps(value, _mm_trunc_ps(value));
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_sub_ps(value, _mm256_floor_ps(value));
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_sub_ps(value, _mm512_floor_ps(value));
            KAIXO_SIMD_BASE_TYPE(float) return value - std::trunc(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL ceil() const noexcept {
            KAIXO_SIMD_CASE(SSE4_1, 128, float) _mm_ceil_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_ceil_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_ceil_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::ceil(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL round() const noexcept {
            KAIXO_SIMD_CASE(AVX512F | AVX512VL, 128, float) return _mm_roundscale_ps(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            KAIXO_SIMD_CASE(AVX512F | AVX512VL, 256, float) return _mm256_roundscale_ps(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            KAIXO_SIMD_CASE(AVX512F | AVX512VL, 512, float) return _mm512_roundscale_ps(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            KAIXO_SIMD_BASE_TYPE(float) return std::round(value);
        }

        // ------------------------------------------------

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL log() const noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_log_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_log_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_log_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::log(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL log2() const noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_log2_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_log2_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_log2_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::log2(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL log10() const noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_log10_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_log10_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_log10_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::log10(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL sqrt() const noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_sqrt_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_sqrt_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_sqrt_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::sqrt(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL cbrt() const noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_cbrt_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_cbrt_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_cbrt_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::cbrt(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL exp() const noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_exp_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_exp_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_exp_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::exp(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL exp2() const noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_exp2_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_exp2_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_exp2_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::exp2(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL exp10() const noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_exp10_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_exp10_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return _mm512_exp10_ps(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL tanh() const noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_tanh_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_tanh_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_tanh_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::tanh(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL abs() const noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_andnot_ps(_mm_set1_ps(-0.0), value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_andnot_ps(_mm256_set1_ps(-0.0), value);
            KAIXO_SIMD_CASE(AVX512F | AVX512DQ, 512, float) return _mm512_andnot_ps(_mm512_set1_ps(-0.0), value);
            KAIXO_SIMD_BASE_TYPE(float) return std::abs(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL cos() const noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_cos_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_cos_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_cos_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::cos(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL cosh() const noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_cosh_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_cosh_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_cosh_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::cosh(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL sin() const noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_sin_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_sin_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_sin_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::sin(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL sinh() const noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_sinh_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_sinh_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_sinh_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::sinh(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL pow(const basic_simd& b) const noexcept {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_pow_ps(value, b.value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_pow_ps(value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_pow_ps(value, b.value);
            KAIXO_SIMD_BASE_TYPE(float) return std::pow(value, b.value);
        }

        // ------------------------------------------------

        template<class To>
        KAIXO_INLINE basic_simd<To, Bits, Instructions> KAIXO_VECTORCALL cast() const noexcept {
            KAIXO_SIMD_BASE return static_cast<To>(value);
            if constexpr (std::same_as<Ty, To>) return value;
            else if constexpr (std::same_as<To, int>) {
                KAIXO_SIMD_CASE(SSE2, 128, float) return _mm_cvtps_epi32(value);
                KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_cvtps_epi32(value);
                KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_cvtps_epi32(value);
            } else if constexpr (std::same_as<To, float>) {
                KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_cvtepi32_ps(value);
                KAIXO_SIMD_CASE(AVX, 256, int) return _mm256_cvtepi32_ps(value);
                KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_cvtepi32_ps(value);
            }
        }

        template<class To>
        KAIXO_INLINE basic_simd<To, Bits, Instructions> KAIXO_VECTORCALL reinterpret() const noexcept {
            KAIXO_SIMD_BASE return std::bit_cast<To>(value);
            if constexpr (std::same_as<Ty, To>) return value;
            else if constexpr (std::same_as<To, int>) {
                KAIXO_SIMD_CASE(SSE2, 128, float) return _mm_castps_si128(value);
                KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_castps_si256(value);
                KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_castps_si512(value);
            } else if constexpr (std::same_as<To, float>) {
                KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_castsi128_ps(value);
                KAIXO_SIMD_CASE(AVX, 256, int) return _mm256_castsi256_ps(value);
                KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_castsi512_ps(value);
            }
        }

        // ------------------------------------------------

        KAIXO_INLINE basic_simd<float, Bits, Instructions> KAIXO_VECTORCALL gather(float const* data) const noexcept {
            KAIXO_SIMD_CASE(AVX2, 128, int) return _mm_i32gather_ps(data, value, sizeof(float));
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_i32gather_ps(data, value, sizeof(float));
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_i32gather_ps(data, value, sizeof(float));
            KAIXO_SIMD_BASE_TYPE(int) return data[value];
        }

        KAIXO_INLINE basic_simd<int, Bits, Instructions> KAIXO_VECTORCALL gather(int const* data) const noexcept {
            KAIXO_SIMD_CASE(AVX2, 128, int) return _mm_i32gather_epi32(data, value, sizeof(int));
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_i32gather_epi32(data, value, sizeof(int));
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_i32gather_epi32(data, value, sizeof(int));
            KAIXO_SIMD_BASE_TYPE(int) return data[value];
        }

        // ------------------------------------------------
        
        KAIXO_INLINE basic_simd KAIXO_VECTORCALL iff(auto then, auto otherwise) const noexcept {
            KAIXO_SIMD_BASE return value ? then() : otherwise();
            else return *this & then() | ~*this & otherwise();
        }

        // ------------------------------------------------

    };

    // ------------------------------------------------

#undef KAIXO_SIMD_CALL

    // ------------------------------------------------

    template<class Simd>
    concept is_simd = requires() {
        typename Simd::base;
        typename Simd::simd_type;
        std::same_as<decltype(Simd::elements), std::size_t>;
        std::same_as<decltype(Simd::bits), std::size_t>;
        std::same_as<decltype(Simd::bytes), std::size_t>;
        std::same_as<decltype(Simd::instructions), instruction_sets>;
    };

    // ------------------------------------------------

    using simd_128 = basic_simd<float, 128, instruction_set::SSE | instruction_set::SSE2 | instruction_set::SSE3 | instruction_set::SSE4_1>;
    using simd_256 = basic_simd<float, 256, simd_128::instructions | instruction_set::AVX | instruction_set::AVX2>;
    using simd_512 = basic_simd<float, 512, simd_256::instructions | instruction_set::AVX512F | instruction_set::AVX512DQ | instruction_set::AVX512VL>;
    
    using simd_128i = basic_simd<int, 128, simd_128::instructions>;
    using simd_256i = basic_simd<int, 256, simd_256::instructions>;
    using simd_512i = basic_simd<int, 512, simd_512::instructions>;

    // ------------------------------------------------
    
    template<class Ty>
    KAIXO_INLINE decltype(auto) choose_simd_path(auto lambda) {
        if ((~supported_instruction_sets & simd_512::instructions) == 0) return lambda.operator()<basic_simd<Ty, 512, simd_512::instructions>>();
        if ((~supported_instruction_sets & simd_256::instructions) == 0) return lambda.operator()<basic_simd<Ty, 256, simd_256::instructions>>();
        if ((~supported_instruction_sets & simd_128::instructions) == 0) return lambda.operator()<basic_simd<Ty, 128, simd_128::instructions>>();
        return lambda.operator()<basic_simd<Ty, 0, 0>>();
    }

    // ------------------------------------------------

    template<class Ty> struct base : std::type_identity<Ty> {};
    template<is_simd Ty> struct base<Ty> : std::type_identity<typename Ty::base> {};
    template<class Ty> using base_t = typename base<Ty>::type;

    // ------------------------------------------------

    template<class Type>
    KAIXO_INLINE Type KAIXO_VECTORCALL loadu(base_t<Type> const* ptr, std::size_t index) noexcept {
        if constexpr (!is_simd<Type>) return ptr[index];
        else return Type(ptr + index);
    }

    template<class Type>
    KAIXO_INLINE Type KAIXO_VECTORCALL load(base_t<Type> const* ptr, std::size_t index) noexcept {
        if constexpr (!is_simd<Type>) return ptr[index];
        else return Type::load(ptr + index);
    }

    template<class Type>
    KAIXO_INLINE void KAIXO_VECTORCALL storeu(base_t<Type>* ptr, const Type& value) noexcept {
        if constexpr (!is_simd<Type>) *ptr = value;
        else value.storeu(ptr);
    }

    template<class Type>
    KAIXO_INLINE void KAIXO_VECTORCALL store(base_t<Type>* ptr, const Type& value) noexcept {
        if constexpr (!is_simd<Type>) *ptr = value;
        else value.store(ptr);
    }

    template<class Type, std::convertible_to<Type> B>
    KAIXO_INLINE Type KAIXO_VECTORCALL bool_and(const Type& condition, B value) noexcept {
        if constexpr (!is_simd<Type>) return condition * value;
        else return condition & value;
    }

    template<class Type, std::invocable A, std::invocable B>
    KAIXO_INLINE Type KAIXO_VECTORCALL iff(const Type& condition, A then, B otherwise) noexcept {
        if constexpr (!is_simd<Type>) return condition ? then() : otherwise();
        else return condition & then() | ~condition & otherwise();
    }

    // Multiply with 1 or -1
    template<class Type, std::convertible_to<Type> B>
    KAIXO_INLINE Type KAIXO_VECTORCALL mul1(const Type& condition, B value) noexcept {
        if constexpr (!is_simd<Type>) return condition * value;
        else return condition ^ ((-0.f) & value); // Toggle sign bit if value has sign bit
    };

    template<class To, class Type>
    KAIXO_INLINE To KAIXO_VECTORCALL cast(const Type& v) noexcept {
        if constexpr (!is_simd<Type>) return (To)v;
        else return v.template cast<To>();
    }

    template<class To, class Type>
    KAIXO_INLINE To KAIXO_VECTORCALL reinterpret(const Type& v) noexcept {
        if constexpr (!is_simd<Type>) return std::bit_cast<To>(v);
        else return v.template reinterpret<To>();
    }

    template<class Type, class Ptr>
    KAIXO_INLINE decltype(auto) KAIXO_VECTORCALL gather(Ptr* data, const Type& index) noexcept {
        if constexpr (!is_simd<Type>) return data[(std::int64_t)index];
        else return index.gather(data);
    }

    template<class Type>
    KAIXO_INLINE base_t<Type> KAIXO_VECTORCALL sum(const Type& value) noexcept {
        if constexpr (!is_simd<Type>) return value;
        else return value.sum();
    }

    // ------------------------------------------------

}