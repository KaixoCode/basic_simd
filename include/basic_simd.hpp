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

#ifdef _MSC_VER
#define KAIXO_VECTORCALL __vectorcall
#define KAIXO_INLINE __forceinline
#endif 

#if defined(__GNUC__) || defined(__GNUG__) || defined(__clang__)
#define ARM_ARCHITECTURE
#define KAIXO_VECTORCALL
#define KAIXO_INLINE
//#include <immintrin.h>
//#include "cpuid.h"
#endif

#ifndef ARM_ARCHITECTURE
#include <immintrin.h>
#endif

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
        FMA      = 0b00000000'00000000'00000100'00000000ull,
        AVX512F  = 0b00000000'00000001'00000000'00000000ull,
        AVX512DQ = 0b00000000'00000010'00000000'00000000ull,
        AVX512VL = 0b00000000'00000100'00000000'00000000ull,
        AVX512BW = 0b00000000'00001000'00000000'00000000ull,
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
    
#ifdef ARM_ARCHITECTURE

    // ------------------------------------------------

    inline const instruction_sets supported_instruction_sets = 0;

    template<class Ty, std::size_t Bits>
    struct underlying_simd : std::type_identity<Ty> {};

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
        using simd_type = Ty;

        // ------------------------------------------------

    };

    // ------------------------------------------------

#else

    // ------------------------------------------------

    KAIXO_INLINE instruction_sets from_flags(instruction_sets result, std::uint32_t ecx, std::uint32_t edx) {
        if ((edx & (1 << 25)) != 0) result |= instruction_set::SSE;
        if ((edx & (1 << 26)) != 0) result |= instruction_set::SSE2;
        if ((ecx & (1 <<  0)) != 0) result |= instruction_set::SSE3;
        if ((ecx & (1 << 19)) != 0) result |= instruction_set::SSE4_1;
        if ((ecx & (1 << 28)) != 0) result |= instruction_set::AVX;
        if ((ecx & (1 << 12)) != 0) result |= instruction_set::FMA;
        return result;
    }
    
    KAIXO_INLINE instruction_sets from_extended_flags(instruction_sets result, std::uint32_t ebx, std::uint32_t /*ecx*/, std::uint32_t /*edx*/) {
        if ((ebx & (1 <<  5)) != 0) result |= instruction_set::AVX2;
        if ((ebx & (1 << 16)) != 0) result |= instruction_set::AVX512F;
        if ((ebx & (1 << 17)) != 0) result |= instruction_set::AVX512DQ;
        if ((ebx & (1 << 30)) != 0) result |= instruction_set::AVX512BW;
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
    struct underlying_simd<short, 0> : std::type_identity<short> {};
    
    template<> 
    struct underlying_simd<short, 128> : std::type_identity<__m128i> {};

    template<>
    struct underlying_simd<short, 256> : std::type_identity<__m256i> {};

    template<> 
    struct underlying_simd<short, 512> : std::type_identity<__m512i> {};

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
    KAIXO_INLINE TYPE KAIXO_VECTORCALL NAME(MASK mask) {                            \
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
    KAIXO_INLINE TYPE KAIXO_VECTORCALL NAME(MASK mask)  {                           \
        constexpr UNDERLYING all_mask = std::bit_cast<UNDERLYING>(0xffffffff);              \
        return FUN(mask & (1ull <<  0) ? all_mask : 0, mask & (1ull <<  1) ? all_mask : 0,  \
                   mask & (1ull <<  2) ? all_mask : 0, mask & (1ull <<  3) ? all_mask : 0,  \
                   mask & (1ull <<  4) ? all_mask : 0, mask & (1ull <<  5) ? all_mask : 0,  \
                   mask & (1ull <<  6) ? all_mask : 0, mask & (1ull <<  7) ? all_mask : 0); \
    }
#define KAIXO_FROM_MASK128(NAME, TYPE, MASK, FUN, UNDERLYING)\
    KAIXO_INLINE TYPE KAIXO_VECTORCALL NAME(MASK mask)  {                           \
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

        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL setzero()  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_setzero_ps();
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_setzero_ps();
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_setzero_ps();
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_setzero_si128();
            KAIXO_SIMD_CASE(AVX, 256, int) return _mm256_setzero_si256();
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_setzero_si512();
            KAIXO_SIMD_CASE(SSE2, 128, short) return _mm_setzero_si128();
            KAIXO_SIMD_CASE(AVX, 256, short) return _mm256_setzero_si256();
            KAIXO_SIMD_CASE(AVX512F, 512, short) return _mm512_setzero_si512();
            KAIXO_SIMD_BASE return 0;
        }
        
        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL setone()  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_set1_ps(std::bit_cast<float>(0xFFFFFFFF));
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_set1_ps(std::bit_cast<float>(0xFFFFFFFF));
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_set1_ps(std::bit_cast<float>(0xFFFFFFFF));
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_set1_epi32(std::bit_cast<int>(0xFFFFFFFF));
            KAIXO_SIMD_CASE(AVX, 256, int) return _mm256_set1_epi32(std::bit_cast<int>(0xFFFFFFFF));
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_set1_epi32(std::bit_cast<int>(0xFFFFFFFF));
            KAIXO_SIMD_CASE(SSE2, 128, short) return _mm_set1_epi16(std::bit_cast<short>(0xFFFF));
            KAIXO_SIMD_CASE(AVX, 256, short) return _mm256_set1_epi16(std::bit_cast<short>(0xFFFF));
            KAIXO_SIMD_CASE(AVX512F, 512, short) return _mm512_set1_epi16(std::bit_cast<short>(0xFFFF));
            KAIXO_SIMD_BASE return 1;
        }
        
        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL setincr()  {
            KAIXO_SIMD_CASE(SSE, 128, float)     return setr(0, 1, 2, 3);
            KAIXO_SIMD_CASE(AVX, 256, float)     return setr(0, 1, 2, 3, 4, 5, 6, 7);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return setr(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            KAIXO_SIMD_CASE(SSE2, 128, int)      return setr(0, 1, 2, 3);
            KAIXO_SIMD_CASE(AVX, 256, int)       return setr(0, 1, 2, 3, 4, 5, 6, 7);
            KAIXO_SIMD_CASE(AVX512F, 512, int)   return setr(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            KAIXO_SIMD_CASE(SSE2, 128, short)    return setr(0, 1, 2, 3, 4, 5, 6, 7);
            KAIXO_SIMD_CASE(AVX, 256, short)     return setr(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            KAIXO_SIMD_CASE(AVX512F, 512, short) return setr(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
            KAIXO_SIMD_BASE return 0;
        }
        
        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL setalternate() {
            KAIXO_SIMD_CASE(SSE, 128, float)     return setr(0, 1, 0, 1);
            KAIXO_SIMD_CASE(AVX, 256, float)     return setr(0, 1, 0, 1, 0, 1, 0, 1);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return setr(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
            KAIXO_SIMD_CASE(SSE2, 128, int)      return setr(0, 1, 0, 1);
            KAIXO_SIMD_CASE(AVX, 256, int)       return setr(0, 1, 0, 1, 0, 1, 0, 1);
            KAIXO_SIMD_CASE(AVX512F, 512, int)   return setr(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
            KAIXO_SIMD_CASE(SSE2, 128, short)    return setr(0, 1, 0, 1, 0, 1, 0, 1);
            KAIXO_SIMD_CASE(AVX, 256, short)     return setr(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
            KAIXO_SIMD_CASE(AVX512F, 512, short) return setr(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
            KAIXO_SIMD_BASE return 0;
        }

        // ------------------------------------------------

        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL setr(base a1)  {
            KAIXO_SIMD_BASE return a1;
        }
        
        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL setr(base a1, base a2, base a3, base a4)  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_setr_ps(a1, a2, a3, a4); 
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_setr_epi32(a1, a2, a3, a4);
        }

        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL setr(base a1, base a2, base a3, base a4, base a5, base a6, base a7, base a8)  {
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_setr_ps(a1, a2, a3, a4, a5, a6, a7, a8); 
            KAIXO_SIMD_CASE(AVX, 256, int) return _mm256_setr_epi32(a1, a2, a3, a4, a5, a6, a7, a8);
            KAIXO_SIMD_CASE(SSE2, 128, short) return _mm_setr_epi16(a1, a2, a3, a4, a5, a6, a7, a8);
        }
            
        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL setr(
                base a1, base  a2, base  a3, base  a4, base  a5, base  a6, base  a7, base  a8,
                base a9, base a10, base a11, base a12, base a13, base a14, base a15, base a16)  {
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_setr_ps(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16); 
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_setr_epi32(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16); 
            KAIXO_SIMD_CASE(AVX, 256, short) return _mm256_setr_epi16(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16); 
        }
        
        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL setr(
                base  a1, base  a2, base  a3, base  a4, base  a5, base  a6, base  a7, base  a8,
                base  a9, base a10, base a11, base a12, base a13, base a14, base a15, base a16,
                base a17, base a18, base a19, base a20, base a21, base a22, base a23, base a24,
                base a25, base a26, base a27, base a28, base a29, base a30, base a31, base a32)  {
            KAIXO_SIMD_CASE(AVX512F, 512, short) return _mm512_setr_epi16(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32);
        }
        
        // ------------------------------------------------

        template<base Default = 0, std::same_as<base> ...Tys> requires (sizeof...(Tys) <= elements)
        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL setfirst(Tys... values)  {
            using index_sequence = std::make_index_sequence<elements - sizeof...(Tys)>;

            return [&]<std::size_t ...Is>(std::index_sequence<Is...>) {
                return setr(values..., (Is, Default)...);
            }(index_sequence{});
        }
        
        template<base Default = 0, std::same_as<base> ...Tys> requires (sizeof...(Tys) <= elements)
        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL setlast(Tys... values)  {
            using index_sequence = std::make_index_sequence<elements - sizeof...(Tys)>;

            return [&]<std::size_t ...Is>(std::index_sequence<Is...>) {
                return setr((Is, Default)..., values...);
            }(index_sequence{});
        }

        // ------------------------------------------------

        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL set1(base val)  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_set1_ps(val);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_set1_ps(val);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_set1_ps(val);
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_set1_epi32(val);
            KAIXO_SIMD_CASE(AVX, 256, int) return _mm256_set1_epi32(val);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_set1_epi32(val);
            KAIXO_SIMD_CASE(SSE2, 128, short) return _mm_set1_epi16(val);
            KAIXO_SIMD_CASE(AVX, 256, short) return _mm256_set1_epi16(val);
            KAIXO_SIMD_CASE(AVX512F, 512, short) return _mm512_set1_epi16(val);
            KAIXO_SIMD_BASE return val;
        }

        // ------------------------------------------------

        // Must be aligned!
        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL load(base const* addr)  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_load_ps(addr);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_load_ps(addr);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_load_ps(addr);
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_load_si128((__m128i const*)addr);
            KAIXO_SIMD_CASE(AVX, 256, int) return _mm256_load_si256((__m256i const*)addr);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_load_si512((__m512i const*)addr);
            KAIXO_SIMD_CASE(SSE2, 128, short) return _mm_load_si128((__m128i const*)addr);
            KAIXO_SIMD_CASE(AVX, 256, short) return _mm256_load_si256((__m256i const*)addr);
            KAIXO_SIMD_CASE(AVX512F, 512, short) return _mm512_load_si512((__m512i const*)addr);
            KAIXO_SIMD_BASE return addr[0];
        }

        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL loadr(base const* addr)  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_loadr_ps(addr);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_setr_ps(addr[7], addr[6], addr[5], addr[4], addr[3], addr[2], addr[1], addr[0]);
            KAIXO_SIMD_BASE return addr[0];
        }

        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL loadu(base const* addr)  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_loadu_ps(addr);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_loadu_ps(addr);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_loadu_ps(addr);
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_loadu_si128((__m128i const*)addr);
            KAIXO_SIMD_CASE(AVX, 256, int) return _mm256_loadu_si256((__m256i const*)addr);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_loadu_si512((__m512i const*)addr);
            KAIXO_SIMD_CASE(SSE2, 128, short) return _mm_loadu_si128((__m128i const*)addr);
            KAIXO_SIMD_CASE(AVX, 256, short) return _mm256_loadu_si256((__m256i const*)addr);
            KAIXO_SIMD_CASE(AVX512F, 512, short) return _mm512_loadu_si512((__m512i const*)addr);
            KAIXO_SIMD_BASE return addr[0];
        }

        // ------------------------------------------------

        KAIXO_INLINE void KAIXO_VECTORCALL store(base* addr) const  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_store_ps(addr, value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_store_ps(addr, value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_store_ps(addr, value);
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_store_si128((__m128i*)addr, value);
            KAIXO_SIMD_CASE(AVX, 256, int) return _mm256_store_si256((__m256i*)addr, value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_store_si512((__m512i*)addr, value);
            KAIXO_SIMD_CASE(SSE2, 128, short) return _mm_store_si128((__m128i*)addr, value);
            KAIXO_SIMD_CASE(AVX, 256, short) return _mm256_store_si256((__m256i*)addr, value);
            KAIXO_SIMD_CASE(AVX512F, 512, short) return _mm512_store_si512((__m512i*)addr, value);
            KAIXO_SIMD_BASE *addr = value;
        }

        KAIXO_INLINE void KAIXO_VECTORCALL storeu(base* addr) const  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_storeu_ps(addr, value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_storeu_ps(addr, value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_storeu_ps(addr, value);
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_storeu_si128((__m128i*)addr, value);
            KAIXO_SIMD_CASE(AVX, 256, int) return _mm256_storeu_si256((__m256i*)addr, value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_storeu_si512((__m512i*)addr, value);
            KAIXO_SIMD_CASE(SSE2, 128, short) return _mm_storeu_si128((__m128i*)addr, value);
            KAIXO_SIMD_CASE(AVX, 256, short) return _mm256_storeu_si256((__m256i*)addr, value);
            KAIXO_SIMD_CASE(AVX512F, 512, short) return _mm512_storeu_si512((__m512i*)addr, value);
            KAIXO_SIMD_BASE *addr = value;
        }

        KAIXO_INLINE void KAIXO_VECTORCALL stream(base* addr) const  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_stream_ps(addr, value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_stream_ps(addr, value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_stream_ps(addr, value);
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_stream_si128((__m128i*)addr, value);
            KAIXO_SIMD_CASE(AVX, 256, int) return _mm256_stream_si256((__m256i*)addr, value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_stream_si512((__m512i*)addr, value);
            KAIXO_SIMD_CASE(SSE2, 128, short) return _mm_stream_si128((__m128i*)addr, value);
            KAIXO_SIMD_CASE(AVX, 256, short) return _mm256_stream_si256((__m256i*)addr, value);
            KAIXO_SIMD_CASE(AVX512F, 512, short) return _mm512_stream_si512((__m512i*)addr, value);
            KAIXO_SIMD_BASE *addr = value;
        }

        // ------------------------------------------------

        simd_type value{};

        // ------------------------------------------------

        KAIXO_INLINE basic_simd() : value() {}
        KAIXO_INLINE basic_simd(base const* addr) : value(loadu(addr).value) {}
        KAIXO_INLINE basic_simd(simd_type val) : value(val) {}
        KAIXO_INLINE explicit basic_simd(bool val) : value(val ? setone().value : setzero().value) {}

        template<class Arg> 
            requires std::convertible_to<Arg, base>
        KAIXO_INLINE basic_simd(Arg val) : value(set1(static_cast<base>(val)).value) {}

        template<std::integral Ty>
        KAIXO_INLINE basic_simd(const basic_simd<Ty, bits, instructions>& other) : value(other.value) {}
        template<std::integral Ty>
        KAIXO_INLINE basic_simd(basic_simd<Ty, bits, instructions>&& other) : value(std::move(other.value)) {}

        template<class ...Args> 
            requires (sizeof...(Args) == elements && (std::convertible_to<Args, base> && ...))
        KAIXO_INLINE basic_simd(Args ... args) : value(setr(static_cast<base>(args)...).value) {}

        // ------------------------------------------------

        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator+(const basic_simd& a, const basic_simd& b)  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_add_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_add_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_add_ps(a.value, b.value);
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_add_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_add_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_add_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(SSE2, 128, short) return _mm_add_epi16(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, short) return _mm256_add_epi16(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512BW, 512, short) return _mm512_add_epi16(a.value, b.value);
            KAIXO_SIMD_BASE return a.value + b.value;
        }

        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator-(const basic_simd& a, const basic_simd& b)  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_sub_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_sub_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_sub_ps(a.value, b.value);
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_sub_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_sub_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_sub_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(SSE2, 128, short) return _mm_sub_epi16(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, short) return _mm256_sub_epi16(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512BW, 512, short) return _mm512_sub_epi16(a.value, b.value);
            KAIXO_SIMD_BASE return a.value - b.value;
        }
            
        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator*(const basic_simd& a, const basic_simd& b)  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_mul_ps(a. value, b.value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_mul_ps(a. value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_mul_ps(a. value, b.value);
            KAIXO_SIMD_CASE(SSE4_1, 128, int) return _mm_mullo_epi32(a. value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_mullo_epi32(a. value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_mullo_epi32(a. value, b.value);
            KAIXO_SIMD_CASE(SSE2, 128, short) return _mm_mullo_epi16(a. value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, short) return _mm256_mullo_epi16(a. value, b.value);
            KAIXO_SIMD_CASE(AVX512BW, 512, short) return _mm512_mullo_epi16(a. value, b.value);
            KAIXO_SIMD_BASE return a.value * b.value;
        }
            
        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator/(const basic_simd& a, const basic_simd& b)  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_div_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_div_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_div_ps(a.value, b.value);
            KAIXO_SIMD_CASE(SSE, 128, int) return _mm_div_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 256, int) return _mm256_div_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_div_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(SSE, 128, int) return _mm_div_epi16(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 256, int) return _mm256_div_epi16(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_div_epi16(a.value, b.value);
            KAIXO_SIMD_BASE return a.value / b.value;
        }
            
        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator&(const basic_simd& a, const basic_simd& b)  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_and_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_and_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512DQ, 512, float) return _mm512_and_ps(a.value, b.value);
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_and_si128(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_and_si256(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_and_si512(a.value, b.value);
            KAIXO_SIMD_CASE(SSE2, 128, short) return _mm_and_si128(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, short) return _mm256_and_si256(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, short) return _mm512_and_si512(a.value, b.value);
            KAIXO_SIMD_BASE return a.value * b.value; // & is used with masks, so in base case multiply (0/1 bool)
        }
            
        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator|(const basic_simd& a, const basic_simd& b)  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_or_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_or_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512DQ, 512, float) return _mm512_or_ps(a.value, b.value);
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_or_si128(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_or_si256(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_or_si512(a.value, b.value);
            KAIXO_SIMD_CASE(SSE2, 128, short) return _mm_or_si128(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, short) return _mm256_or_si256(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, short) return _mm512_or_si512(a.value, b.value);
            KAIXO_SIMD_BASE return a.value + b.value; // | is used with masks, so in base case add (either one should be 0)
        }
            
        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator^(const basic_simd& a, const basic_simd& b)  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_xor_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_xor_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512DQ, 512, float) return _mm512_xor_ps(a.value, b.value);
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_xor_si128(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_xor_si256(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_xor_si512(a.value, b.value);
            KAIXO_SIMD_CASE(SSE2, 128, short) return _mm_xor_si128(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, short) return _mm256_xor_si256(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, short) return _mm512_xor_si512(a.value, b.value);
            KAIXO_SIMD_BASE_TYPE(int) return a.value ^ b.value;
        }

        // ------------------------------------------------

        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator==(const basic_simd& a, const basic_simd& b)  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_cmpeq_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 128, float) return _mm_cmp_ps(a.value, b.value, _CMP_EQ_OS);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_cmp_ps(a.value, b.value, _CMP_EQ_OS);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_from_mask_ps(_mm512_cmpeq_ps_mask(a.value, b.value));
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_cmpeq_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_cmpeq_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_from_mask_epi32(_mm512_cmpeq_epi32_mask(a.value, b.value));
            KAIXO_SIMD_CASE(SSE2, 128, short) return _mm_cmpeq_epi16(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, short) return _mm256_cmpeq_epi16(a.value, b.value);
            //KAIXO_SIMD_CASE(AVX512F, 512, short) 
            KAIXO_SIMD_BASE return a.value == b.value;
        }
            
        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator!=(const basic_simd& a, const basic_simd& b)  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_cmpneq_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 128, float) return _mm_cmp_ps(a.value, b.value, _CMP_NEQ_OS);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_cmp_ps(a.value, b.value, _CMP_NEQ_OS);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_from_mask_ps(_mm512_cmpneq_ps_mask(a.value, b.value));
            KAIXO_SIMD_CASE_ALL(128, int) return ~(a == b);
            KAIXO_SIMD_CASE_ALL(256, int) return ~(a == b);
            KAIXO_SIMD_CASE_ALL(512, int) return ~(a == b);
            KAIXO_SIMD_CASE_ALL(128, short) return ~(a == b);
            KAIXO_SIMD_CASE_ALL(256, short) return ~(a == b);
            KAIXO_SIMD_CASE_ALL(512, short) return ~(a == b);
            KAIXO_SIMD_BASE return a.value != b.value;
        }
            
        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator>(const basic_simd& a, const basic_simd& b)  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_cmpgt_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 128, float) return _mm_cmp_ps(a.value, b.value, _CMP_GT_OS);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_cmp_ps(a.value, b.value, _CMP_GT_OS);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_from_mask_ps(_mm512_cmp_ps_mask(a.value, b.value, _CMP_GT_OS));
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_cmpgt_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_cmpgt_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_cmpgt_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(SSE2, 128, short) return _mm_cmpgt_epi16(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, short) return _mm256_cmpgt_epi16(a.value, b.value);
            //KAIXO_SIMD_CASE(AVX512F, 512, short)
            KAIXO_SIMD_BASE return a.value > b.value;
        }
            
        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator<(const basic_simd& a, const basic_simd& b)  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_cmplt_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 128, float) return _mm_cmp_ps(a.value, b.value, _CMP_LT_OS);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_cmp_ps(a.value, b.value, _CMP_LT_OS);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_from_mask_ps(_mm512_cmp_ps_mask(a.value, b.value, _CMP_LT_OS));
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_cmplt_epi32(a.value, b.value);
            KAIXO_SIMD_CASE_ALL(256, int) return (b > a) | (b > a);
            KAIXO_SIMD_CASE_ALL(512, int) return (b > a) | (b > a);
            KAIXO_SIMD_CASE(SSE2, 128, short) return _mm_cmplt_epi16(a.value, b.value);
            KAIXO_SIMD_CASE_ALL(256, short) return (b > a) | (b > a);
            KAIXO_SIMD_CASE_ALL(512, short) return (b > a) | (b > a);
            KAIXO_SIMD_BASE return a.value < b.value;
        }
            
        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator>=(const basic_simd& a, const basic_simd& b)  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_cmpge_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 128, float) return _mm_cmp_ps(a.value, b.value, _CMP_GE_OS);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_cmp_ps(a.value, b.value, _CMP_GE_OS);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_from_mask_ps(_mm512_cmp_ps_mask(a.value, b.value, _CMP_GE_OS));
            KAIXO_SIMD_CASE_ALL(128, int) return (a > b) | (a == b);
            KAIXO_SIMD_CASE_ALL(256, int) return (a > b) | (a == b);
            KAIXO_SIMD_CASE_ALL(512, int) return (a > b) | (a == b);
            KAIXO_SIMD_CASE_ALL(128, short) return (a > b) | (a == b);
            KAIXO_SIMD_CASE_ALL(256, short) return (a > b) | (a == b);
            KAIXO_SIMD_CASE_ALL(512, short) return (a > b) | (a == b);
            KAIXO_SIMD_BASE return a.value >= b.value;
        }
            
        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator<=(const basic_simd& a, const basic_simd& b)  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_cmple_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 128, float) return _mm_cmp_ps(a.value, b.value, _CMP_LE_OS);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_cmp_ps(a.value, b.value, _CMP_LE_OS);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_from_mask_ps(_mm512_cmp_ps_mask(a.value, b.value, _CMP_LE_OS));
            KAIXO_SIMD_CASE_ALL(128, int) return (a < b) | (a == b);
            KAIXO_SIMD_CASE_ALL(256, int) return (a < b) | (a == b);
            KAIXO_SIMD_CASE_ALL(512, int) return (a < b) | (a == b);
            KAIXO_SIMD_CASE_ALL(128, short) return (a < b) | (a == b);
            KAIXO_SIMD_CASE_ALL(256, short) return (a < b) | (a == b);
            KAIXO_SIMD_CASE_ALL(512, short) return (a < b) | (a == b);
            KAIXO_SIMD_BASE return a.value <= b.value;
        }

        // ------------------------------------------------

        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator<<(const basic_simd& a, const basic_simd& b)  {
            KAIXO_SIMD_CASE(AVX2, 128, int) return _mm_sllv_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_sllv_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_sllv_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512BW | AVX512VL, 128, short) return _mm_sllv_epi16(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512BW | AVX512VL, 256, short) return _mm256_sllv_epi16(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512BW, 512, short) return _mm512_sllv_epi16(a.value, b.value);
            KAIXO_SIMD_BASE_TYPE(int) return a.value << b.value;
        }

        template<std::integral Arg>
        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator<<(const basic_simd& a, Arg b)  {
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_slli_epi32(a.value, static_cast<int>(b));
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_slli_epi32(a.value, static_cast<int>(b));
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_slli_epi32(a.value, static_cast<unsigned int>(b));
            KAIXO_SIMD_CASE(SSE2, 128, short) return _mm_slli_epi16(a.value, static_cast<int>(b));
            KAIXO_SIMD_CASE(AVX2, 256, short) return _mm256_slli_epi16(a.value, static_cast<int>(b));
            KAIXO_SIMD_CASE(AVX512BW, 512, short) return _mm512_slli_epi16(a.value, static_cast<unsigned int>(b));
            KAIXO_SIMD_BASE_TYPE(int) return a.value << b;
        }

        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator>>(const basic_simd& a, const basic_simd& b)  {
            KAIXO_SIMD_CASE(AVX2, 128, int) return _mm_srlv_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_srlv_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_srlv_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512BW | AVX512VL, 128, short) return _mm_srlv_epi16(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512BW | AVX512VL, 256, short) return _mm256_srlv_epi16(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512BW, 512, short) return _mm512_srlv_epi16(a.value, b.value);
            KAIXO_SIMD_BASE_TYPE(int) return a.value >> b.value;
        }

        template<std::integral Arg>
        friend KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator>>(const basic_simd& a, Arg b)  {
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_srli_epi32(a.value, static_cast<int>(b));
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_srli_epi32(a.value, static_cast<int>(b));
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_srli_epi32(a.value, static_cast<unsigned int>(b));
            KAIXO_SIMD_CASE(SSE2, 128, short) return _mm_srli_epi16(a.value, static_cast<int>(b));
            KAIXO_SIMD_CASE(AVX2, 256, short) return _mm256_srli_epi16(a.value, static_cast<int>(b));
            KAIXO_SIMD_CASE(AVX512BW, 512, short) return _mm512_srli_epi16(a.value, static_cast<int>(b));
            KAIXO_SIMD_BASE_TYPE(int) return a.value >> b;
        }

        // ------------------------------------------------

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator~() const  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_xor_ps(value, _mm_set1_ps(std::bit_cast<float>(0xFFFFFFFF)));
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_xor_ps(value, _mm256_set1_ps(std::bit_cast<float>(0xFFFFFFFF)));
            KAIXO_SIMD_CASE(AVX512DQ, 512, float) return _mm512_xor_ps(value, _mm512_set1_ps(std::bit_cast<float>(0xFFFFFFFF)));
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_xor_si128(value, _mm_set1_epi32(std::bit_cast<int>(0xFFFFFFFF)));
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_xor_si256(value, _mm256_set1_epi32(std::bit_cast<int>(0xFFFFFFFF)));
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_xor_si512(value, _mm512_set1_epi32(std::bit_cast<int>(0xFFFFFFFF)));
            KAIXO_SIMD_CASE(SSE2, 128, short) return _mm_xor_si128(value, _mm_set1_epi32(std::bit_cast<int>(0xFFFFFFFF)));
            KAIXO_SIMD_CASE(AVX2, 256, short) return _mm256_xor_si256(value, _mm256_set1_epi32(std::bit_cast<int>(0xFFFFFFFF)));
            KAIXO_SIMD_CASE(AVX512F, 512, short) return _mm512_xor_si512(value, _mm512_set1_epi32(std::bit_cast<int>(0xFFFFFFFF)));
            KAIXO_SIMD_BASE return !value;
        }
        
        KAIXO_INLINE basic_simd KAIXO_VECTORCALL operator-() const  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_xor_ps(value, _mm_set1_ps(-0.f));
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_xor_ps(value, _mm256_set1_ps(-0.f));
            KAIXO_SIMD_CASE(AVX512DQ, 512, float) return _mm512_xor_ps(value, _mm512_set1_ps(-0.f));
            KAIXO_SIMD_BASE return 0 - *this;
        }

        // ------------------------------------------------

        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL min(const basic_simd& a, const basic_simd& b)  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_min_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_min_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_min_ps(a.value, b.value);
            KAIXO_SIMD_CASE(SSE4_1, 128, int) return _mm_min_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_min_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_min_epi32(a.value, b.value);
            KAIXO_SIMD_BASE return a.value < b.value ? a.value : b.value;
        }

        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL max(const basic_simd& a, const basic_simd& b)  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_max_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_max_ps(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_max_ps(a.value, b.value);
            KAIXO_SIMD_CASE(SSE4_1, 128, int) return _mm_max_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_max_epi32(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_max_epi32(a.value, b.value);
            KAIXO_SIMD_BASE return a.value > b.value ? a.value : b.value;
        }

        // ------------------------------------------------

        // (a * b) + c
        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL fmadd(const basic_simd& a, const basic_simd& b, const basic_simd& c)  {
            KAIXO_SIMD_CASE(FMA, 128, float) return _mm_fmadd_ps(a.value, b.value, c.value);
            KAIXO_SIMD_CASE(FMA, 256, float) return _mm256_fmadd_ps(a.value, b.value, c.value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_fmadd_ps(a.value, b.value, c.value);
            KAIXO_SIMD_CASE_ALL(128, float) return (a * b) + c;
            KAIXO_SIMD_CASE_ALL(256, float) return (a * b) + c;
            KAIXO_SIMD_CASE_ALL(512, float) return (a * b) + c;
            KAIXO_SIMD_BASE return (a.value * b.value) + c.value;
        }
        
        // (a * b) - c
        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL fmsub(const basic_simd& a, const basic_simd& b, const basic_simd& c)  {
            KAIXO_SIMD_CASE(FMA, 128, float) return _mm_fmsub_ps(a.value, b.value, c.value);
            KAIXO_SIMD_CASE(FMA, 256, float) return _mm256_fmsub_ps(a.value, b.value, c.value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_fmsub_ps(a.value, b.value, c.value);
            KAIXO_SIMD_CASE_ALL(128, float) return (a * b) - c;
            KAIXO_SIMD_CASE_ALL(256, float) return (a * b) - c;
            KAIXO_SIMD_CASE_ALL(512, float) return (a * b) - c;
            KAIXO_SIMD_BASE return (a.value * b.value) - c.value;
        }

        // ------------------------------------------------

        // SSE3 | SSE
        KAIXO_INLINE static base KAIXO_VECTORCALL _sum_general(simd_type a)  {
            base vals[elements];
            basic_simd{ a }.store(vals);
            return std::accumulate(vals, vals + elements, base{});
        }
            
        // SSE3 | SSE
        KAIXO_INLINE static base KAIXO_VECTORCALL _sum_128_f(__m128 a)  {
            __m128 shuf = _mm_movehdup_ps(a);
            __m128 sums = _mm_add_ps(a, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            return _mm_cvtss_f32(sums);
        }

        // AVX | SSE
        KAIXO_INLINE static base KAIXO_VECTORCALL _sum_256_f(__m256 a)  {
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
        KAIXO_INLINE static base KAIXO_VECTORCALL _sum_512_f(__m512 a)  {
            __m256 v0 = _mm512_castps512_ps256(a);
            __m256 v1 = _mm512_extractf32x8_ps(a, 1);
            __m256 x0 = _mm256_add_ps(v0, v1);
            return _sum_256_f(x0);
        }

        KAIXO_INLINE base KAIXO_VECTORCALL sum() const  {
            KAIXO_SIMD_CASE(SSE3 | SSE, 128, float) return _sum_128_f(value);
            KAIXO_SIMD_CASE(AVX | SSE, 256, float) return _sum_256_f(value);
            KAIXO_SIMD_CASE(AVX512F | AVX512DQ | AVX | SSE, 512, float) return _sum_512_f(value);
            KAIXO_SIMD_CASE_ALL(128, float) return _sum_general(value);
            KAIXO_SIMD_CASE_ALL(256, float) return _sum_general(value);
            KAIXO_SIMD_CASE_ALL(512, float) return _sum_general(value);
            KAIXO_SIMD_CASE_ALL(128, int) return _sum_general(value);
            KAIXO_SIMD_CASE_ALL(256, int) return _sum_general(value);
            KAIXO_SIMD_CASE_ALL(512, int) return _sum_general(value);
            KAIXO_SIMD_CASE_ALL(128, short) return _sum_general(value);
            KAIXO_SIMD_CASE_ALL(256, short) return _sum_general(value);
            KAIXO_SIMD_CASE_ALL(512, short) return _sum_general(value);
            KAIXO_SIMD_BASE return value;
        }

        // ------------------------------------------------
        
        KAIXO_INLINE basic_simd KAIXO_VECTORCALL reverse() const  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_shuffle_ps(value, value, _MM_SHUFFLE(0, 1, 2, 3));
            KAIXO_SIMD_CASE(AVX, 256, float) {
                auto v = _mm256_permute2f128_ps(value, value, 0x01);
                return _mm256_shuffle_ps(v, v, _MM_SHUFFLE(0, 1, 2, 3));
            }
            KAIXO_SIMD_BASE_TYPE(float) return value;
        }

        // ------------------------------------------------

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL trunc() const  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_trunc_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_trunc_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_trunc_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::trunc(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL floor() const  {
            KAIXO_SIMD_CASE(SSE4_1, 128, float) return _mm_floor_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_floor_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_floor_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::floor(value);
        }
        
        KAIXO_INLINE basic_simd KAIXO_VECTORCALL ceil() const  {
            KAIXO_SIMD_CASE(SSE4_1, 128, float) _mm_ceil_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_ceil_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_ceil_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::ceil(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL round() const  {
            KAIXO_SIMD_CASE(AVX512F | AVX512VL, 128, float) return _mm_roundscale_ps(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            KAIXO_SIMD_CASE(AVX512F | AVX512VL, 256, float) return _mm256_roundscale_ps(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            KAIXO_SIMD_CASE(AVX512F | AVX512VL, 512, float) return _mm512_roundscale_ps(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            KAIXO_SIMD_BASE_TYPE(float) return std::round(value);
        }

        // ------------------------------------------------

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL sign() const  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_or_ps(_mm_set1_ps(1.f), _mm_and_ps(value, _mm_set1_ps(-0.f)));
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_or_ps(_mm256_set1_ps(1.f), _mm256_and_ps(value, _mm256_set1_ps(-0.f)));
            KAIXO_SIMD_CASE(AVX512F | AVX512DQ, 512, float) return _mm512_or_ps(_mm512_set1_ps(1.f), _mm512_and_ps(value, _mm512_set1_ps(-0.f)));
            KAIXO_SIMD_BASE_TYPE(float) return value < 0 ? -1 : 1;
        }

        // ------------------------------------------------

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL log() const  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_log_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_log_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_log_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::log(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL log2() const  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_log2_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_log2_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_log2_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::log2(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL log10() const  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_log10_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_log10_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_log10_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::log10(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL sqrt() const  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_sqrt_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_sqrt_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_sqrt_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::sqrt(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL cbrt() const  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_cbrt_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_cbrt_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_cbrt_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::cbrt(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL exp() const  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_exp_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_exp_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_exp_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::exp(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL exp2() const  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_exp2_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_exp2_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_exp2_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::exp2(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL exp10() const  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_exp10_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_exp10_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return _mm512_exp10_ps(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL tanh() const  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_tanh_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_tanh_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_tanh_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::tanh(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL abs() const  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_andnot_ps(_mm_set1_ps(-0.0), value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_andnot_ps(_mm256_set1_ps(-0.0), value);
            KAIXO_SIMD_CASE(AVX512F | AVX512DQ, 512, float) return _mm512_andnot_ps(_mm512_set1_ps(-0.0), value);
            KAIXO_SIMD_BASE_TYPE(float) return std::abs(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL cos() const  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_cos_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_cos_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_cos_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::cos(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL cosh() const {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_cosh_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_cosh_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_cosh_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::cosh(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL sin() const {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_sin_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_sin_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_sin_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::sin(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL sinh() const {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_sinh_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_sinh_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_sinh_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return std::sinh(value);
        }

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL pow(const basic_simd& b) const {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_pow_ps(value, b.value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_pow_ps(value, b.value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_pow_ps(value, b.value);
            KAIXO_SIMD_BASE_TYPE(float) return std::pow(value, b.value);
        }

        // ------------------------------------------------

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL rcp() const {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_rcp_ps(value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_rcp_ps(value);
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_rcp14_ps(value);
            KAIXO_SIMD_BASE_TYPE(float) return 1.f / value;
        }

        // ------------------------------------------------

        template<class To>
        KAIXO_INLINE basic_simd<To, Bits, Instructions> KAIXO_VECTORCALL cast() const {
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
        KAIXO_INLINE basic_simd<To, Bits, Instructions> KAIXO_VECTORCALL reinterpret() const {
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

        KAIXO_INLINE basic_simd<float, Bits, Instructions> KAIXO_VECTORCALL gather(float const* data) const {
            KAIXO_SIMD_CASE(AVX2, 128, int) return _mm_i32gather_ps(data, value, sizeof(float));
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_i32gather_ps(data, value, sizeof(float));
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_i32gather_ps(value, data, sizeof(float));
            KAIXO_SIMD_BASE_TYPE(int) return data[value];
        }

        KAIXO_INLINE basic_simd<int, Bits, Instructions> KAIXO_VECTORCALL gather(int const* data) const {
            KAIXO_SIMD_CASE(AVX2, 128, int) return _mm_i32gather_epi32(data, value, sizeof(int));
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_i32gather_epi32(data, value, sizeof(int));
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_i32gather_epi32(value, data, sizeof(int));
            KAIXO_SIMD_BASE_TYPE(int) return data[value];
        }

        // ------------------------------------------------
        
        KAIXO_INLINE basic_simd KAIXO_VECTORCALL iff(auto then, auto otherwise) const {
            KAIXO_SIMD_BASE return value ? then() : otherwise();
            else return *this & then() | ~*this & otherwise();
        }

        // ------------------------------------------------
        
        KAIXO_INLINE basic_simd KAIXO_VECTORCALL fast_abs() const {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_andnot_ps(_mm_set1_ps(-0.0), value);
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_andnot_ps(_mm256_set1_ps(-0.0), value);
            KAIXO_SIMD_CASE(AVX512F | AVX512DQ, 512, float) return _mm512_andnot_ps(_mm512_set1_ps(-0.0), value);
            KAIXO_SIMD_BASE_TYPE(float) return value > 0 ? value : -value;
        }
        
        // Requires -0.5 < value < 0.5, outputs sin(value * 2 * pi)
        KAIXO_INLINE basic_simd KAIXO_VECTORCALL fast_normalized_sin() const {
            KAIXO_SIMD_CASE(SSE | FMA, 128, float) {
                // value * ((-16.f * abs(value)) + 8.f)
                auto _inter1 = _mm_fmadd_ps(_mm_set1_ps(-16.f), _mm_andnot_ps(_mm_set1_ps(-0.0), value), _mm_set1_ps(8.f));
                auto _inter2 = _mm_mul_ps(_inter1, value);
                // _inter2 * ((0.224f * abs(_inter2)) + 0.776f)
                auto _inter3 = _mm_fmadd_ps(_mm_set1_ps(0.224f), _mm_andnot_ps(_mm_set1_ps(-0.0), _inter2), _mm_set1_ps(0.776f));
                return _mm_mul_ps(_inter3, _inter2);
            }
            else KAIXO_SIMD_CASE(SSE, 128, float) {
                // value * (8.f - (16.f * abs(value)))
                auto _inter1 = _mm_sub_ps(_mm_set1_ps(8.f), _mm_mul_ps(_mm_set1_ps(16.f), _mm_andnot_ps(_mm_set1_ps(-0.0), value)));
                auto _inter2 = _mm_mul_ps(_inter1, value);
                // _inter2 * (0.776f + (0.224f * abs(_inter2)))
                auto _inter3 = _mm_add_ps(_mm_set1_ps(0.776f), _mm_mul_ps(_mm_set1_ps(0.224f), _mm_andnot_ps(_mm_set1_ps(-0.0), _inter2)));
                return _mm_mul_ps(_inter3, _inter2);
            }
            KAIXO_SIMD_CASE(AVX | FMA, 256, float) {
                // value * ((-16.f * abs(value)) + 8.f)
                auto _inter1 = _mm256_fmadd_ps(_mm256_set1_ps(-16.f), _mm256_andnot_ps(_mm256_set1_ps(-0.0), value), _mm256_set1_ps(8.f));
                auto _inter2 = _mm256_mul_ps(_inter1, value);
                // _inter2 * ((0.224f * abs(_inter2)) + 0.776f)
                auto _inter3 = _mm256_fmadd_ps(_mm256_set1_ps(0.224f), _mm256_andnot_ps(_mm256_set1_ps(-0.0), _inter2), _mm256_set1_ps(0.776f));
                return _mm256_mul_ps(_inter3, _inter2);
            }
            else KAIXO_SIMD_CASE(AVX, 256, float) {
                // value * (8.f - (16.f * abs(value)))
                auto _inter1 = _mm256_sub_ps(_mm256_set1_ps(8.f), _mm256_mul_ps(_mm256_set1_ps(16.f), _mm256_andnot_ps(_mm256_set1_ps(-0.0), value)));
                auto _inter2 = _mm256_mul_ps(_inter1, value);
                // _inter2 * (0.776f + (0.224f * abs(_inter2)))
                auto _inter3 = _mm256_add_ps(_mm256_set1_ps(0.776f), _mm256_mul_ps(_mm256_set1_ps(0.224f), _mm256_andnot_ps(_mm256_set1_ps(-0.0), _inter2)));
                return _mm256_mul_ps(_inter3, _inter2);
            }
            KAIXO_SIMD_CASE(AVX512F | AVX512DQ, 512, float) {
                // value * ((-16.f * abs(value)) + 8.f)
                auto _inter1 = _mm512_fmadd_ps(_mm512_set1_ps(-16.f), _mm512_andnot_ps(_mm512_set1_ps(-0.0), value), _mm512_set1_ps(8.f));
                auto _inter2 = _mm512_mul_ps(_inter1, value);
                // _inter2 * ((0.224f * abs(_inter2)) + 0.776f)
                auto _inter3 = _mm512_fmadd_ps(_mm512_set1_ps(0.224f), _mm512_andnot_ps(_mm512_set1_ps(-0.0), _inter2), _mm512_set1_ps(0.776f));
                return _mm512_mul_ps(_inter3, _inter2);
            }
            KAIXO_SIMD_BASE_TYPE(float) { 
                auto approx = value * (8.0f - 16.0f * (value > 0 ? value : -value));
                return approx * (0.776f + 0.224f * (approx > 0 ? approx : -approx));
            }
        }
        
        KAIXO_INLINE basic_simd KAIXO_VECTORCALL fmod1() const {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_sub_ps(value, _mm_trunc_ps(value));
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_sub_ps(value, _mm256_trunc_ps(value));
            KAIXO_SIMD_CASE(AVX512F, 512, float) return _mm512_sub_ps(value, _mm512_trunc_ps(value));
            KAIXO_SIMD_BASE_TYPE(float) return value - static_cast<float>(static_cast<std::int64_t>(value));
        }

        // ------------------------------------------------

        KAIXO_INLINE static basic_simd<short, bits, instructions> KAIXO_VECTORCALL max_epi16(const basic_simd& a, const basic_simd& b) {
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_max_epi16(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_max_epi16(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512BW, 512, int) return _mm512_max_epi16(a.value, b.value);
            KAIXO_SIMD_CASE(SSE2, 128, short) return _mm_max_epi16(a.value, b.value);
            KAIXO_SIMD_CASE(AVX2, 256, short) return _mm256_max_epi16(a.value, b.value);
            KAIXO_SIMD_CASE(AVX512BW, 512, short) return _mm512_max_epi16(a.value, b.value);
            KAIXO_SIMD_BASE_TYPE(int) {
                const int a1 = a.value & 0x0000FFFF;
                const int a2 = a.value & 0xFFFF0000;
                const int b1 = a.value & 0x0000FFFF;
                const int b2 = b.value & 0xFFFF0000;
                return (a1 > b1 ? a1 : b1) + (a2 > b2 ? a2 : b2);
            }
        }

        // ------------------------------------------------

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL complex_mul(const basic_simd b) const {
            KAIXO_SIMD_CASE(AVX | FMA, 256, float) {
                // a = [r0,i0,r1,i1,r2,i2,r3,i3]
                // b = [wr0,wi0,wr1,wi1,wr2,wi2,wr3,wi3]
                __m256 ar = _mm256_shuffle_ps(value, value, 0xA0); // [r0,r1,r2,r3]
                __m256 ai = _mm256_shuffle_ps(value, value, 0xF5); // [i0,i1,i2,i3]
                __m256 br = _mm256_shuffle_ps(b.value, b.value, 0xA0);
                __m256 bi = _mm256_shuffle_ps(b.value, b.value, 0xF5);

                __m256 real = _mm256_fmsub_ps(ar, br, _mm256_mul_ps(ai, bi)); // ar*br - ai*bi
                __m256 imag = _mm256_fmadd_ps(ar, bi, _mm256_mul_ps(ai, br)); // ar*bi + ai*br

                // interleave real/imag back
                __m256 ri_lo = _mm256_unpacklo_ps(real, imag); // r0,i0,r1,i1
                __m256 ri_hi = _mm256_unpackhi_ps(real, imag); // r2,i2,r3,i3
                return _mm256_permute2f128_ps(ri_lo, ri_hi, 0x20); // combine
            }
        }

        // ------------------------------------------------

        KAIXO_INLINE basic_simd KAIXO_VECTORCALL _random_value1() const  {
            KAIXO_SIMD_CASE(SSE, 128, float) return _mm_set1_ps(std::bit_cast<float>(0xFFFFFFFF));
            KAIXO_SIMD_CASE(AVX, 256, float) return _mm256_set1_ps(std::bit_cast<float>(0xFFFFFFFF));
            KAIXO_SIMD_CASE(AVX512DQ, 512, float) return _mm512_set1_ps(std::bit_cast<float>(0xFFFFFFFF));
            KAIXO_SIMD_CASE(SSE2, 128, int) return _mm_set1_epi32(std::bit_cast<int>(0xFFFFFFFF));
            KAIXO_SIMD_CASE(AVX2, 256, int) return _mm256_set1_epi32(std::bit_cast<int>(0xFFFFFFFF));
            KAIXO_SIMD_CASE(AVX512F, 512, int) return _mm512_set1_epi32(std::bit_cast<int>(0xFFFFFFFF));
            KAIXO_SIMD_BASE return !value;
        }

        KAIXO_INLINE static basic_simd KAIXO_VECTORCALL noise() {
            KAIXO_SIMD_CASE(AVX2, 256, int) {
                thread_local struct {
                    __m256i part1{ .m256i_u64 { 0xe7a1b8b86f088db3, 0x383db7c7e2c8d0c4, 0x9f1cbdee9c6b99e1, 0x5e4d910b0a57edc5, } };
                    __m256i part2{ .m256i_u64 { 0x4f61df2c3ca81ac1, 0x2719b1b6a8c91597, 0xc64fd97a0d9bf86a, 0x492ed6ecf82b7d4b, } };
                } state;

                __m256i s1 = state.part1;
                const __m256i s0 = state.part2;
                state.part1 = state.part2;
                s1 = _mm256_xor_si256(state.part2, _mm256_slli_epi64(state.part2, 23));
                state.part2 = _mm256_xor_si256(
                    _mm256_xor_si256(_mm256_xor_si256(s1, s0),
                        _mm256_srli_epi64(s1, 18)), _mm256_srli_epi64(s0, 5));
                return _mm256_add_epi64(state.part2, s0);
            }

            KAIXO_SIMD_CASE(AVX | AVX2, 256, float) {
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
                    _mm256_set1_ps(static_cast<float>(std::numeric_limits<std::uint32_t>::max())));
            }
        }

        // ------------------------------------------------

    };

    // ------------------------------------------------

#undef KAIXO_SIMD_CALL
#endif

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

    using simd_128 = basic_simd<float, 128, instruction_set::SSE | instruction_set::SSE2 | instruction_set::SSE3 | instruction_set::SSE4_1 | instruction_set::FMA>;
    using simd_256 = basic_simd<float, 256, simd_128::instructions | instruction_set::AVX | instruction_set::AVX2>;
    using simd_512 = basic_simd<float, 512, simd_256::instructions | instruction_set::AVX512F | instruction_set::AVX512DQ | instruction_set::AVX512VL>;
    
    using simd_128i = basic_simd<int, 128, simd_128::instructions>;
    using simd_256i = basic_simd<int, 256, simd_256::instructions>;
    using simd_512i = basic_simd<int, 512, simd_512::instructions>;

    // ------------------------------------------------
    
#ifdef ARM_ARCHITECTURE
    template<class Ty>
    KAIXO_INLINE decltype(auto) choose_simd_path(auto lambda) {
        return lambda.template operator()<Ty>();
    }
#else
    template<class Ty>
    KAIXO_INLINE decltype(auto) choose_simd_path(auto lambda) {
        if ((~supported_instruction_sets & simd_512::instructions) == 0) return lambda.template operator()<basic_simd<Ty, 512, simd_512::instructions>>();
        if ((~supported_instruction_sets & simd_256::instructions) == 0) return lambda.template operator()<basic_simd<Ty, 256, simd_256::instructions>>();
        if ((~supported_instruction_sets & simd_128::instructions) == 0) return lambda.template operator()<basic_simd<Ty, 128, simd_128::instructions>>();
        return lambda.template operator()<Ty>();
    }
#endif

    // ------------------------------------------------

    template<class Ty> struct base : std::type_identity<Ty> {};
    template<is_simd Ty> struct base<Ty> : std::type_identity<typename Ty::base> {};
    template<class Ty> using base_t = typename base<Ty>::type;

    // ------------------------------------------------
    
    template<class Ty, class To> struct change_base : std::type_identity<To> {};
    template<is_simd Ty, class To> struct change_base<Ty, To> : std::type_identity<basic_simd<To, Ty::bits, Ty::instructions>> {};
    template<class Ty, class To> using change_base_t = typename change_base<Ty, To>::type;

    // ------------------------------------------------
    
    template<class Ty> struct simd_elements : std::integral_constant<std::size_t, 1> {};
    template<is_simd Ty> struct simd_elements<Ty> : std::integral_constant<std::size_t, Ty::elements> {};
    template<class Ty> constexpr std::size_t simd_elements_v = simd_elements<Ty>::value;

    // ------------------------------------------------

    template<class Type>
    KAIXO_INLINE auto KAIXO_VECTORCALL max_epi16(const Type& a, const Type& b)  { 
        if constexpr (std::same_as<Type, int>) {
            const int a1 = a & 0x0000FFFF;
            const int a2 = a & 0xFFFF0000;
            const int b1 = a & 0x0000FFFF;
            const int b2 = b & 0xFFFF0000;
            return (a1 > b1 ? a1 : b1) + (a2 > b2 ? a2 : b2);
        } else {
            return Type::max_epi16(a, b);
        }
    }

    // ------------------------------------------------
    
    template<class A, class B>
    KAIXO_INLINE decltype(auto) KAIXO_VECTORCALL simd_min(const A& a, const B& b)  {
        using SimdType = std::conditional_t<is_simd<A>, A, B>;
        if constexpr (!is_simd<SimdType>) return a < b ? a : b;
        else return SimdType::min(a, b);
    }
    
    template<class A, class B>
    KAIXO_INLINE decltype(auto) KAIXO_VECTORCALL simd_max(const A& a, const B& b)  {
        using SimdType = std::conditional_t<is_simd<A>, A, B>;
        if constexpr (!is_simd<SimdType>) return a > b ? a : b;
        else return SimdType::max(a, b);
    }

    // ------------------------------------------------
    
    template<class Type>
    KAIXO_INLINE Type KAIXO_VECTORCALL fmadd(const Type& a, const Type& b, const Type& c)  { 
        if constexpr (!is_simd<Type>) return (a * b) + c;
        else return Type::fmadd(a, b, c);
    }
    
    template<class Type>
    KAIXO_INLINE Type KAIXO_VECTORCALL fmsub(const Type& a, const Type& b, const Type& c)  { 
        if constexpr (!is_simd<Type>) return (a * b) - c;
        else return Type::fmsub(a, b, c);
    }

    // ------------------------------------------------
    
    template<class Type>
    KAIXO_INLINE Type KAIXO_VECTORCALL rcp(const Type& a)  { 
        if constexpr (!is_simd<Type>) return 1.f / a;
        else return a.rcp();
    }

    // ------------------------------------------------

    template<class Type>
    KAIXO_INLINE Type KAIXO_VECTORCALL setincr()  {
        if constexpr (!is_simd<Type>) return 0;
        else return Type::setincr();
    }
    
    template<class Type, auto Default = 0, class ...Tys>
    KAIXO_INLINE Type KAIXO_VECTORCALL setfirst(Tys... values)  {
        if constexpr (!is_simd<Type>) return (values, ...);
        else return Type::template setfirst<Default>(values...);
    }
    
    template<class Type>
    KAIXO_INLINE Type KAIXO_VECTORCALL loadu(base_t<Type> const* ptr, std::size_t index)  {
        if constexpr (!is_simd<Type>) return ptr[index];
        else return Type(ptr + index);
    }

    template<class Type>
    KAIXO_INLINE Type KAIXO_VECTORCALL load(base_t<Type> const* ptr, std::size_t index)  {
        if constexpr (!is_simd<Type>) return ptr[index];
        else return Type::load(ptr + index);
    }

    template<class Type>
    KAIXO_INLINE void KAIXO_VECTORCALL storeu(base_t<Type>* ptr, const Type& value)  {
        if constexpr (!is_simd<Type>) *ptr = value;
        else value.storeu(ptr);
    }

    template<class Type>
    KAIXO_INLINE void KAIXO_VECTORCALL store(base_t<Type>* ptr, const Type& value)  {
        if constexpr (!is_simd<Type>) *ptr = value;
        else value.store(ptr);
    }

    template<class Type>
    KAIXO_INLINE void KAIXO_VECTORCALL stream(base_t<Type>* ptr, const Type& value)  {
        if constexpr (!is_simd<Type>) *ptr = value;
        else value.stream(ptr);
    }

    template<class Type, std::convertible_to<Type> B>
    KAIXO_INLINE Type KAIXO_VECTORCALL bool_and(const Type& condition, B value)  {
        if constexpr (!is_simd<Type>) return condition * value;
        else return condition & value;
    }

    template<class Type, class A, class B>
    KAIXO_INLINE Type KAIXO_VECTORCALL iff(const Type& condition, const A& then, const B& otherwise)  {
        if constexpr (std::invocable<A> && std::invocable<B>) {
            if constexpr (!is_simd<Type>) return condition ? then() : otherwise();
            else return condition & then() | ~condition & otherwise();
        } else if constexpr (std::invocable<A>) {
            if constexpr (!is_simd<Type>) return condition ? then() : otherwise;
            else return condition & then() | ~condition & otherwise;
        } else if constexpr (std::invocable<B>) {
            if constexpr (!is_simd<Type>) return condition ? then : otherwise();
            else return condition & then | ~condition & otherwise();
        } else {
            if constexpr (!is_simd<Type>) return condition ? then : otherwise;
            else return condition & then | ~condition & otherwise;
        }
    }

    // Multiply with 1 or -1
    template<class Type, std::convertible_to<Type> B>
    KAIXO_INLINE Type KAIXO_VECTORCALL mul1(const Type& condition, B value)  {
        if constexpr (!is_simd<Type>) return condition * value;
        else return condition ^ ((-0.f) & value); // Toggle sign bit if value has sign bit
    };

    template<class To, class Type>
    KAIXO_INLINE auto KAIXO_VECTORCALL cast(const Type& v)  {
        if constexpr (!is_simd<Type>) return (To)v;
        else return v.template cast<To>();
    }

    template<class To, class Type>
    KAIXO_INLINE auto KAIXO_VECTORCALL reinterpret(const Type& v)  {
        if constexpr (!is_simd<Type>) return std::bit_cast<To>(v);
        else return v.template reinterpret<To>();
    }

    template<class Type, class Ptr>
    KAIXO_INLINE decltype(auto) KAIXO_VECTORCALL gather(Ptr* data, const Type& index)  {
        if constexpr (!is_simd<Type>) return data[(std::int64_t)index];
        else return index.gather(data);
    }

    template<class Type>
    KAIXO_INLINE base_t<Type> KAIXO_VECTORCALL sum(const Type& value)  {
        if constexpr (!is_simd<Type>) return value;
        else return value.sum();
    }

    // ------------------------------------------------

}