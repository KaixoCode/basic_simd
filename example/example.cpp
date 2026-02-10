
#include "basic_simd.hpp"




int main() {


    using namespace::kaixo;

    using simd_t = simd<float, 0>;

    simd_t a{ 1 };
    simd_t b{ 1 };
    simd_t c{ 3 };

    auto res12 = a == b;

    simd_t::blend(a == b, a, b);

    float data[2];

    simd_t val = simd_t::load(data + 0);

    simd_t::store(data, val);

    simd_256 p;
    simd_256i q;

    p.sincos();

    simd_256::sincos(p);

    q.gather(data);

    p.floor();

    simd_256::blend(simd_256::is_negative(p), p, p);

    p += 1;

    auto aefa = p.to_int();

    simd<int, 0> x;
    simd<int, 0> y;

    simd<int, 0> w = x << y;

    auto res = a == b & c;

    

    simd_256::elements;


    //using namespace kaixo;
    //using enum instruction_set;
    //
    //choose_simd_path<float>([]<class simd_type>() {
    //
    //    float data[16]{ 
    //        1, 1, 2, 2, 3, 4, 5, 6,
    //        1, 1, 4, 2, 5, 4, 7, 6,
    //    };
    //
    //    simd_type value1 = load<simd_type>(data, 0);
    //    simd_type value2 = load<simd_type>(data, simd_elements_v<simd_type>);
    //
    //    auto res = value1 == value2;
    //    auto val = iff(res, [] { return 1.f; }, [] { return 0.f; });
    //
    //    return 0;
    //});

    return 0;

}