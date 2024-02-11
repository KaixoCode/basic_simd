
#include "simd.hpp"

int main() {

    using namespace kaixo;
    using enum instruction_set;

    choose_simd_path([]<class simd_type>() {

        float data[16]{ 
            1, 1, 2, 2, 3, 4, 5, 6,
            1, 1, 4, 2, 5, 4, 7, 6,
        };

        simd_type value1 = data;
        simd_type value2 = data + simd_type::elements;

        value1.instructions;

        auto res = value1 == value2;
        auto val = res.iff([] { return 1.f; }, [] { return 0.f; });

        return 0;
    });

    return 0;

}