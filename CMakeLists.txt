
# ==============================================

cmake_minimum_required(VERSION 3.15)

# ==============================================

set (CMAKE_CXX_STANDARD 23)

project(${NAME} VERSION ${VERSION})

# ==============================================

set(BASIC_SIMD_SOURCE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

# ==============================================

file(GLOB_RECURSE BASIC_SIMD_EXAMPLE_SOURCE
    "${BASIC_SIMD_SOURCE_DIRECTORY}/include/*.hpp"
    "${BASIC_SIMD_SOURCE_DIRECTORY}/example/*.cpp"
)

source_group(TREE ${BASIC_SIMD_SOURCE_DIRECTORY} FILES ${BASIC_SIMD_EXAMPLE_SOURCE})

add_executable(basic_simd_example
    ${BASIC_SIMD_EXAMPLE_SOURCE})

target_include_directories(basic_simd_example
    PRIVATE
        ${BASIC_SIMD_SOURCE_DIRECTORY}/include/)

# ==============================================

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
)
FetchContent_MakeAvailable(googletest)
add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)

enable_testing()

file(GLOB_RECURSE BASIC_SIMD_TESTS_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.hpp"
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${BASIC_SIMD_TESTS_SOURCE})

add_executable(basic_simd_tests ${BASIC_SIMD_TESTS_SOURCE})
add_dependencies(basic_simd_tests gtest)
target_include_directories(basic_simd_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(basic_simd_tests GTest::GTest)

add_test(basic_simd_tests_gtests basic_simd_tests)

# ==============================================
