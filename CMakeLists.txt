
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

add_executable(BasicSimdExample
    ${BASIC_SIMD_EXAMPLE_SOURCE})

target_include_directories(BasicSimdExample
    PRIVATE
        ${BASIC_SIMD_SOURCE_DIRECTORY}/include/)

# ==============================================
