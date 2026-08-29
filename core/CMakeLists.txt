# VectorForge Core Library
add_library(vectorforge_core
    src/vector.cpp
    src/distance.cpp
    src/simd_math.cpp
    src/vector_index.cpp
    src/hnsw_index.cpp
    src/product_quantizer.cpp
)

target_include_directories(vectorforge_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(vectorforge_core PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(vectorforge_core PRIVATE /W4)
else()
    target_compile_options(vectorforge_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Main Demonstration Executable
add_executable(vectorforge_main
    src/main.cpp
)

target_link_libraries(vectorforge_main
    PRIVATE
        vectorforge_core
)

target_compile_features(vectorforge_main PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(vectorforge_main PRIVATE /W4)
else()
    target_compile_options(vectorforge_main PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Test Executable
add_executable(vectorforge_test
    tests/test_main.cpp
)

target_link_libraries(vectorforge_test
    PRIVATE
        vectorforge_core
)

target_compile_features(vectorforge_test PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(vectorforge_test PRIVATE /W4)
else()
    target_compile_options(vectorforge_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_test(NAME VectorForgeTests COMMAND vectorforge_test)

# Benchmark Executable
add_executable(vectorforge_benchmark
    benchmarks/benchmark_main.cpp
)

target_link_libraries(vectorforge_benchmark
    PRIVATE
        vectorforge_core
)

target_compile_features(vectorforge_benchmark PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(vectorforge_benchmark PRIVATE /W4)
else()
    target_compile_options(vectorforge_benchmark PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Scale Validation Benchmark Executable (1K, 10K, 50K)
add_executable(vectorforge_scale_benchmark
    benchmarks/scale_benchmark.cpp
)

target_link_libraries(vectorforge_scale_benchmark
    PRIVATE
        vectorforge_core
)

target_compile_features(vectorforge_scale_benchmark PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(vectorforge_scale_benchmark PRIVATE /W4)
else()
    target_compile_options(vectorforge_scale_benchmark PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Phase 3 SIMD Microbenchmark & End-to-End Search Benchmark Executable
add_executable(vectorforge_simd_benchmark
    benchmarks/simd_benchmark.cpp
)

target_link_libraries(vectorforge_simd_benchmark
    PRIVATE
        vectorforge_core
)

target_compile_features(vectorforge_simd_benchmark PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(vectorforge_simd_benchmark PRIVATE /W4)
else()
    target_compile_options(vectorforge_simd_benchmark PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Phase 4 Product Quantization (PQ) Benchmark Executable
add_executable(vectorforge_pq_benchmark
    benchmarks/pq_benchmark.cpp
)

target_link_libraries(vectorforge_pq_benchmark
    PRIVATE
        vectorforge_core
)

target_compile_features(vectorforge_pq_benchmark PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(vectorforge_pq_benchmark PRIVATE /W4)
else()
    target_compile_options(vectorforge_pq_benchmark PRIVATE -Wall -Wextra -Wpedantic)
endif()

# =========================================================================
# Phase 5: pybind11 Python Extension Module
# =========================================================================

execute_process(
    COMMAND python -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_DIR_RESULT
)

set(PYBIND11_FINDPYTHON ON)
set(Python3_EXECUTABLE "C:/Users/harsh/AppData/Local/Programs/Python/Python314/python.exe")
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

if(PYBIND11_DIR_RESULT EQUAL 0 AND EXISTS "${PYBIND11_CMAKE_DIR}")
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(vectorforge
        src/bindings.cpp
    )

    target_link_libraries(vectorforge
        PRIVATE
            vectorforge_core
    )

    if(MINGW)
        target_link_options(vectorforge PRIVATE -static-libgcc -static-libstdc++)
    endif()

    target_compile_features(vectorforge PUBLIC cxx_std_20)

    # Output module to server/ and build/core/
    set_target_properties(vectorforge PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/server"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/server"
    )
endif()
