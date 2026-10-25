cmake_minimum_required(VERSION 3.16)
project(sparse_packing_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
option(BUILD_STATIC "Link static OpenFHE libraries" OFF)
# Same installation convention as Openfhe_PE/CMakeLists.txt.
# HOME is normally an environment variable, not a predefined CMake variable.
if(DEFINED HOME AND NOT "${HOME}" STREQUAL "")
    list(APPEND CMAKE_PREFIX_PATH "${HOME}/local/openFHE")
elseif(DEFINED ENV{HOME})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/local/openFHE")
endif()
find_package(OpenFHE CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
message(STATUS "OpenFHE version: ${BASE_OPENFHE_VERSION}")

# Follow the OpenFHE 1.2.x external application configuration.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenFHE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenFHE_EXE_LINKER_FLAGS}")
link_directories(${OpenFHE_LIBDIR})
add_executable(sparse_packing_bench sparse_packing.cpp)
target_include_directories(sparse_packing_bench PRIVATE
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/third-party/include
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe)
if(BUILD_STATIC)
    target_link_libraries(sparse_packing_bench PRIVATE ${OpenFHE_STATIC_LIBRARIES})
else()
    target_link_libraries(sparse_packing_bench PRIVATE ${OpenFHE_SHARED_LIBRARIES})
endif()
target_link_libraries(sparse_packing_bench PRIVATE OpenMP::OpenMP_CXX)
target_compile_definitions(sparse_packing_bench PRIVATE
    BENCH_OPENFHE_VERSION="${BASE_OPENFHE_VERSION}")

add_executable(sparse_conv2d_test sparse_conv2d_test.cpp ../../src/cnn/sparse_conv2d.cpp ../../src/cnn/conv2d.cpp ../../src/cnn/layer.cpp ../../src/utils/globals.cpp ../../src/utils/helper.cpp)
target_include_directories(sparse_conv2d_test PRIVATE ${OpenFHE_INCLUDE} ${OpenFHE_INCLUDE}/third-party/include ${OpenFHE_INCLUDE}/core ${OpenFHE_INCLUDE}/pke ${OpenFHE_INCLUDE}/binfhe)
if(BUILD_STATIC)
 target_link_libraries(sparse_conv2d_test PRIVATE ${OpenFHE_STATIC_LIBRARIES})
else()
 target_link_libraries(sparse_conv2d_test PRIVATE ${OpenFHE_SHARED_LIBRARIES})
endif()
target_link_libraries(sparse_conv2d_test PRIVATE OpenMP::OpenMP_CXX)
enable_testing()
add_test(NAME sparse_conv2d_correctness COMMAND sparse_conv2d_test)

set_tests_properties(sparse_conv2d_correctness PROPERTIES TIMEOUT 600)

# OpenFHE exports -Werror; legacy helper warnings must not block this test.
target_compile_options(sparse_conv2d_test PRIVATE -Wno-error=unused-variable -Wno-error=sign-compare)

add_executable(sparse_activation_test sparse_activation_test.cpp
    ../../src/cnn/sparse_activation.cpp ../../src/cnn/sparse_conv2d.cpp
    ../../src/cnn/layer.cpp ../../src/utils/globals.cpp ../../src/utils/helper.cpp
    ../../src/encryption/setparam.cpp)
target_include_directories(sparse_activation_test PRIVATE ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/third-party/include ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke ${OpenFHE_INCLUDE}/binfhe)
if(BUILD_STATIC)
    target_link_libraries(sparse_activation_test PRIVATE ${OpenFHE_STATIC_LIBRARIES})
else()
    target_link_libraries(sparse_activation_test PRIVATE ${OpenFHE_SHARED_LIBRARIES})
endif()
target_link_libraries(sparse_activation_test PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(sparse_activation_test PRIVATE -Wno-error=unused-variable -Wno-error=sign-compare)
add_test(NAME sparse_activation_correctness COMMAND sparse_activation_test)
set_tests_properties(sparse_activation_correctness PROPERTIES TIMEOUT 900)

# Baseline is an unchanged pre-roundtrip fixture, linked only into this benchmark.
add_executable(relu_roundtrip_bench relu_roundtrip_bench.cpp
    baseline/baseline_sparse_activation.cpp ../../src/cnn/sparse_activation.cpp
    ../../src/cnn/layer.cpp ../../src/utils/globals.cpp ../../src/utils/helper.cpp
    ../../src/encryption/setparam.cpp)
target_include_directories(relu_roundtrip_bench PRIVATE ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/third-party/include ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke ${OpenFHE_INCLUDE}/binfhe)
if(BUILD_STATIC)
    target_link_libraries(relu_roundtrip_bench PRIVATE ${OpenFHE_STATIC_LIBRARIES})
else()
    target_link_libraries(relu_roundtrip_bench PRIVATE ${OpenFHE_SHARED_LIBRARIES})
endif()
target_link_libraries(relu_roundtrip_bench PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(relu_roundtrip_bench PRIVATE -Wno-error=unused-variable -Wno-error=sign-compare)

# Isolate active sign count from the fixed 1024-input reverse transform.
add_executable(reverse_active_bench reverse_active_bench.cpp
    ../../src/cnn/layer.cpp ../../src/utils/globals.cpp ../../src/utils/helper.cpp
    ../../src/encryption/setparam.cpp)
target_include_directories(reverse_active_bench PRIVATE ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/third-party/include ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke ${OpenFHE_INCLUDE}/binfhe)
if(BUILD_STATIC)
    target_link_libraries(reverse_active_bench PRIVATE ${OpenFHE_STATIC_LIBRARIES})
else()
    target_link_libraries(reverse_active_bench PRIVATE ${OpenFHE_SHARED_LIBRARIES})
endif()
target_link_libraries(reverse_active_bench PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(reverse_active_bench PRIVATE -Wno-error=unused-variable -Wno-error=sign-compare)

# Fresh-process A/B measurement of convolution scratch and rotation-key memory.
add_executable(conv_memory_bench conv_memory_bench.cpp ../../src/cnn/sparse_conv2d.cpp
    ../../src/cnn/conv2d.cpp ../../src/cnn/layer.cpp ../../src/utils/globals.cpp ../../src/utils/helper.cpp)
target_include_directories(conv_memory_bench PRIVATE ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/third-party/include ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke ${OpenFHE_INCLUDE}/binfhe)
if(BUILD_STATIC)
    target_link_libraries(conv_memory_bench PRIVATE ${OpenFHE_STATIC_LIBRARIES})
else()
    target_link_libraries(conv_memory_bench PRIVATE ${OpenFHE_SHARED_LIBRARIES})
endif()
target_link_libraries(conv_memory_bench PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(conv_memory_bench PRIVATE -Wno-error=unused-variable -Wno-error=sign-compare)

add_executable(client_region_test client_region_test.cpp)
add_test(NAME client_region_mapping COMMAND client_region_test)
