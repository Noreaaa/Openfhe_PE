cmake_minimum_required(VERSION 3.16)

project(demo CXX)
set(CMAKE_CXX_STANDARD 17)
option(BUILD_STATIC "Set to ON to include static versions of the library" OFF)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};$ENV{HOME}/local/openFHE")
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};$ENV{HOME}/local/cnpy")


find_library(CNPY_LIB cnpy HINTS "$ENV{HOME}/local/cnpy/lib" REQUIRED)
find_path(CNPY_INCLUDE_DIR cnpy.h HINTS "$ENV{HOME}/local/cnpy/include" REQUIRED)

message(STATUS "CNPY_INCLUDE_DIR: ${CNPY_INCLUDE_DIR}")
message(STATUS "CNPY_LIB: ${CNPY_LIB}")

include_directories(${CNPY_INCLUDE_DIR})


find_package(OpenMP REQUIRED)
find_package(OpenFHE CONFIG REQUIRED)

if(OpenMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if (OpenFHE_FOUND)
    message(STATUS "FOUND PACKAGE OpenFHE")
    message(STATUS "OpenFHE Version: ${BASE_OPENFHE_VERSION}")
    message(STATUS "OpenFHE installed as shared libraries: ${OpenFHE_SHARED}")
    message(STATUS "OpenFHE include files location: ${OpenFHE_INCLUDE}")
    message(STATUS "OpenFHE lib files location: ${OpenFHE_LIBDIR}")
    message(STATUS "OpenFHE Native Backend size: ${OpenFHE_NATIVE_SIZE}")
else()
    message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif()

set(CMAKE_CXX_FLAGS ${OpenFHE_CXX_FLAGS})

include_directories(${OPENMP_INCLUDES})
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)

link_directories(${OpenFHE_LIBDIR})
link_directories(${OPENMP_LIBRARIES})



if(BUILD_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    link_libraries(${OpenFHE_STATIC_LIBRARIES})
else()
    set(CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
    link_libraries(${OpenFHE_SHARED_LIBRARIES})
endif()

#
file(GLOB ENCRYPTION "src/encryption/*.cpp")
add_library(encryption ${ENCRYPTION})
target_include_directories(encryption PUBLIC "src/encryption")

file(GLOB CNN "src/cnn/*.cpp")
add_library(cnn ${CNN})
target_include_directories(cnn PUBLIC "src/cnn")

file(GLOB SOURCES "src/*.cpp")
file(GLOB common_sources src/utils/*.cpp)
add_executable(test "src/main.cpp" ${common_sources}
    src/init/hcnn.cpp
    src/init/resnet18.cpp
    src/init/baseline_test.cpp
)

target_link_libraries(test encryption)
target_link_libraries(test cnn)
target_link_libraries(test ${CNPY_LIB} z)

set_property(TARGET test PROPERTY CXX_STANDARD 17)

target_include_directories(test PRIVATE ${CNPY_INCLUDE_DIR})
target_include_directories(test PRIVATE ${CMAKE_SOURCE_DIR}/src/include)


add_compile_options(-w)

target_compile_options(test PRIVATE
    -Wall                  
    -Wextra                
    -Wno-unused-variable   
    -Wno-error             
)

# Single-layer benchmark with the actual ResNet third convolution weights.
add_executable(resnet_conv3_bench tests/sparse_packing/resnet_conv3_bench.cpp ${common_sources})
target_link_libraries(resnet_conv3_bench PRIVATE cnn encryption ${CNPY_LIB} z OpenMP::OpenMP_CXX)
target_include_directories(resnet_conv3_bench PRIVATE ${CMAKE_SOURCE_DIR}/src/include)

# Real-image accuracy check against an independent long-double plaintext reference.
add_executable(sparse_real_accuracy tests/sparse_packing/sparse_real_accuracy.cpp ${common_sources})
target_link_libraries(sparse_real_accuracy PRIVATE cnn encryption ${CNPY_LIB} z OpenMP::OpenMP_CXX)
target_include_directories(sparse_real_accuracy PRIVATE ${CMAKE_SOURCE_DIR}/src/include)

# Fully encrypted, one-ciphertext-per-channel ResNet prefix baseline.
add_executable(channelwise_prefix_bench tests/sparse_packing/channelwise_prefix_bench.cpp ${common_sources})
target_link_libraries(channelwise_prefix_bench PRIVATE cnn encryption ${CNPY_LIB} z OpenMP::OpenMP_CXX)
target_include_directories(channelwise_prefix_bench PRIVATE ${CMAKE_SOURCE_DIR}/src/include)

# Matched-layout sparse convolution optimization benchmark (real ResNet weights).
add_executable(sparse_conv_opt_bench tests/sparse_packing/sparse_conv_opt_bench.cpp ${common_sources})
target_link_libraries(sparse_conv_opt_bench PRIVATE cnn encryption ${CNPY_LIB} z OpenMP::OpenMP_CXX)
target_include_directories(sparse_conv_opt_bench PRIVATE ${CMAKE_SOURCE_DIR}/src/include)

# Historical decrypted-sign activation fixture; production `test` stays encrypted.
add_executable(test_decrypted_mask src/main.cpp ${common_sources}
    src/init/hcnn.cpp src/init/resnet18.cpp src/init/baseline_test.cpp
    tests/sparse_packing/baseline/baseline_sparse_activation.cpp)
target_compile_definitions(test_decrypted_mask PRIVATE PFHE_DECRYPTED_MASK_BASELINE=1)
target_link_libraries(test_decrypted_mask PRIVATE cnn encryption ${CNPY_LIB} z OpenMP::OpenMP_CXX)
target_include_directories(test_decrypted_mask PRIVATE ${CMAKE_SOURCE_DIR}/src/include)

# Unmodified historical PFHE convolution: representative ResNet stage shapes.
add_executable(legacy_conv_shape_bench tests/sparse_packing/legacy_conv_shape_bench.cpp ${common_sources})
target_link_libraries(legacy_conv_shape_bench PRIVATE cnn encryption ${CNPY_LIB} z OpenMP::OpenMP_CXX)
target_include_directories(legacy_conv_shape_bench PRIVATE ${CMAKE_SOURCE_DIR}/src/include)

# Fully encrypted channel-wise convolution shapes, sharing the prefix implementation.
add_executable(channelwise_conv_shape_bench tests/sparse_packing/channelwise_conv_shape_bench.cpp ${common_sources})
target_link_libraries(channelwise_conv_shape_bench PRIVATE cnn encryption ${CNPY_LIB} z OpenMP::OpenMP_CXX)
target_include_directories(channelwise_conv_shape_bench PRIVATE ${CMAKE_SOURCE_DIR}/src/include)

# Full approximation-plus-parallel-bootstrap environment for representative shapes.
add_executable(channelwise_approx_boot_shape_bench tests/sparse_packing/channelwise_approx_boot_shape_bench.cpp ${common_sources})
target_link_libraries(channelwise_approx_boot_shape_bench PRIVATE cnn encryption ${CNPY_LIB} z OpenMP::OpenMP_CXX)
target_include_directories(channelwise_approx_boot_shape_bench PRIVATE ${CMAKE_SOURCE_DIR}/src/include)
