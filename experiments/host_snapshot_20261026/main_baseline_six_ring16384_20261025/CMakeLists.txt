cmake_minimum_required(VERSION 3.16)
project(original_baseline_six LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
list(APPEND CMAKE_PREFIX_PATH "/root/local/openFHE")
find_package(OpenFHE CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${OpenFHE_CXX_FLAGS}")
set(ROOT "/root/Openfhe_PE")
add_executable(baseline_six source/main_six.cpp source/model_instrumented.cpp source/conv2d_bounds_fixed.cpp
    ${ROOT}/src/utils/dataloader.cpp ${ROOT}/src/utils/globals.cpp ${ROOT}/src/utils/helper.cpp)
target_include_directories(baseline_six PRIVATE ${ROOT}/src ${ROOT}/src/include ${ROOT}/src/cnn
    ${CMAKE_CURRENT_SOURCE_DIR}/compat /root/local/cnpy/include
    ${OpenFHE_INCLUDE} ${OpenFHE_INCLUDE}/third-party/include ${OpenFHE_INCLUDE}/core ${OpenFHE_INCLUDE}/pke ${OpenFHE_INCLUDE}/binfhe)
target_link_directories(baseline_six PRIVATE ${OpenFHE_LIBDIR})
target_link_libraries(baseline_six PRIVATE
    /root/benchmarks/main_original_full_encrypted_shapes_20261025/build/libcnn.a
    /root/benchmarks/main_original_full_encrypted_shapes_20261025/build/libencryption.a
    ${OpenFHE_SHARED_LIBRARIES} /root/local/cnpy/lib/libcnpy.so z OpenMP::OpenMP_CXX)
target_compile_options(baseline_six PRIVATE -Wno-error)
add_executable(bounds_regression source/bounds_regression.cpp source/conv2d_bounds_fixed.cpp
    ${ROOT}/src/utils/globals.cpp ${ROOT}/src/utils/helper.cpp)
target_include_directories(bounds_regression PRIVATE ${ROOT}/src/cnn ${ROOT}/src/include
    ${OpenFHE_INCLUDE} ${OpenFHE_INCLUDE}/third-party/include ${OpenFHE_INCLUDE}/core ${OpenFHE_INCLUDE}/pke ${OpenFHE_INCLUDE}/binfhe)
target_link_directories(bounds_regression PRIVATE ${OpenFHE_LIBDIR})
target_link_libraries(bounds_regression PRIVATE
    /root/benchmarks/main_original_full_encrypted_shapes_20261025/build/libcnn.a
    ${OpenFHE_SHARED_LIBRARIES} OpenMP::OpenMP_CXX)
target_compile_options(bounds_regression PRIVATE -Wno-error)
