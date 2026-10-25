cmake_minimum_required(VERSION 3.16)
project(main_experiment_recovery LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/local/openFHE" "$ENV{HOME}/local/cnpy")
find_package(OpenFHE CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_path(CNPY_INCLUDE_DIR cnpy.h HINTS "$ENV{HOME}/local/cnpy/include" REQUIRED)
find_library(CNPY_LIBRARY cnpy HINTS "$ENV{HOME}/local/cnpy/lib" REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenFHE_CXX_FLAGS}")
get_filename_component(REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(OLD "${CMAKE_CURRENT_SOURCE_DIR}/main_original_full_encrypted_shapes_20261025")
set(SIX "${CMAKE_CURRENT_SOURCE_DIR}/main_baseline_six_ring16384_20261025")
set(COMMON_CNN
    ${REPOSITORY_ROOT}/src/cnn/layer.cpp
    ${REPOSITORY_ROOT}/src/cnn/bootstrap.cpp
    ${REPOSITORY_ROOT}/src/cnn/linear.cpp
    ${REPOSITORY_ROOT}/src/cnn/pool.cpp
    ${REPOSITORY_ROOT}/src/cnn/region_update.cpp)
set(COMMON_UTILS
    ${REPOSITORY_ROOT}/src/utils/dataloader.cpp
    ${REPOSITORY_ROOT}/src/utils/globals.cpp
    ${REPOSITORY_ROOT}/src/utils/helper.cpp)
function(configure_snapshot target)
    target_include_directories(${target} PRIVATE
        ${REPOSITORY_ROOT}/src ${REPOSITORY_ROOT}/src/include
        ${REPOSITORY_ROOT}/src/cnn ${REPOSITORY_ROOT}/src/encryption
        ${OLD}/compat ${CNPY_INCLUDE_DIR}
        ${OpenFHE_INCLUDE} ${OpenFHE_INCLUDE}/third-party/include
        ${OpenFHE_INCLUDE}/core ${OpenFHE_INCLUDE}/pke ${OpenFHE_INCLUDE}/binfhe)
    target_link_directories(${target} PRIVATE ${OpenFHE_LIBDIR})
    target_link_libraries(${target} PRIVATE ${OpenFHE_SHARED_LIBRARIES}
        ${CNPY_LIBRARY} z OpenMP::OpenMP_CXX)
    target_compile_options(${target} PRIVATE -Wno-error)
endfunction()
add_executable(snapshot_original_baseline
    ${OLD}/source/main.cpp ${OLD}/source/model.cpp
    ${OLD}/source/conv2d.cpp ${OLD}/source/activation.cpp
    ${OLD}/source/encryption.cpp ${COMMON_CNN} ${COMMON_UTILS})
configure_snapshot(snapshot_original_baseline)
add_executable(snapshot_baseline_six
    ${SIX}/source/main_six.cpp ${SIX}/source/model_instrumented.cpp
    ${SIX}/source/conv2d_bounds_fixed.cpp
    ${REPOSITORY_ROOT}/src/cnn/activation.cpp
    ${REPOSITORY_ROOT}/src/encryption/encryption.cpp
    ${COMMON_CNN} ${COMMON_UTILS})
configure_snapshot(snapshot_baseline_six)
add_executable(snapshot_bounds_regression
    ${SIX}/source/bounds_regression.cpp ${SIX}/source/conv2d_bounds_fixed.cpp
    ${REPOSITORY_ROOT}/src/cnn/layer.cpp
    ${REPOSITORY_ROOT}/src/utils/globals.cpp ${REPOSITORY_ROOT}/src/utils/helper.cpp)
configure_snapshot(snapshot_bounds_regression)
