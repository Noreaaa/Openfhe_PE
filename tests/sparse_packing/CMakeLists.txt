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
