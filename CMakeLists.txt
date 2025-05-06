cmake_minimum_required(VERSION 3.5.1)

project(demo CXX)
set(CMAKE_CXX_STANDARD 17)
option(BUILD_STATIC "Set to ON to include static versions of the library" OFF)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${HOME}/local/openFHE")
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${HOME}/local/cnpy")


find_library(CNPY_LIB cnpy PATHS  /mnt/shared/gpfs/home/billdai2/local/cnpy/lib REQUIRED)
find_path(CNPY_INCLUDE_DIR cnpy.h PATHS  /mnt/shared/gpfs/home/billdai2/local/cnpy/include REQUIRED)

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

# 设置 CXX 编译选项
set(CMAKE_CXX_FLAGS ${OpenFHE_CXX_FLAGS})

include_directories(${OPENMP_INCLUDES})
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)

link_directories(${OpenFHE_LIBDIR})
link_directories(${OPENMP_LIBRARIES})
link_directories(/mnt/shared/gpfs/home/billdai2/local/cnpy/lib)


if(BUILD_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    link_libraries(${OpenFHE_STATIC_LIBRARIES})
else()
    set(CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
    link_libraries(${OpenFHE_SHARED_LIBRARIES})
endif()

# 添加源文件并创建可执行文件
file(GLOB ENCRYPTION "src/encryption/*.cpp")
add_library(encryption ${ENCRYPTION})
target_include_directories(encryption PUBLIC "src/encryption")

file(GLOB CNN "src/cnn/*.cpp")
add_library(cnn ${CNN})
target_include_directories(cnn PUBLIC "src/cnn")

file(GLOB SOURCES "src/*.cpp")
file(GLOB common_sources src/utils/*.cpp)
add_executable(test "src/main.cpp" ${common_sources})

target_link_libraries(test encryption)
target_link_libraries(test cnn)
target_link_libraries(test ${CNPY_LIB} z)

set_property(TARGET test PROPERTY CXX_STANDARD 17)

target_include_directories(test PRIVATE ${CNPY_INCLUDE_DIR})
target_include_directories(test PRIVATE ${CMAKE_SOURCE_DIR}/src/include)

# 禁用所有警告
add_compile_options(-w)

target_compile_options(test PRIVATE
    -Wall                  # 启用所有常见警告
    -Wextra                # 启用额外的警告
    -Wno-unused-variable   # 忽略未使用变量的警告
    -Wno-error             # 禁止将警告视为错误
)
