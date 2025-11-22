# Build options
option(USE_DEBUG "Build with debug symbols and without optimization" OFF)
option(USE_SANITIZER "Use santizer flags" OFF)
option(USE_OPENMP "Use openMP" ON)
option(USE_HOMEBREW_FALLBACK "(macOS-only) also look in 'brew --prefix' for libraries (e.g. OpenMP)" ON)
option(BUILD_TEST "Build C++ tests with Google Test" OFF)
option(BUILD_DEBUG_TARGETS "Build Standalone C++ Programs for Debugging" ON)
option(BUILD_PYTHON "Build Shared Library for Python Package" OFF)

# Require at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to CMake 3.20
cmake_minimum_required(VERSION 3.20)

# Define the project
project(stochtree LANGUAGES C CXX)

# Sanitizer support
set(
  ENABLED_SANITIZERS
  "address" "undefined"
  CACHE
  STRING
  "Semicolon separated list of sanitizer names, e.g., 'address;leak'. \
Supported sanitizers are address, leak, undefined and thread."
)
if(USE_SANITIZER)
  include(cmake/Sanitizer.cmake)
  enable_sanitizers("${ENABLED_SANITIZERS}")
endif()

# Debug flags
if(USE_DEBUG)
    add_definitions(-DDEBUG)
endif()

# Linker flags (empty by default, updated if using openmp)
set(
  STOCHTREE_LINK_FLAGS
  ""
)

# Unix / MinGW compiler flags
if(UNIX OR MINGW OR CYGWIN)
  set(
    CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -pthread -w"
  )
  if(USE_DEBUG)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
  else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas -Wno-unused-private-field")
endif()

# MSVC compiler flags
if(MSVC)
    set(
      variables
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_MINSIZEREL
        CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_MINSIZEREL
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /MP /utf-8")
    if(USE_DEBUG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Od")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /Ob2 /Oi /Ot /Oy")
    endif()
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    if(NOT USE_DEBUG)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")
    endif()
endif()

# OpenMP
if(USE_OPENMP)
  if(APPLE)
    find_package(OpenMP)
    if(NOT OpenMP_FOUND)
      if(USE_HOMEBREW_FALLBACK)
        execute_process(COMMAND brew --prefix libomp
                        OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(OpenMP_C_FLAGS "-Xclang -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_C_INCLUDE_DIR "")
        set(OpenMP_CXX_INCLUDE_DIR "")
        set(OpenMP_C_LIB_NAMES libomp)
        set(OpenMP_CXX_LIB_NAMES libomp)
        set(OpenMP_libomp_LIBRARY ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)
      endif()
      find_package(OpenMP REQUIRED)
    endif()
  else()
    find_package(OpenMP REQUIRED)
  endif()
  # Update flags with openmp 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Header file directory
set(StochTree_HEADER_DIR ${PROJECT_SOURCE_DIR}/include)

# boost.math header file directory
set(BOOSTMATH_HEADER_DIR ${PROJECT_SOURCE_DIR}/deps/boost_math/include)

# Eigen header file directory
set(EIGEN_HEADER_DIR ${PROJECT_SOURCE_DIR}/deps/eigen)
add_definitions(-DEIGEN_MPL2_ONLY)
add_definitions(-DEIGEN_DONT_PARALLELIZE)

# fast_double_parser header file directory
set(FAST_DOUBLE_PARSER_HEADER_DIR ${PROJECT_SOURCE_DIR}/deps/fast_double_parser/include)

# fmt header file directory
set(FMT_HEADER_DIR ${PROJECT_SOURCE_DIR}/deps/fmt/include)

# Library directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

# Aggregate the source files underpinning the implementation in the C++ library
file(
  GLOB 
  SOURCES 
  src/container.cpp
  src/cutpoint_candidates.cpp
  src/data.cpp
  src/io.cpp
  src/json11.cpp
  src/leaf_model.cpp
  src/partition_tracker.cpp
  src/random_effects.cpp
  src/tree.cpp
)

# Define the C++ source code as a target
add_library(stochtree_objs OBJECT ${SOURCES})

# Include the headers in the source library
if(USE_OPENMP)
  target_include_directories(stochtree_objs PRIVATE ${StochTree_HEADER_DIR} ${BOOSTMATH_HEADER_DIR} ${EIGEN_HEADER_DIR} ${FAST_DOUBLE_PARSER_HEADER_DIR} ${FMT_HEADER_DIR} ${OpenMP_CXX_INCLUDE_DIR})
  target_link_libraries(stochtree_objs PRIVATE ${OpenMP_libomp_LIBRARY})
else()
  target_include_directories(stochtree_objs PRIVATE ${StochTree_HEADER_DIR} ${BOOSTMATH_HEADER_DIR} ${EIGEN_HEADER_DIR} ${FAST_DOUBLE_PARSER_HEADER_DIR} ${FMT_HEADER_DIR})
endif()

# Python shared library
if (BUILD_PYTHON)
  # Add pybind11 module
  add_subdirectory(deps/pybind11)
  pybind11_add_module(stochtree_cpp src/py_stochtree.cpp)

  # Link to C++ source and headers
  if(USE_OPENMP)
    target_include_directories(stochtree_cpp PRIVATE ${StochTree_HEADER_DIR} ${BOOSTMATH_HEADER_DIR} ${EIGEN_HEADER_DIR} ${FAST_DOUBLE_PARSER_HEADER_DIR} ${FMT_HEADER_DIR} ${OpenMP_CXX_INCLUDE_DIR})
    target_link_libraries(stochtree_cpp PRIVATE stochtree_objs ${OpenMP_libomp_LIBRARY})
  else()
    target_include_directories(stochtree_cpp PRIVATE ${StochTree_HEADER_DIR} ${BOOSTMATH_HEADER_DIR} ${EIGEN_HEADER_DIR} ${FAST_DOUBLE_PARSER_HEADER_DIR} ${FMT_HEADER_DIR})
    target_link_libraries(stochtree_cpp PRIVATE stochtree_objs)
  endif()

  # EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
  # define (VERSION_INFO) here.
  target_compile_definitions(stochtree_cpp PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
endif()

# Build C++ test program
if(BUILD_TEST)
  # Check if user specified a local clone of the GoogleTest repo, use Github repo if not
  if (NOT DEFINED GOOGLETEST_GIT_REPO)
    set(GOOGLETEST_GIT_REPO https://github.com/google/googletest.git)
  endif()
  
  # Fetch and install GoogleTest dependency
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY ${GOOGLETEST_GIT_REPO}
    GIT_TAG        6910c9d9165801d8827d628cb72eb7ea9dd538c5 # release-1.16.0
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  # Build test suite
  enable_testing()
  include(GoogleTest)
  file(GLOB CPP_TEST_SOURCES test/cpp/*.cpp)
  add_executable(teststochtree ${CPP_TEST_SOURCES})
  set(STOCHTREE_TEST_HEADER_DIR ${PROJECT_SOURCE_DIR}/test/cpp)
  if(USE_OPENMP)
    target_include_directories(teststochtree PRIVATE ${StochTree_HEADER_DIR} ${BOOSTMATH_HEADER_DIR} ${EIGEN_HEADER_DIR} ${STOCHTREE_TEST_HEADER_DIR} ${FAST_DOUBLE_PARSER_HEADER_DIR} ${FMT_HEADER_DIR} ${OpenMP_CXX_INCLUDE_DIR})
    target_link_libraries(teststochtree PRIVATE stochtree_objs GTest::gtest_main ${OpenMP_libomp_LIBRARY})
  else()
    target_include_directories(teststochtree PRIVATE ${StochTree_HEADER_DIR} ${BOOSTMATH_HEADER_DIR} ${EIGEN_HEADER_DIR} ${STOCHTREE_TEST_HEADER_DIR} ${FAST_DOUBLE_PARSER_HEADER_DIR} ${FMT_HEADER_DIR})
    target_link_libraries(teststochtree PRIVATE stochtree_objs GTest::gtest_main)
  endif()
  gtest_discover_tests(teststochtree)
endif()

# Standalone C++ Program for Debugging
if(BUILD_DEBUG_TARGETS)
  # Build test suite
  add_executable(debugstochtree debug/api_debug.cpp)
  set(StochTree_DEBUG_HEADER_DIR ${PROJECT_SOURCE_DIR}/debug)
  if(USE_OPENMP)
    target_include_directories(debugstochtree PRIVATE ${StochTree_HEADER_DIR} ${BOOSTMATH_HEADER_DIR} ${EIGEN_HEADER_DIR} ${StochTree_DEBUG_HEADER_DIR} ${FAST_DOUBLE_PARSER_HEADER_DIR} ${FMT_HEADER_DIR} ${OpenMP_CXX_INCLUDE_DIR})
    target_link_libraries(debugstochtree PRIVATE stochtree_objs ${OpenMP_libomp_LIBRARY})
  else()
    target_include_directories(debugstochtree PRIVATE ${StochTree_HEADER_DIR} ${BOOSTMATH_HEADER_DIR} ${EIGEN_HEADER_DIR} ${StochTree_DEBUG_HEADER_DIR} ${FAST_DOUBLE_PARSER_HEADER_DIR} ${FMT_HEADER_DIR})
    target_link_libraries(debugstochtree PRIVATE stochtree_objs)
  endif()
endif()

