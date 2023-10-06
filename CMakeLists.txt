# Build options
option(USE_DEBUG "Set to ON for Debug mode" ON)
option(USE_SANITIZER "Use santizer flags" ON)
option(BUILD_CLI "Build the 'lightbgm' command-line interface in addition to lib_lightgbm" ON)
option(BUILD_CPP_TEST "Build C++ tests with Google Test" ON)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to CMake 3.16
cmake_minimum_required(VERSION 3.16)

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

if(UNIX OR MINGW OR CYGWIN)
  set(
    CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -w"
  )
  if(USE_DEBUG)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
  else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas -Wno-unused-private-field")
endif()

# Header file directory
set(StochTree_HEADER_DIR ${PROJECT_SOURCE_DIR}/include)

# boost.math header file directory
set(BOOSTMATH_HEADER_DIR ${PROJECT_SOURCE_DIR}/dependencies/boost_math/include)

# Library directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

# Aggregate the source files underpinning the implementation in the C++ library
file(
  GLOB 
  SOURCES 
  src/config.cpp
  src/interface.cpp
  src/io.cpp
  src/json11.cpp
  src/model.cpp
  src/model_draw.cpp
  src/optimization.cpp
  src/train_data.cpp
  src/tree.cpp
)

# Define the C++ source code as a target
add_library(stochtree_objs OBJECT ${SOURCES})

# Include the headers in the source library
target_include_directories(stochtree_objs PRIVATE ${StochTree_HEADER_DIR} ${BOOSTMATH_HEADER_DIR})

if(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif()

if(BUILD_CLI)
  add_executable(stochtree src/main.cpp)
  target_include_directories(stochtree PRIVATE ${StochTree_HEADER_DIR} ${BOOSTMATH_HEADER_DIR})
  target_link_libraries(stochtree PRIVATE stochtree_objs)
  install(
    TARGETS stochtree
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
  )
endif()

if(BUILD_CPP_TEST)
  # Download the GoogleTest dependency if necessary
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        e2239ee6043f73722e7aa812a459f54a28552929 # release-1.14.0
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
  target_include_directories(teststochtree PRIVATE ${StochTree_HEADER_DIR} ${BOOSTMATH_HEADER_DIR} ${STOCHTREE_TEST_HEADER_DIR})
  target_link_libraries(teststochtree PRIVATE stochtree_objs GTest::gtest_main)
  gtest_discover_tests(teststochtree)
endif()
