cmake_minimum_required(VERSION 3.0)
enable_testing()

set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -O3")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenMP)
if(OpenMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(../src)
file(GLOB test_files "test_*.cpp")

foreach(test ${test_files})
  get_filename_component(test_name ${test} NAME_WE)
  message(${test_name})
  add_executable(${test_name} ${test})
  add_test(${test_name} "${CMAKE_CURRENT_BINARY_DIR}/${test_name}")
endforeach()
