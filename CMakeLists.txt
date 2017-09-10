cmake_minimum_required(VERSION 3.0)

project(FixedNets)
set(FixedNets_VERSION_MAJOR 1)
set(FixedNets_VERSION_MINOR 0)

# set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_COMPILER "g++-7")
set(CMAKE_CXX_FLAGS "-g -MMD -MP -Wall -Wextra -Winit-self -Wno-unused-parameter -Wfloat-equal -std=c++14 -O3")

find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(src)
add_executable(main src/main.cpp)
