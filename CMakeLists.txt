cmake_minimum_required(VERSION 3.0)

project(FixedNets)
set(FixedNets_VERSION_MAJOR 1)
set(FixedNets_VERSION_MINOR 0)

set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_FLAGS "-g -MMD -MP -Wall -Wextra -Winit-self -Wno-unused-parameter -Wfloat-equal -std=c++14 -O3")

include_directories(src)
add_executable(main src/main.cpp)
