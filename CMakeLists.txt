cmake_minimum_required(VERSION 3.16)
project(YigithanWrapping)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-mavx -mavx2 -fopenmp)
add_link_options(-fopenmp)
find_package(SDL2 REQUIRED)
add_executable(YigithanWrapping main.cpp lfqueue.h)
target_link_libraries(YigithanWrapping SDL2 SDL2_image)