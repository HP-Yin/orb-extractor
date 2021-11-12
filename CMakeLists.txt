cmake_minimum_required(VERSION 2.8)
project(fts_extract_match)

set(CMAKE_BUILD_TYPE "Debug")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 ${SSE_FLAGS} -g -O3 -march=native")

find_package(OpenCV 3 REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}
)

add_library(fts_extract SHARED
    orb_extractor.cpp
)

add_executable(test_orb test_orb.cpp)
target_link_libraries(test_orb ${OpenCV_LIBS} fts_extract)


