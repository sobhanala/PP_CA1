cmake_minimum_required(VERSION 3.29)
project(Q1)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenCV REQUIRED PATHS "C:/tools/opencv/build" NO_DEFAULT_PATH)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Q1 Q1.cpp)

# Link OpenCV libraries
target_link_libraries(Q1 ${OpenCV_LIBS})
