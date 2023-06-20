cmake_minimum_required(VERSION 3.10)
project(mlp_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(mlp_cpp src/main.cpp src/data_loader.cpp)
target_include_directories(mlp_cpp PUBLIC include ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(mlp_cpp ${OpenCV_LIBS} fmt::fmt)
