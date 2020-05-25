cmake_minimum_required(VERSION 3.0)
project(ch6)

set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Eigen
find_package(Eigen3 3.3.0 REQUIRED)
message("Eigen3 Version: ${Eigen3_VERSION}")
include_directories(${EIGEN3_INCLUDE_DIR})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDES_DIRS})

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# Executable
add_executable(gaussNewton gaussNewton.cpp)
target_link_libraries(gaussNewton ${OpenCV_LIBS})

add_executable(ceresCurceFitting ceresCurceFitting.cpp)
target_link_libraries(ceresCurceFitting ${OpenCV_LIBS} ${CERES_LIBRARIES})