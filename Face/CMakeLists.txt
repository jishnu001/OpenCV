cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
project( DetectFace )
find_package( OpenCV REQUIRED )
add_executable( DetectFace detect_face.cpp )
target_link_libraries( DetectFace ${OpenCV_LIBS} )
