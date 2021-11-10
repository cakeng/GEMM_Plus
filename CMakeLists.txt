cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test LANGUAGES C CXX ASM)
add_subdirectory(include)
add_executable(main main.cpp)
target_include_directories(main PUBLIC
			    "${PROJECT_BINARY_DIR}"
                "${PROJECT_SOURCE_DIR}/include"
                "${PROJECT_SOURCE_DIR}/include/kernels"
                          )
		  target_link_libraries(main ptmm miniTensor armnn XNNPACK pthreadpool cpuinfo clog openblas direct18 comp)
set(CMAKE_BUILD_TYPE Release)
set_property(TARGET main PROPERTY CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -march=native -Ofast -pthread -fPIC")




