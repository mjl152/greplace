cmake_minimum_required (VERSION 2.8)
project (greplace)
set (greplace_VERSION_MAJOR 1)
set (greplace_VERSION_MINOR 0)

list( APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3 ${CMAKE_CXX_FLAGS}")
list( APPEND CMAKE_C_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")

find_package(CUDA)
find_package(OpenCV REQUIRED)


include_directories("${PROJECT_BINARY_DIR}")
#
#if (CUDA_VERSION)
	#cuda_compile (ALPHA_FILTER_KERNEL_O alpha_filter_kernel.cu)
  #list( APPEND CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
  #compile (MAIN_O main.cpp)
 # compile (CPU_O cpu.cpp)
#  compile (GPU_O gpu.cpp)
#	cuda_add_executable(greplace main.cpp cpu.cpp person.cpp gpu.cpp #alpha_filter_kernel.cu)
 # cuda_add_executable(greplace-psearch greplace-psearch.cpp
  #                    greplace-psearch-cpu.cpp greplace-psearch-gpu.cpp cpu.cpp
 #                    person.cpp gpu.cpp alpha_filter_kernel.cu)
#else ()
  #list( APPEND CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
#	add_executable(greplace main.cpp cpu.cpp person.cpp)
  add_executable(greplace-psearch greplace-psearch.cpp greplace-psearch-cpu.cpp cpu.cpp person.cpp)
#endif ()

#target_link_libraries (greplace ${OpenCV_LIBS})
target_link_libraries (greplace-psearch ${OpenCV_LIBS})
