cmake_minimum_required(VERSION 3.30)
project(CudaMCTS-Checkers CUDA CXX)

############################# Language Standards #############################
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

############################# Cuda Specific Flags ############################
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 75 80 86)

################################ Source Files ################################
file(GLOB_RECURSE SOURCE_FILES "src/*.cu" "src/*.cpp" main.cpp)

############################# Executable Target ##############################
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

################################## Includes ##################################
target_include_directories(${PROJECT_NAME} PRIVATE include)

######################### Libraries and Dependencies #########################
find_package(CUDAToolkit REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

############################### Compile Options ###############################
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif()

# C++ Compile Options
target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0 -g -Wall -Wextra -pedantic>
)

# CUDA Compile Options
target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G -g -lineinfo>
)