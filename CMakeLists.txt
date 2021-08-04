cmake_minimum_required(VERSION 3.11)

project(QHW
    VERSION 1.0
    DESCRIPTION "Quasi Harmonic Weights"
    LANGUAGES CXX)

option(USE_CUDA "" OFF)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

    set(CMAKE_CXX_EXTENSIONS OFF)

    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

endif()

if(USE_CUDA)

    # https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
    enable_language(CUDA)

    include(CheckLanguage)
    check_language(CUDA)

    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 11)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()

endif(USE_CUDA)

# The executable code is here
add_subdirectory(qhw)
