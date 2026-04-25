cmake_minimum_required(VERSION 3.18)
project(MissileDefenseSim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ─── Find required packages ───────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

# Eigen3 (header-only, optional for advanced computations)
find_package(Eigen3 QUIET)

# ─── Source files ─────────────────────────────────────────────────────────────
set(SOURCES
    main.cu
)

add_executable(defense ${SOURCES})

# ─── Include directories ──────────────────────────────────────────────────────
target_include_directories(defense PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${GLUT_INCLUDE_DIR}
)

if(Eigen3_FOUND)
    target_include_directories(defense PRIVATE ${EIGEN3_INCLUDE_DIR})
    target_compile_definitions(defense PRIVATE EIGEN_AVAILABLE)
    message(STATUS "Eigen3 found at ${EIGEN3_INCLUDE_DIR}")
else()
    # Fallback: check common location
    if(EXISTS "/usr/include/eigen3")
        target_include_directories(defense PRIVATE "/usr/include/eigen3")
        target_compile_definitions(defense PRIVATE EIGEN_AVAILABLE)
        message(STATUS "Eigen3 found at /usr/include/eigen3")
    else()
        message(WARNING "Eigen3 not found. Building without Eigen support.")
    endif()
endif()

# ─── Link libraries ───────────────────────────────────────────────────────────
target_link_libraries(defense PRIVATE
    CUDA::cudart
    CUDA::curand
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
)

# ─── CUDA architecture ────────────────────────────────────────────────────────
# Default to sm_75 (Turing / RTX 20-series); override with -DCUDA_ARCH=XX
if(NOT DEFINED CUDA_ARCH)
    set(CUDA_ARCH "75")
endif()
set_target_properties(defense PROPERTIES
    CUDA_ARCHITECTURES "${CUDA_ARCH}"
)

# ─── Compiler flags ───────────────────────────────────────────────────────────
target_compile_options(defense PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# On Linux, link math and GL utilities
if(UNIX)
    target_link_libraries(defense PRIVATE m GLU)
endif()

message(STATUS "Build configuration:")
message(STATUS "  CUDA architectures : ${CUDA_ARCH}")
message(STATUS "  OpenGL             : ${OPENGL_LIBRARIES}")
message(STATUS "  GLUT               : ${GLUT_LIBRARIES}")
