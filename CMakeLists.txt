# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.24)
project(prowl VERSION 0.1.0 LANGUAGES CXX)

option(PROWL_ENABLE_CUDA "Build the native NVIDIA CUDA backend" OFF)
option(PROWL_BUILD_TESTS "Build C++ unit tests (Catch2)" ON)
option(PROWL_BUILD_PYTHON "Build the nanobind Python extension" ON)
option(PROWL_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(PROWL_NATIVE_ARCH "Compile C++ for the build CPU (-march=native: AVX2/FMA kernels in Eigen); not portable" OFF)
set(PROWL_CUDA_ARCHITECTURES "80;90;120" CACHE STRING "CUDA SM targets (A100=80, H100=90, B200=100, RTX50=120)")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(PROWL_NATIVE_ARCH)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(nlohmann_json 3 REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)
find_path(PROWL_CBLAS_INCLUDE cblas.h PATH_SUFFIXES openblas openblas-pthread x86_64-linux-gnu x86_64-linux-gnu/openblas-pthread)
if(NOT PROWL_CBLAS_INCLUDE)
  message(FATAL_ERROR "cblas.h not found; install OpenBLAS development headers")
endif()

add_library(prowl_warnings INTERFACE)
target_compile_options(prowl_warnings INTERFACE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion>
  $<$<AND:$<BOOL:${PROWL_WARNINGS_AS_ERRORS}>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>>:-Werror>)

add_library(prowl_core
  src/vasp/gvectors.cpp
  src/vasp/wavecar.cpp
  src/vasp/poscar.cpp
  src/vasp/potcar.cpp
  src/basis/radial.cpp
  src/basis/ylm.cpp
  src/basis/sto_basis.cpp
  src/basis/twocenter.cpp
  src/projection/projection.cpp
  src/projection/tables.cpp
  src/projection/streaming.cpp
  src/analysis/analysis.cpp
  src/analysis/reconstruction.cpp
  src/analysis/window.cpp
  src/analysis/numerics.cpp
  src/backend/cpu/cpu_backend.cpp
  src/backend/backend.cpp
  src/backend/threading.cpp
)
target_include_directories(prowl_core PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src> PRIVATE ${PROWL_CBLAS_INCLUDE})
target_link_libraries(prowl_core PUBLIC Eigen3::Eigen PRIVATE nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX BLAS::BLAS prowl_warnings)
target_compile_definitions(prowl_core PUBLIC PROWL_DATA_DIR="${PROJECT_SOURCE_DIR}/data")

if(PROWL_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit 12.8 REQUIRED)
  add_library(prowl_cuda src/backend/cuda/cuda_backend.cu)
  set_target_properties(prowl_cuda PROPERTIES CUDA_ARCHITECTURES "${PROWL_CUDA_ARCHITECTURES}"
                                               CUDA_SEPARABLE_COMPILATION OFF)
  target_include_directories(prowl_cuda PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)
  target_link_libraries(prowl_cuda PUBLIC Eigen3::Eigen PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
  target_link_libraries(prowl_core PUBLIC prowl_cuda)
  target_compile_definitions(prowl_core PUBLIC PROWL_HAVE_CUDA=1)
endif()

add_executable(prowl-native src/cli/main.cpp)
target_link_libraries(prowl-native PRIVATE prowl_core prowl_warnings)

if(PROWL_BUILD_PYTHON)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_core NB_STATIC src/python/bindings.cpp)
  target_link_libraries(_core PRIVATE prowl_core)
  if(SKBUILD)
    install(TARGETS _core LIBRARY DESTINATION prowl)
    install(FILES data/basis/koga/koga_hf_sto.json DESTINATION prowl/data/basis/koga)
    install(FILES data/basis/polarization/prowl_lda_polarization.json DESTINATION prowl/data/basis/polarization)
    install(FILES data/basis/paw-pbe-sto-1/paw-pbe-sto-1.json data/basis/paw-pbe-sto-1/LICENSE.md
            DESTINATION prowl/data/basis/paw-pbe-sto-1)
  else()
    # Developer builds drop the extension next to the pure-Python package.
    set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/python/prowl")
  endif()
endif()

if(PROWL_BUILD_TESTS)
  enable_testing()
  find_package(Catch2 3 REQUIRED)
  add_executable(prowl_unit_tests
    tests/unit/test_gvectors.cpp
    tests/unit/test_wavecar_synthetic.cpp
    tests/unit/test_radial_ylm.cpp
    tests/unit/test_basis.cpp
    tests/unit/test_twocenter.cpp
    tests/unit/test_backend.cpp
    tests/unit/test_threading.cpp
    tests/unit/test_analysis.cpp
    tests/unit/test_potcar.cpp
    tests/unit/test_paw_toy.cpp)
  target_link_libraries(prowl_unit_tests PRIVATE prowl_core Catch2::Catch2WithMain OpenMP::OpenMP_CXX prowl_warnings)
  include(Catch)
  catch_discover_tests(prowl_unit_tests WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endif()
