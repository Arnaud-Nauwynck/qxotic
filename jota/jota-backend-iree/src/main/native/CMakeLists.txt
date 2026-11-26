# libjota_iree_rt: the IREE C runtime (VM + HAL + local CPU drivers) as one shared library, so the
# Java FFM binding can dlopen it. IREE ships its runtime only as static libraries; this project
# builds the runtime alone (no compiler, no LLVM: C only, a few minutes) from an IREE source tree
# and links iree_runtime_unified whole into a shared object with its iree_* API visible.
#
#   cmake -S src/main/native -B build -DIREE_SOURCE_DIR=/path/to/iree   (submodule third_party/flatcc)
#   cmake --build build --target jota_iree_rt
#
# Use the same IREE revision as the compiler (libIREECompiler) that produces the .vmfb modules:
# `ireeCompilerGetRevision()` prints it, e.g. "3.14.0rc20261115 @ 5332c682...".
cmake_minimum_required(VERSION 3.24)
project(jota_iree_rt C CXX)

set(IREE_SOURCE_DIR "" CACHE PATH "IREE source checkout (with third_party/flatcc)")
if(NOT IREE_SOURCE_DIR OR NOT EXISTS "${IREE_SOURCE_DIR}/runtime/src/iree/runtime/api.h")
  message(FATAL_ERROR "Set -DIREE_SOURCE_DIR to an IREE source checkout (got '${IREE_SOURCE_DIR}')")
endif()
if(NOT EXISTS "${IREE_SOURCE_DIR}/third_party/flatcc/CMakeLists.txt")
  message(FATAL_ERROR "IREE submodule third_party/flatcc is missing: "
                      "git -C ${IREE_SOURCE_DIR} submodule update --init third_party/flatcc")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Runtime only.
set(IREE_BUILD_COMPILER OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_BINDINGS_TFLITE OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_BINDINGS_TFLITE_JAVA OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_ALL_CHECK_TEST_MODULES OFF CACHE BOOL "" FORCE)
set(IREE_ENABLE_CPUINFO ON CACHE BOOL "" FORCE)
# Only third_party/flatcc is needed for the runtime; do not require the others.
set(IREE_ERROR_ON_MISSING_SUBMODULES OFF CACHE BOOL "" FORCE)
# The API must stay visible in the shared object.
set(IREE_VISIBILITY_HIDDEN OFF CACHE BOOL "" FORCE)

# CPU drivers first; GPU drivers are opt-in (-DJOTA_IREE_DRIVERS_GPU=ON adds the defaults IREE
# detects for the platform: cuda / hip / vulkan / metal when their SDK headers are found).
option(JOTA_IREE_DRIVERS_GPU "Also build IREE's default GPU HAL drivers" OFF)
if(NOT JOTA_IREE_DRIVERS_GPU)
  set(IREE_HAL_DRIVER_DEFAULTS OFF CACHE BOOL "" FORCE)
  set(IREE_HAL_DRIVER_LOCAL_SYNC ON CACHE BOOL "" FORCE)
  set(IREE_HAL_DRIVER_LOCAL_TASK ON CACHE BOOL "" FORCE)
endif()
set(IREE_HAL_EXECUTABLE_LOADER_DEFAULTS OFF CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF ON CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_LOADER_SYSTEM_LIBRARY ON CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE ON CACHE BOOL "" FORCE)

add_subdirectory("${IREE_SOURCE_DIR}" iree EXCLUDE_FROM_ALL)

add_library(jota_iree_rt SHARED jota_iree_rt.c)
target_link_libraries(jota_iree_rt PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,iree_runtime_unified>")
if(UNIX AND NOT APPLE)
  target_link_options(jota_iree_rt PRIVATE "-Wl,--no-undefined")
endif()
