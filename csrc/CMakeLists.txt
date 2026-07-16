# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-FL project

cmake_minimum_required(VERSION 3.26)
project(vllm_fl_extensions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED VLLM_VENDOR)
    if(DEFINED ENV{VLLM_VENDOR})
        set(VLLM_VENDOR $ENV{VLLM_VENDOR})
    endif()
endif()

if(NOT VLLM_VENDOR)
    message(FATAL_ERROR
        "VLLM_VENDOR is required but not specified.\n"
        "Set VLLM_VENDOR=cuda."
    )
endif()

set(SUPPORTED_VENDORS cuda ascend)
if(NOT VLLM_VENDOR IN_LIST SUPPORTED_VENDORS)
    message(FATAL_ERROR
        "Unsupported vendor: ${VLLM_VENDOR}\n"
        "Supported vendors: ${SUPPORTED_VENDORS}"
    )
endif()

message(STATUS "vLLM-FL Extensions vendor: ${VLLM_VENDOR}")

if(VLLM_PYTHON_EXECUTABLE)
    set(Python_EXECUTABLE ${VLLM_PYTHON_EXECUTABLE})
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
message(STATUS "Python: ${Python_EXECUTABLE} (${Python_VERSION})")

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})

find_package(Torch REQUIRED)
message(STATUS "PyTorch: ${Torch_VERSION}")

if(VLLM_VENDOR STREQUAL "ascend")
    set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})

    if(NOT DEFINED TORCH_NPU_PATH)
        message(FATAL_ERROR
            "TORCH_NPU_PATH is required for Ascend builds. "
            "It is normally supplied by setup.py."
        )
    endif()
    message(STATUS "torch_npu package path: ${TORCH_NPU_PATH}")
    include_directories(${TORCH_NPU_PATH}/include)
    link_directories(${TORCH_NPU_PATH}/lib)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(${VLLM_VENDOR})

# Optional PTO chunk-GDN megakernel pre-compilation (Ascend only).
option(BUILD_PTO_CHUNK_GDN "Pre-compile PTO GDN megakernel via Bisheng" OFF)
if(VLLM_VENDOR STREQUAL "ascend" AND BUILD_PTO_CHUNK_GDN)
    add_subdirectory(pto_chunk_gdn EXCLUDE_FROM_ALL)
endif()
