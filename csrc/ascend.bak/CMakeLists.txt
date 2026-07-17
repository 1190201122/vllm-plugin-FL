# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-FL project

set(VLLM_FL_ASCEND_SRCS
    torch_bindings.cpp
    aclnn_adapter/op_api_common.h
    aclnn_adapter/NPUBridge.cpp
    aclnn_adapter/NPUBridge.h
    aclnn_adapter/NPUStorageImpl.cpp
    aclnn_adapter/NPUStorageImpl.h
    # Framework operator torch adapters (op_host/op_kernel/ are built by the
    # CANN toolchain and are present here for reference / vendor packaging).
    add_rms_norm_bias/add_rms_norm_bias_torch_adpt.h
    apply_top_k_top_p_custom/apply_top_k_top_p_custom_torch_adpt.h
    causal_conv1d/causal_conv1d_torch_adpt.h
    copy_and_expand_eagle_inputs/copy_and_expand_eagle_inputs_torch_adpt.h
    dispatch_ffn_combine/dispatch_ffn_combine_torch_adpt.h
    dispatch_gmm_combine_decode/dispatch_gmm_combine_decode_torch_adpt.h
    dispatch_layout/dispatch_layout_torch_adpt.h
    gemma_rms_norm/gemma_rms_norm_torch_adpt.h
    grouped_matmul_swiglu_quant_weight_nz_tensor_list/grouped_matmul_swiglu_quant_torch_adpt.h
    lightning_indexer_quant/lightning_indexer_quant_torch_adpt.h
    lightning_indexer_vllm/lightning_indexer_vllm_torch_adpt.h
    matmul_allreduce_add_rmsnorm/matmul_allreduce_add_rmsnorm_torch_adpt.h
    moe_gating_top_k/moe_gating_top_k_torch_adpt.h
    moe_grouped_matmul/moe_grouped_matmul_torch_adpt.h
    moe_init_routing_custom/moe_init_routing_custom_torch_adpt.h
    sparse_flash_attention/sparse_flash_attention_torch_adpt.h
    transpose_kv_cache_by_block/transpose_kv_cache_by_block_torch_adpt.h
)

# Note: the framework operator source trees (op_host/op_kernel) and their
# CANN custom-op build system live under this directory for packaging, but
# they are compiled and deployed independently of the torch extension.
# See csrc/ascend/build_cann_ops.sh for the standalone CANN build workflow.

# Locate the AscendC kernel cmake helpers shipped with the CANN toolkit.
if(EXISTS ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

file(GLOB VLLM_FL_ASCEND_KERNELS kernels/*.cpp)
ascendc_library(vllm_fl_kernels SHARED ${VLLM_FL_ASCEND_KERNELS})

Python_add_library(_C_ascend MODULE WITH_SOABI ${VLLM_FL_ASCEND_SRCS})
target_compile_definitions(_C_ascend PRIVATE TORCH_EXTENSION_NAME=_C_ascend)

target_include_directories(_C_ascend PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/aclnn_adapter
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(_C_ascend PRIVATE
    ${TORCH_LIBRARIES}
    torch_npu
    vllm_fl_kernels
    ascendcl
    opapi
    tiling_api
    register
    platform
    ascendalog
    dl
)

target_compile_options(_C_ascend PRIVATE -O3)
target_link_options(_C_ascend PRIVATE "-Wl,-rpath,$ORIGIN")

install(TARGETS _C_ascend vllm_fl_kernels LIBRARY DESTINATION vllm_fl COMPONENT _C_ascend)
