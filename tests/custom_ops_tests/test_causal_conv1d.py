# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal connectivity test for the npu_causal_conv1d_custom framework op.
#
# NOTE: this test requires the CANN custom op package to be discoverable at
# runtime. Before running, source the generated environment script:
#   source /usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/bin/set_env.bash

import os
import sys

CUSTOM_OPP_MARKER = "/workspace/vllm-plugin-FL/vllm_fl/_cann_ops_custom/vendors/custom_transformer"


def _check_custom_op_env() -> None:
    """Ensure LD_LIBRARY_PATH was set before Python started."""
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    opp_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    if CUSTOM_OPP_MARKER not in ld_path or CUSTOM_OPP_MARKER not in opp_path:
        print(
            "ERROR: CANN custom op environment is not set.\n"
            "Please run the following command in your shell first, then re-run this test:\n"
            f"  source {CUSTOM_OPP_MARKER}/bin/set_env.bash",
            file=sys.stderr,
        )
        sys.exit(1)


_check_custom_op_env()

import torch
import torch_npu

# Load the Ascend C++ extension that registers torch.ops._C_ascend.
import vllm_fl._C_ascend  # noqa: F401

DEVICE = "npu:0"


def test_causal_conv1d_varlen():
    """Varlen prefill mode (run_mode=0) with 2 sequences."""
    batch = 2
    seqlen = 16
    dim = 16  # must be aligned to 16
    width = 4  # in [2, 4]
    state_len = width - 1
    num_cache_lines = batch

    total_seqlen = batch * seqlen

    x = torch.randn(total_seqlen, dim, dtype=torch.float16, device=DEVICE)
    weight = torch.randn(width, dim, dtype=torch.float16, device=DEVICE)
    conv_state = torch.randn(num_cache_lines, state_len, dim, dtype=torch.float16, device=DEVICE)
    bias = torch.randn(dim, dtype=torch.float16, device=DEVICE)
    query_start_loc = torch.tensor([0, seqlen, total_seqlen], dtype=torch.int32, device=DEVICE)
    cache_indices = torch.arange(batch, dtype=torch.int32, device=DEVICE)
    initial_state_mode = torch.zeros(batch, dtype=torch.int32, device=DEVICE)

    output = torch.empty_like(x)

    result = torch.ops._C_ascend.npu_causal_conv1d_custom(
        output,
        x,
        weight,
        conv_state,
        bias,
        query_start_loc,
        cache_indices,
        initial_state_mode,
        None,  # num_accepted_tokens
        activation_mode=1,
        pad_slot_id=-1,
        run_mode=0,
    )

    assert result.shape == x.shape
    assert result.device == x.device


if __name__ == "__main__":
    test_causal_conv1d_varlen()
    print("npu_causal_conv1d_custom varlen test passed")
