# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal connectivity test for the npu_recurrent_gated_delta_rule framework op.
#
# NOTE: this test requires the CANN custom op package to be discoverable at
# runtime. Before running, source the generated environment script:
#   source /workspace/vllm-plugin-FL/vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash

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

import vllm_fl._C_ascend  # noqa: F401

DEVICE = "npu:0"


def test_recurrent_gated_delta_rule():
    """Basic recurrent gated delta rule connectivity test."""
    t = 16
    nk = 2
    nv = 4  # must be a multiple of nk
    dk = 32
    dv = 32
    batch = 2

    query = torch.randn(t, nk, dk, dtype=torch.bfloat16, device=DEVICE)
    key = torch.randn(t, nk, dk, dtype=torch.bfloat16, device=DEVICE)
    value = torch.randn(t, nv, dv, dtype=torch.bfloat16, device=DEVICE)
    beta = torch.randn(t, nv, dtype=torch.bfloat16, device=DEVICE)
    state = torch.randn(batch, nv, dv, dk, dtype=torch.bfloat16, device=DEVICE)

    actual_seq_lengths = torch.full((batch,), t, dtype=torch.int32, device=DEVICE)
    ssm_state_indices = torch.arange(batch, dtype=torch.int32, device=DEVICE)

    output = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
        query,
        key,
        value,
        state,
        beta=beta,
        scale=dk ** -0.5,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=ssm_state_indices,
    )

    assert output.shape == value.shape
    assert output.device == value.device


if __name__ == "__main__":
    test_recurrent_gated_delta_rule()
    print("npu_recurrent_gated_delta_rule test passed")
