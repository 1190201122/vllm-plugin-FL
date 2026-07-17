# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal connectivity test for the npu_fused_gdn_gating framework op.
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


def test_fused_gdn_gating():
    """Basic fused GDN gating connectivity test."""
    batch = 4
    num_heads = 8

    A_log = torch.randn(num_heads, dtype=torch.bfloat16, device=DEVICE)
    a = torch.randn(batch, num_heads, dtype=torch.bfloat16, device=DEVICE)
    b = torch.randn(batch, num_heads, dtype=torch.bfloat16, device=DEVICE)
    dt_bias = torch.randn(num_heads, dtype=torch.bfloat16, device=DEVICE)

    g, beta_output = torch.ops._C_ascend.npu_fused_gdn_gating(
        A_log, a, b, dt_bias, beta=1.0, threshold=20.0
    )

    assert g.shape == (1, batch, num_heads)
    assert beta_output.shape == (1, batch, num_heads)
    assert g.device == A_log.device


if __name__ == "__main__":
    test_fused_gdn_gating()
    print("npu_fused_gdn_gating test passed")
