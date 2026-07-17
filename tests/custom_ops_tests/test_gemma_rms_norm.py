# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal connectivity test for the npu_gemma_rms_norm framework op.
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


def test_gemma_rms_norm():
    """Basic GemmaRMSNorm connectivity test."""
    x = torch.randn(16, 128, dtype=torch.float16, device=DEVICE)
    gamma = torch.randn(128, dtype=torch.float16, device=DEVICE)

    y, rstd = torch.ops._C_ascend.npu_gemma_rms_norm(x, gamma, 1e-6)

    assert y.shape == x.shape
    assert y.device == x.device
    assert rstd.shape == (x.size(0), 1)


if __name__ == "__main__":
    test_gemma_rms_norm()
    print("npu_gemma_rms_norm test passed")
