# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal connectivity test for the chunk_gated_delta_rule_fwd_h framework op.
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


def test_chunk_gated_delta_rule_fwd_h():
    """Basic chunk_gated_delta_rule_fwd_h connectivity test."""
    b = 2
    h = 2
    t = 64
    k = 32
    hv = 4
    v = 32
    chunk_size = 64

    k_tensor = torch.randn(b, h, t, k, dtype=torch.bfloat16, device=DEVICE)
    w_tensor = torch.randn(b, h, t, k, dtype=torch.bfloat16, device=DEVICE)
    u_tensor = torch.randn(b, hv, t, v, dtype=torch.bfloat16, device=DEVICE)
    g_tensor = torch.randn(b, h, t, k, dtype=torch.float32, device=DEVICE)
    initial_state = torch.randn(b, hv, k, v, dtype=torch.bfloat16, device=DEVICE)

    cu_seqlens = [0, t, b * t]

    h_out, v_new_out, final_state_out = torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
        k_tensor,
        w_tensor,
        u_tensor,
        g=g_tensor,
        gk=None,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
        use_exp2=False,
        transpose_state_layout=False,
    )

    nt = (t + chunk_size - 1) // chunk_size
    assert h_out.shape == (b, hv, nt, k, v)
    assert v_new_out.shape == u_tensor.shape
    assert final_state_out.shape == (b, hv, k, v)


if __name__ == "__main__":
    test_chunk_gated_delta_rule_fwd_h()
    print("chunk_gated_delta_rule_fwd_h test passed")
