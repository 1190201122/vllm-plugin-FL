# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Single-case debug test for add_rms_norm_bias in vllm-plugin-FL.

from typing import Optional, Tuple

import pytest
import torch

# 加载 vllm-plugin-FL 的 Ascend 扩展，触发 torch.ops._C_ascend 注册。
# 如果编译安装成功，vllm_fl._C_ascend 会被安装到 vllm_fl 包下。
import vllm_fl._C_ascend  # noqa: F401


def add_rms_norm_bias_ref(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    beta: Optional[torch.Tensor],
    epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """纯 PyTorch 参考实现。"""
    x = x1 + x2
    x_fp32 = x.to(torch.float32)
    gamma_fp32 = gamma.to(torch.float32)

    norm_dims = list(range(-len(gamma.shape), 0))
    variance = x_fp32.pow(2).mean(dim=norm_dims, keepdim=True)
    rstd = torch.rsqrt(variance + epsilon)
    y_fp32 = x_fp32 * rstd * gamma_fp32

    if beta is not None:
        y_fp32 = y_fp32 + beta.to(torch.float32)

    y = y_fp32.to(x1.dtype)

    rstd_shape = list(x1.shape)
    for i in range(-1, -len(gamma.shape) - 1, -1):
        rstd_shape[i] = 1
    rstd = rstd.view(rstd_shape).to(torch.float32)

    return y, rstd, x


@pytest.mark.parametrize(
    "num_tokens,hidden_size,dtype,use_beta",
    [
        (16, 128, torch.float16, True),
    ],
)
@torch.inference_mode()
def test_add_rms_norm_bias_plugin_fl(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    use_beta: bool,
) -> None:
    """单例调试：测试 vllm-plugin-FL 的 add_rms_norm_bias 绑定。"""
    device = f"npu:{0}"
    epsilon = 1e-6
    seed = 0

    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)

    shape = (num_tokens, hidden_size)
    gamma_shape = (hidden_size,)

    x1 = torch.randn(shape, dtype=dtype, device=device)
    x2 = torch.randn(shape, dtype=dtype, device=device)
    gamma = torch.randn(gamma_shape, dtype=dtype, device=device)
    beta = torch.randn(gamma_shape, dtype=dtype, device=device) if use_beta else None

    ref_y, ref_rstd, ref_x = add_rms_norm_bias_ref(x1, x2, gamma, beta, epsilon)
    custom_y, custom_rstd, custom_x = torch.ops._C_ascend.npu_add_rms_norm_bias(
        x1, x2, gamma, beta, epsilon
    )

    print(
        f"Testing add_rms_norm_bias (vllm-plugin-FL): "
        f"tokens={num_tokens}, hidden={hidden_size}, dtype={dtype}, beta={use_beta}"
    )

    assert ref_y.shape == custom_y.shape
    assert ref_y.dtype == custom_y.dtype

    torch.testing.assert_close(
        custom_y,
        ref_y,
        atol=1e-2,
        rtol=1e-3,
        msg="Output y mismatch",
    )
    torch.testing.assert_close(
        custom_rstd,
        ref_rstd,
        atol=1e-3,
        rtol=1e-4,
        msg="Output rstd mismatch",
    )
    torch.testing.assert_close(
        custom_x,
        ref_x,
        atol=1e-5,
        rtol=1e-5,
        msg="Output x mismatch",
    )
