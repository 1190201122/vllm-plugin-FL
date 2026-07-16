# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal smoke tests for vllm-plugin-FL migrated framework operators.
# Each op is executed in an isolated subprocess so that a single AICore
# failure does not corrupt the NPU context for remaining tests.

import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import torch
import torch_npu

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE = "npu:0"


def _get_soc_name() -> str:
    """Map torch_npu SOC version to the canonical SOC name used in build files."""
    soc_version = torch_npu.npu.get_soc_version()
    # Ascend 910B variants (A2)
    if 220 <= soc_version <= 225:
        return "ascend910b"
    # Ascend 910_93 variants (A3)
    if 250 <= soc_version <= 255:
        return "ascend910_93"
    return f"soc_version_{soc_version}"


# Op -> set of SOC names on which the op is expected to run.
# Ops not listed here are assumed to work on every SOC.
OP_SOC_SUPPORT: dict[str, set[str]] = {
    # These ops rely on underlying CANN kernels that are only available on A3.
    "get_dispatch_layout": {"ascend910_93"},
    "dispatch_prefill": {"ascend910_93"},
    "combine_prefill": {"ascend910_93"},
}

# Ops that can be exercised meaningfully on a single NPU card.
SINGLE_CARD_OPS = {
    "npu_add_rms_norm_bias": """
        x1 = torch.randn(16, 128, dtype=torch.float16, device=DEVICE)
        x2 = torch.randn(16, 128, dtype=torch.float16, device=DEVICE)
        gamma = torch.randn(128, dtype=torch.float16, device=DEVICE)
        beta = torch.randn(128, dtype=torch.float16, device=DEVICE)
        torch.ops._C_ascend.npu_add_rms_norm_bias(x1, x2, gamma, beta, 1e-6)
    """,
    "npu_gemma_rms_norm": """
        x = torch.randn(16, 128, dtype=torch.float16, device=DEVICE)
        gamma = torch.randn(128, dtype=torch.float16, device=DEVICE)
        torch.ops._C_ascend.npu_gemma_rms_norm(x, gamma, 1e-6)
    """,
    "npu_apply_top_k_top_p": """
        logits = torch.randn(4, 1024, dtype=torch.float32, device=DEVICE)
        p = torch.full((4,), 0.9, dtype=torch.float32, device=DEVICE)
        k = torch.full((4,), 50, dtype=torch.int32, device=DEVICE)
        torch.ops._C_ascend.npu_apply_top_k_top_p(logits, p=p, k=k)
    """,
    "moe_gating_top_k": """
        x = torch.randn(8, 64, dtype=torch.float32, device=DEVICE)
        bias = torch.randn(64, dtype=torch.float32, device=DEVICE)
        torch.ops._C_ascend.moe_gating_top_k(
            x, k=4, k_group=1, group_count=1, group_select_mode=0,
            renorm=1, norm_type=0, out_flag=False,
            routed_scaling_factor=1.0, eps=1e-20, bias_opt=bias)
    """,
    "npu_moe_init_routing_custom": """
        x = torch.randn(16, 32, dtype=torch.float16, device=DEVICE)
        expert_idx = torch.randint(0, 4, (16, 2), dtype=torch.int32, device=DEVICE)
        torch.ops._C_ascend.npu_moe_init_routing_custom(
            x, expert_idx, scale=None, offset=None, active_num=16,
            expert_capacity=8, expert_num=4, drop_pad_mode=0,
            expert_tokens_num_type=0, expert_tokens_num_flag=True,
            quant_mode=-1, active_expert_range=[0, 4], row_idx_type=0)
    """,
    "get_dispatch_layout": """
        topk_idx = torch.randint(0, 4, (8, 2), dtype=torch.int32, device=DEVICE)
        torch.ops._C_ascend.get_dispatch_layout(topk_idx, num_experts=4, num_ranks=1)
    """,
    "npu_copy_and_expand_eagle_inputs": """
        target_token_ids = torch.randint(1, 1000, (10,), dtype=torch.int32, device=DEVICE)
        target_positions = torch.arange(10, dtype=torch.int32, device=DEVICE)
        next_token_ids = torch.randint(1, 1000, (2,), dtype=torch.int32, device=DEVICE)
        query_start_loc = torch.tensor([0, 5, 10], dtype=torch.int32, device=DEVICE)
        query_end_loc = torch.tensor([4, 9], dtype=torch.int32, device=DEVICE)
        torch.ops._C_ascend.npu_copy_and_expand_eagle_inputs(
            target_token_ids, target_positions, next_token_ids,
            query_start_loc, query_end_loc,
            padding_token_id=0, parallel_drafting_token_id=100,
            num_padding_slots_per_request=2, shift_input_ids=False,
            total_draft_tokens=20)
    """,
    "npu_causal_conv1d_custom": """
        # x: [total_seqlen, dim], weight: [width, dim], conv_state: [num_cache_lines, state_len, dim]
        seqlen, dim, width, num_seq = 128, 16, 4, 1
        state_len = width - 1
        x = torch.randn(seqlen, dim, dtype=torch.float16, device=DEVICE)
        weight = torch.randn(width, dim, dtype=torch.float16, device=DEVICE)
        conv_state = torch.randn(num_seq, state_len, dim, dtype=torch.float16, device=DEVICE)
        bias = torch.randn(dim, dtype=torch.float16, device=DEVICE)
        query_start_loc = [0, seqlen]
        cache_indices = list(range(num_seq))
        initial_state_mode = [0] * num_seq
        num_accepted_tokens = []
        torch.ops._C_ascend.npu_causal_conv1d_custom(
            x, weight, conv_state, bias,
            query_start_loc, cache_indices, initial_state_mode, num_accepted_tokens,
            activation_mode=1, pad_slot_id=-1, run_mode=0)
    """,
    "moe_grouped_matmul": """
        # x: [M, K], weight: [group_num, K, N] in FRACTAL_NZ, group_list: [group_num, 2]
        m, k, n, group_num = 16, 32, 64, 2
        x = torch.randn(m, k, dtype=torch.float16, device=DEVICE)
        weight = torch.randn(group_num, k, n, dtype=torch.float16, device=DEVICE)
        weight_nz = torch_npu.npu_format_cast(weight, 29)
        group_list = torch.tensor([[0, 8], [8, 16]], dtype=torch.int64, device=DEVICE)
        torch.ops._C_ascend.moe_grouped_matmul(
            x, weight_nz, group_list, split_item=0, group_type=0, group_list_type=0)
    """,
    "grouped_matmul_swiglu_quant_weight_nz_tensor_list": """
        M, K, E, N = 256, 512, 2, 256
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=DEVICE)
        weight = torch.randint(-128, 127, (E, K, N), dtype=torch.int8)
        weight_nz = [torch_npu.npu_format_cast(weight[i].to(DEVICE), 29) for i in range(E)]
        weight_scale = [torch.rand(N, dtype=torch.float32, device=DEVICE) * 0.5 + 0.1 for _ in range(E)]
        x_scale = torch.rand(M, dtype=torch.float32, device=DEVICE) * 0.5 + 0.1
        group_list = torch.tensor([128, 256], dtype=torch.int64, device=DEVICE)
        torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list(
            x, weight_nz, weight_scale, x_scale, group_list)
    """,
    "dispatch_prefill": """
        x = torch.randn(8, 64, dtype=torch.float16, device=DEVICE)
        topk_idx = torch.randint(0, 4, (8, 2), dtype=torch.int32, device=DEVICE)
        topk_weights = torch.rand(8, 2, dtype=torch.float32, device=DEVICE)
        num_tokens_per_rank = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        is_token_in_rank = torch.ones(8, 1, dtype=torch.bool, device=DEVICE)
        num_tokens_per_expert = torch.zeros(4, dtype=torch.int32, device=DEVICE)
        torch.ops._C_ascend.dispatch_prefill(
            x, topk_idx, topk_weights, num_tokens_per_rank,
            is_token_in_rank, num_tokens_per_expert,
            num_worst_tokens=0, groupEp="", rank=0, num_ranks=1)
    """,
    "combine_prefill": """
        x = torch.randn(8, 64, dtype=torch.float16, device=DEVICE)
        topk_idx = torch.randint(0, 4, (8, 2), dtype=torch.int32, device=DEVICE)
        topk_weights = torch.rand(8, 2, dtype=torch.float32, device=DEVICE)
        src_idx = torch.zeros(8, dtype=torch.int32, device=DEVICE)
        send_head = torch.zeros(4, dtype=torch.int32, device=DEVICE)
        torch.ops._C_ascend.combine_prefill(
            x, topk_idx, topk_weights, src_idx, send_head,
            grouEp="", rank=0, num_ranks=1)
    """,
    "transpose_kv_cache_by_block": """
        layers = 2
        block_num = 8
        block_size = 64
        num_kv_head = 4
        head_dim = 64
        split_num = 4
        k_caches = [torch.randn(block_num, block_size, num_kv_head, head_dim,
                                dtype=torch.float16, device=DEVICE) for _ in range(layers)]
        v_caches = [torch.randn(block_num, block_size, num_kv_head, head_dim,
                                dtype=torch.float16, device=DEVICE) for _ in range(layers)]
        block_ids = torch.randint(0, block_num, (4,), dtype=torch.int64, device=DEVICE)
        torch.ops._C_ascend.transpose_kv_cache_by_block(
            k_caches, v_caches, block_ids, block_size, num_kv_head,
            head_dim, split_num, layers)
    """,
    "get_masked_input_and_mask": """
        input_ids = torch.tensor([5, 15, 25, 35, 45], dtype=torch.int64, device=DEVICE)
        masked, mask = torch.ops._C_ascend.get_masked_input_and_mask(
            input_ids,
            org_vocab_start_index=10,
            org_vocab_end_index=20,
            num_org_vocab_padding=5,
            added_vocab_start_index=30,
            added_vocab_end_index=40)
        assert masked.shape == input_ids.shape
        assert mask.dtype == torch.bool
    """,
    "bgmv_shrink": """
        B, hidden_in, hidden_out, num_loras = 4, 128, 16, 8
        x = torch.randn(B, hidden_in, dtype=torch.float16, device=DEVICE)
        weight = torch.randn(num_loras, hidden_out, hidden_in, dtype=torch.float16, device=DEVICE)
        indices = torch.randint(0, num_loras, (B,), dtype=torch.int64, device=DEVICE)
        y = torch.zeros(B, hidden_out, dtype=torch.float16, device=DEVICE)
        torch.ops._C_ascend.bgmv_shrink(x, weight, indices, y, 0.5)
    """,
    "bgmv_expand": """
        B, hidden_in, hidden_out, num_loras = 4, 16, 128, 8
        x = torch.randn(B, hidden_in, dtype=torch.float16, device=DEVICE)
        weight = torch.randn(num_loras, hidden_out, hidden_in, dtype=torch.float16, device=DEVICE)
        indices = torch.randint(0, num_loras, (B,), dtype=torch.int64, device=DEVICE)
        y = torch.randn(B, hidden_out * 2, dtype=torch.float16, device=DEVICE)
        torch.ops._C_ascend.bgmv_expand(x, weight, indices, y, 0, hidden_out)
    """,
    "sgmv_shrink": """
        B, hidden_in, lora_rank, num_loras = 4, 128, 16, 4
        x = torch.randn(B, hidden_in, dtype=torch.float16, device=DEVICE)
        weight = torch.randn(num_loras, lora_rank, hidden_in, dtype=torch.float16, device=DEVICE)
        lora_indices = torch.randint(0, num_loras, (B,), dtype=torch.int64, device=DEVICE)
        seq_len = torch.ones(B, dtype=torch.int64, device=DEVICE)
        y = torch.zeros(B, lora_rank, dtype=torch.float16, device=DEVICE)
        torch.ops._C_ascend.sgmv_shrink(x, weight, lora_indices, seq_len, y, 0.5)
    """,
    "sgmv_expand": """
        B, lora_rank, hidden_out, num_loras = 4, 16, 128, 4
        x = torch.randn(B, lora_rank, dtype=torch.float16, device=DEVICE)
        weight = torch.randn(num_loras, hidden_out, lora_rank, dtype=torch.float16, device=DEVICE)
        lora_indices = torch.randint(0, num_loras, (B,), dtype=torch.int64, device=DEVICE)
        seq_len = torch.ones(B, dtype=torch.int64, device=DEVICE)
        y = torch.randn(B, hidden_out * 2, dtype=torch.float16, device=DEVICE)
        torch.ops._C_ascend.sgmv_expand(x, weight, lora_indices, seq_len, y, 0, hidden_out)
    """,
}

# Ops that are known to require multi-rank HCCL or special runtime configs.
SKIPPED_OPS = {
    "matmul_allreduce_add_rmsnorm": "requires HCCL multi-rank communicator",
    "dispatch_ffn_combine": "requires HCCL multi-rank communicator",
    "dispatch_gmm_combine_decode": "requires HCCL multi-rank communicator",
    "npu_sparse_flash_attention": "requires sparse flash attention runtime config",
    "npu_lightning_indexer": "requires sparse indexer runtime config",
    "npu_lightning_indexer_quant": "requires sparse indexer runtime config",
}


def _build_op_script(op_name: str, body: str) -> str:
    """Generate a standalone Python script that exercises one operator."""
    body = textwrap.indent(textwrap.dedent(body).strip(), "    ")
    return (
        "import sys\n"
        "import torch\n"
        "import torch_npu\n"
        "torch_npu.npu.config.allow_internal_format = True\n\n"
        "try:\n"
        "    from vllm_fl.utils import enable_custom_op\n"
        "    enable_custom_op()\n"
        "    import vllm_fl._C_ascend  # noqa: F401\n"
        "except ImportError as exc:\n"
        "    print(f'IMPORT_ERROR: {exc}')\n"
        "    sys.exit(2)\n\n"
        f'DEVICE = "{DEVICE}"\n'
        "try:\n"
        f"{body}\n"
        "    print('SMOKE_TEST_OK')\n"
        "    sys.stdout.flush()\n"
        "except Exception as exc:\n"
        "    print(f'FAIL: {type(exc).__name__}: {exc}')\n"
        "    sys.exit(1)\n"
    )


def _run_single(op_name: str, body: str, timeout: int = 120) -> tuple[str, str]:
    """Run one op smoke test in a fresh Python process."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_build_op_script(op_name, body))
        script_path = f.name

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent / "vllm-plugin-FL") + os.pathsep + env.get("PYTHONPATH", "")

    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        status = "OK" if proc.returncode == 0 and "SMOKE_TEST_OK" in stdout.splitlines() else "FAIL"
        detail = stdout if stdout else stderr
        return status, detail
    except subprocess.TimeoutExpired:
        return "TIMEOUT", f"did not finish within {timeout}s"
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def main() -> int:
    print("=" * 70)
    print("vllm-plugin-FL framework operator smoke test")
    print(f"Detected SOC: {_get_soc_name()}")
    print("=" * 70)

    # Verify extension is importable in the current environment first.
    try:
        spec = importlib.util.find_spec("vllm_fl._C_ascend")
        if spec is None:
            raise ImportError("vllm_fl._C_ascend not found")
    except Exception as exc:
        print(f"Cannot import vllm_fl._C_ascend: {exc}")
        print("Rebuild with: VLLM_VENDOR=ascend pip install -e ./vllm-plugin-FL")
        return 2

    results: dict[str, tuple[str, str]] = {}
    current_soc = _get_soc_name()

    for op_name, body in SINGLE_CARD_OPS.items():
        supported_socs = OP_SOC_SUPPORT.get(op_name)
        if supported_socs is not None and current_soc not in supported_socs:
            reason = f"not supported on {current_soc} (requires {', '.join(sorted(supported_socs))})"
            results[op_name] = ("SKIP", reason)
            print(f"\nTesting {op_name} ...\n  -> SKIP ({reason})")
            continue

        print(f"\nTesting {op_name} ...", flush=True)
        status, detail = _run_single(op_name, body)
        results[op_name] = (status, detail)
        print(f"  -> {status}")
        if status != "OK" and detail:
            print(f"     {detail.splitlines()[-1][:200]}")

    for op_name, reason in SKIPPED_OPS.items():
        results[op_name] = ("SKIP", reason)
        print(f"\nTesting {op_name} ...\n  -> SKIP ({reason})")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    ok = sum(1 for s, _ in results.values() if s == "OK")
    failed = sum(1 for s, _ in results.values() if s == "FAIL")
    skipped = sum(1 for s, _ in results.values() if s == "SKIP")
    timeout = sum(1 for s, _ in results.values() if s == "TIMEOUT")

    for op_name, (status, detail) in results.items():
        badge = {"OK": "PASS", "FAIL": "FAIL", "SKIP": "SKIP", "TIMEOUT": "T/O"}[status]
        print(f"[{badge:4}] {op_name}")
        if status in ("FAIL", "TIMEOUT") and detail:
            print(f"       {detail.splitlines()[-1][:160]}")

    print("-" * 70)
    print(f"Total: {len(results)} | PASS: {ok} | FAIL: {failed} | SKIP: {skipped} | TIMEOUT: {timeout}")
    return 0 if failed == 0 and timeout == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
