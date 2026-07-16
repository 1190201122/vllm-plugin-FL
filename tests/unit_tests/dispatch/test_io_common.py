# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for vllm_fl.dispatch.io_common module - utility functions.
"""

import os
from unittest.mock import MagicMock, patch

import torch

from vllm_fl.dispatch.io_common import (
    ModeManager,
    advance_step,
    expand_layer_specs,
    format_result,
    format_value,
    get_current_module,
    get_exec_order,
    get_rank,
    get_step,
    is_io_active,
    make_guard,
    make_module_tag_from_ctx,
    module_context_matches,
    next_exec_order,
    parse_rank_filter,
    parse_step_range,
    parse_step_range_env,
    parse_torch_funcs_config,
    pop_module_context,
    push_module_context,
    register_step_callback,
    reset_exec_order,
    reset_rank,
    reset_step,
    set_io_active,
    should_inspect_dispatch_op,
    should_inspect_torch_func,
    tensor_stats,
    unregister_step_callback,
)


class TestModeManager:
    def test_enter_and_is_entered(self):
        mgr = ModeManager()
        mode = MagicMock()
        mode.__enter__ = MagicMock(return_value=mode)
        mode.__exit__ = MagicMock(return_value=False)

        mgr.enter("test", mode)
        assert mgr.is_entered("test") is True
        assert mgr.is_entered("other") is False

    def test_enter_idempotent(self):
        mgr = ModeManager()
        mode = MagicMock()
        mode.__enter__ = MagicMock(return_value=mode)

        mgr.enter("test", mode)
        mgr.enter("test", mode)
        assert mode.__enter__.call_count == 1

    def test_request_exit(self):
        mgr = ModeManager()
        mode = MagicMock()
        mode.__enter__ = MagicMock(return_value=mode)
        mode.__exit__ = MagicMock(return_value=False)

        mgr.enter("test", mode)
        mgr.request_exit("test")
        assert mgr.is_entered("test") is False

    def test_request_exit_nonexistent(self):
        mgr = ModeManager()
        mgr.request_exit("nonexistent")

    def test_exit_all(self):
        mgr = ModeManager()
        mode1 = MagicMock()
        mode1.__enter__ = MagicMock(return_value=mode1)
        mode1.__exit__ = MagicMock(return_value=False)
        mode2 = MagicMock()
        mode2.__enter__ = MagicMock(return_value=mode2)
        mode2.__exit__ = MagicMock(return_value=False)

        mgr.enter("a", mode1)
        mgr.enter("b", mode2)
        mgr.exit_all()
        assert mgr.is_entered("a") is False
        assert mgr.is_entered("b") is False


class TestIoActive:
    def test_default_inactive(self):
        set_io_active(False)
        assert is_io_active() is False

    def test_set_active(self):
        set_io_active(True)
        assert is_io_active() is True
        set_io_active(False)


class TestMakeGuard:
    def test_guard_initially_inactive(self):
        guard_active, set_guard = make_guard()
        assert guard_active() is False

    def test_set_guard_active(self):
        guard_active, set_guard = make_guard()
        set_guard(True)
        assert guard_active() is True
        set_guard(False)
        assert guard_active() is False

    def test_independent_guards(self):
        guard1_active, set_guard1 = make_guard()
        guard2_active, set_guard2 = make_guard()
        set_guard1(True)
        assert guard1_active() is True
        assert guard2_active() is False


class TestExecOrder:
    def setup_method(self):
        reset_exec_order()

    def test_next_exec_order_increments(self):
        assert next_exec_order() == 1
        assert next_exec_order() == 2
        assert next_exec_order() == 3

    def test_reset_exec_order(self):
        next_exec_order()
        next_exec_order()
        reset_exec_order()
        assert get_exec_order() == 0

    def test_get_exec_order_no_increment(self):
        reset_exec_order()
        next_exec_order()
        val = get_exec_order()
        assert val == get_exec_order()


class TestStepTracking:
    def setup_method(self):
        reset_step()

    def test_get_step_initial(self):
        assert get_step() == 0

    def test_advance_step(self):
        result = advance_step()
        assert result == 1
        assert get_step() == 1

    def test_reset_step(self):
        advance_step()
        advance_step()
        reset_step()
        assert get_step() == 0

    def test_step_callback(self):
        called = []

        def cb(step, seen_modules, seen_ops):
            called.append(step)

        register_step_callback(cb)
        advance_step()
        assert called == [0]
        unregister_step_callback(cb)

    def test_unregister_nonexistent_callback(self):
        unregister_step_callback(lambda s, m, o: None)


class TestParseStepRange:
    def test_dash_format(self):
        assert parse_step_range("0-2") == (0, 3)
        assert parse_step_range("5-10") == (5, 11)

    def test_single_integer(self):
        assert parse_step_range("5") == (5, 6)
        assert parse_step_range("0") == (0, 1)

    def test_none_returns_none(self):
        assert parse_step_range(None) is None

    def test_empty_returns_none(self):
        assert parse_step_range("") is None

    def test_non_string_returns_none(self):
        assert parse_step_range(123) is None  # type: ignore

    def test_invalid_format_returns_none(self):
        assert parse_step_range("abc") is None


class TestParseStepRangeEnv:
    def test_from_env_var(self):
        with patch.dict(os.environ, {"TEST_STEP": "0-2"}):
            result = parse_step_range_env("TEST_STEP")
            assert result == (0, 3)

    def test_empty_env_var(self):
        with patch.dict(os.environ, {"TEST_STEP": ""}):
            result = parse_step_range_env("TEST_STEP")
            assert result is None

    def test_missing_env_var(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TEST_STEP_MISSING", None)
            result = parse_step_range_env("TEST_STEP_MISSING")
            assert result is None


class TestParseRankFilter:
    def test_all_returns_none(self):
        assert parse_rank_filter("all") is None
        assert parse_rank_filter("ALL") is None

    def test_empty_returns_none(self):
        assert parse_rank_filter("") is None

    def test_single_rank(self):
        assert parse_rank_filter("0") == {0}

    def test_multiple_ranks(self):
        assert parse_rank_filter("0,2,4") == {0, 2, 4}

    def test_invalid_values_skipped(self):
        assert parse_rank_filter("0,abc,2") == {0, 2}


class TestExpandLayerSpecs:
    def test_integer_expansion(self):
        result = expand_layer_specs({"0", "1"})
        assert "model.layers.0" in result
        assert "model.layers.1" in result

    def test_range_expansion(self):
        result = expand_layer_specs({"0-2"})
        assert "model.layers.0" in result
        assert "model.layers.1" in result
        assert "model.layers.2" in result

    def test_full_path_kept(self):
        result = expand_layer_specs({"model.layers.0.self_attn"})
        assert "model.layers.0.self_attn" in result

    def test_glob_pattern_kept(self):
        result = expand_layer_specs({"model.layers.*.self_attn"})
        assert "model.layers.*.self_attn" in result

    def test_empty_spec_skipped(self):
        result = expand_layer_specs({"", "0"})
        assert "model.layers.0" in result
        assert len(result) == 1

    def test_custom_prefix(self):
        result = expand_layer_specs({"0"}, prefix="encoder.layers.")
        assert "encoder.layers.0" in result


class TestParseTorchFuncsConfig:
    def test_zero_disabled(self):
        enabled, funcs = parse_torch_funcs_config("0")
        assert enabled is False
        assert funcs == set()

    def test_one_enables_all(self):
        enabled, funcs = parse_torch_funcs_config("1")
        assert enabled is True
        assert funcs == set()

    def test_specific_funcs(self):
        enabled, funcs = parse_torch_funcs_config("matmul,softmax")
        assert enabled is True
        assert funcs == {"matmul", "softmax"}

    def test_empty_disabled(self):
        enabled, funcs = parse_torch_funcs_config("")
        assert enabled is False


class TestShouldInspectTorchFunc:
    def test_disabled_returns_false(self):
        assert should_inspect_torch_func("matmul", False, set(), True, set()) is False

    def test_in_filter(self):
        assert (
            should_inspect_torch_func("matmul", True, {"matmul"}, True, set()) is True
        )
        assert (
            should_inspect_torch_func("softmax", True, {"matmul"}, True, set()) is False
        )

    def test_skip_dunder(self):
        assert should_inspect_torch_func("__add__", True, set(), True, set()) is False

    def test_skip_trivial(self):
        assert should_inspect_torch_func("size", True, set(), True, set()) is False


class TestShouldInspectDispatchOp:
    def test_match_all(self):
        assert should_inspect_dispatch_op("mm", True, set()) is True

    def test_in_filter(self):
        assert should_inspect_dispatch_op("mm", False, {"mm", "add"}) is True
        assert should_inspect_dispatch_op("sub", False, {"mm", "add"}) is False

    def test_empty_filter_matches_all(self):
        assert should_inspect_dispatch_op("anything", False, set()) is True


class TestModuleContext:
    def setup_method(self):
        import vllm_fl.dispatch.io_common as io_common

        stack = getattr(io_common._module_context, "stack", None)
        if stack:
            stack.clear()

    def test_push_pop(self):
        push_module_context("Linear")
        assert get_current_module() == "Linear"
        pop_module_context()
        assert get_current_module() is None

    def test_nested_context(self):
        push_module_context("Attention")
        push_module_context("Linear")
        assert get_current_module() == "Linear"
        pop_module_context()
        assert get_current_module() == "Attention"
        pop_module_context()

    def test_module_context_matches(self):
        push_module_context("Linear")
        assert module_context_matches({"Linear"}) is True
        assert module_context_matches({"Conv2d"}) is False
        pop_module_context()


class TestMakeModuleTagFromCtx:
    def test_with_name_and_path(self):
        result = make_module_tag_from_ctx(
            "Linear", "model.layers.0.q_proj", for_json=True
        )
        assert result == "Linear:model.layers.0.q_proj"

    def test_with_name_only(self):
        result = make_module_tag_from_ctx("Linear", "", for_json=True)
        assert result == "Linear"

    def test_empty_name(self):
        result = make_module_tag_from_ctx("", "some.path")
        assert result == ""

    def test_display_format(self):
        result = make_module_tag_from_ctx("Linear", "model.layers.0", for_json=False)
        assert result == "[Linear:model.layers.0]"


class TestFormatValue:
    def test_tensor(self):
        t = torch.randn(2, 3)
        result = format_value(t)
        assert "Tensor(" in result
        assert "shape=[2, 3]" in result

    def test_none(self):
        assert format_value(None) == "None"

    def test_bool(self):
        assert format_value(True) == "True"

    def test_int(self):
        assert format_value(42) == "42"

    def test_small_list(self):
        result = format_value([1, 2, 3])
        assert "list" in result

    def test_large_list(self):
        result = format_value(list(range(10)))
        assert "len=10" in result


class TestFormatResult:
    def test_single_tensor(self):
        t = torch.randn(2, 3)
        result = format_result(t)
        assert "result:" in result

    def test_tuple_result(self):
        result = format_result((torch.randn(2), torch.randn(3)))
        assert "result[0]" in result
        assert "result[1]" in result


class TestTensorStats:
    def test_basic_stats(self):
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        stats = tensor_stats(t)
        assert stats["shape"] == [4]
        assert "float" in stats["dtype"]
        assert "min" in stats
        assert "max" in stats
        assert stats["min"] == 1.0
        assert stats["max"] == 4.0

    def test_empty_tensor(self):
        t = torch.tensor([])
        stats = tensor_stats(t)
        assert stats["shape"] == [0]
        assert "min" not in stats

    def test_integer_tensor(self):
        t = torch.tensor([1, 2, 3])
        stats = tensor_stats(t)
        assert "min" in stats
        assert "mean" not in stats


class TestGetRank:
    def setup_method(self):
        reset_rank()

    def test_default_rank_zero(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RANK", None)
            os.environ.pop("LOCAL_RANK", None)
            reset_rank()
            assert get_rank() == 0

    def test_rank_from_env(self):
        with patch.dict(os.environ, {"RANK": "3"}):
            reset_rank()
            assert get_rank() == 3

    def test_local_rank_fallback(self):
        with patch.dict(os.environ, {"LOCAL_RANK": "2"}):
            os.environ.pop("RANK", None)
            reset_rank()
            assert get_rank() == 2

    def teardown_method(self):
        reset_rank()
