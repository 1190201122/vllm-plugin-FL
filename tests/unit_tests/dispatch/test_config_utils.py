# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for vllm_fl.dispatch.config.utils module.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from vllm_fl.dispatch.config.utils import (
    get_config_path,
    get_effective_config,
    get_flagos_blacklist,
    get_oot_blacklist,
    get_per_op_order,
    get_vendor_device_map,
    load_platform_config,
)


class TestGetConfigPath:
    def test_returns_path_for_known_platform(self):
        path = get_config_path("ascend")
        if path is not None:
            assert path.name == "ascend.yaml"
            assert path.exists()

    def test_returns_none_for_unknown_platform(self):
        path = get_config_path("nonexistent_platform_xyz")
        assert path is None

    def test_auto_detect_platform(self):
        with patch(
            "vllm_fl.dispatch.config.utils.get_platform_name",
            return_value="nonexistent",
        ):
            path = get_config_path()
            assert path is None


class TestLoadPlatformConfig:
    def test_load_known_platform(self):
        config = load_platform_config("ascend")
        if config is not None:
            assert isinstance(config, dict)

    def test_load_unknown_platform_returns_none(self):
        config = load_platform_config("nonexistent_platform_xyz")
        assert config is None

    def test_load_with_invalid_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(": invalid: yaml: [")
            f.flush()
            config_path = Path(f.name)

        try:
            with patch(
                "vllm_fl.dispatch.config.utils.get_config_path",
                return_value=config_path,
            ):
                config = load_platform_config("test")
                assert config is None
        finally:
            os.unlink(config_path)


class TestGetPerOpOrder:
    def test_with_valid_config(self):
        config = {
            "per_op": {
                "silu_and_mul": ["vendor", "flagos"],
                "rms_norm": ["flagos", "reference"],
            }
        }
        result = get_per_op_order(config)
        assert result == {
            "silu_and_mul": ["vendor", "flagos"],
            "rms_norm": ["flagos", "reference"],
        }

    def test_with_string_value(self):
        config = {"per_op": {"silu_and_mul": "vendor"}}
        result = get_per_op_order(config)
        assert result == {"silu_and_mul": ["vendor"]}

    def test_with_no_per_op(self):
        config = {"other_key": "value"}
        result = get_per_op_order(config)
        assert result is None

    def test_with_none_config_and_no_platform(self):
        with patch(
            "vllm_fl.dispatch.config.utils.load_platform_config", return_value=None
        ):
            result = get_per_op_order(None)
            assert result is None

    def test_with_invalid_per_op_type(self):
        config = {"per_op": "not_a_dict"}
        result = get_per_op_order(config)
        assert result is None

    def test_with_empty_per_op(self):
        config = {"per_op": {}}
        result = get_per_op_order(config)
        assert result is None


class TestGetFlagosBlacklist:
    def test_with_valid_blacklist(self):
        config = {"flagos_blacklist": ["fused_moe", "rotary_embedding"]}
        result = get_flagos_blacklist(config)
        assert result == ["fused_moe", "rotary_embedding"]

    def test_with_no_blacklist(self):
        config = {"other_key": "value"}
        result = get_flagos_blacklist(config)
        # Returns None when key is absent (empty list from default)
        assert result is None or result == []

    def test_with_empty_blacklist(self):
        config = {"flagos_blacklist": []}
        result = get_flagos_blacklist(config)
        assert result == []

    def test_with_none_config(self):
        with patch(
            "vllm_fl.dispatch.config.utils.load_platform_config", return_value=None
        ):
            result = get_flagos_blacklist(None)
            assert result is None

    def test_with_non_list_blacklist(self):
        config = {"flagos_blacklist": "not_a_list"}
        result = get_flagos_blacklist(config)
        assert result is None


class TestGetOotBlacklist:
    def test_with_valid_blacklist(self):
        config = {"oot_blacklist": ["silu_and_mul"]}
        result = get_oot_blacklist(config)
        assert result == ["silu_and_mul"]

    def test_with_no_blacklist(self):
        config = {}
        result = get_oot_blacklist(config)
        assert result is None or result == []

    def test_with_none_config(self):
        with patch(
            "vllm_fl.dispatch.config.utils.load_platform_config", return_value=None
        ):
            result = get_oot_blacklist(None)
            assert result is None


class TestGetEffectiveConfig:
    def test_user_config_from_env(self):
        config_data = {"prefer": "vendor", "strict": True}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()
            config_path = f.name

        try:
            with patch.dict(os.environ, {"VLLM_FL_CONFIG": config_path}):
                result = get_effective_config()
                assert result["prefer"] == "vendor"
                assert result["strict"] is True
        finally:
            os.unlink(config_path)

    def test_nonexistent_user_config_falls_through(self):
        with (
            patch.dict(os.environ, {"VLLM_FL_CONFIG": "/nonexistent/path.yaml"}),
            patch(
                "vllm_fl.dispatch.config.utils.load_platform_config", return_value=None
            ),
        ):
            result = get_effective_config()
            assert result == {}

    def test_empty_env_var(self):
        with (
            patch.dict(os.environ, {"VLLM_FL_CONFIG": ""}),
            patch(
                "vllm_fl.dispatch.config.utils.load_platform_config", return_value=None
            ),
        ):
            result = get_effective_config()
            assert result == {}

    def test_platform_config_fallback(self):
        platform_config = {"prefer": "flagos"}
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_FL_CONFIG", None)
            with patch(
                "vllm_fl.dispatch.config.utils.load_platform_config",
                return_value=platform_config,
            ):
                result = get_effective_config()
                assert result == platform_config


class TestGetVendorDeviceMap:
    def test_returns_dict(self):
        result = get_vendor_device_map()
        assert isinstance(result, dict)

    def test_contains_known_vendors(self):
        result = get_vendor_device_map()
        assert "nvidia" in result
        assert "ascend" in result

    def test_entries_have_required_fields(self):
        result = get_vendor_device_map()
        for vendor, info in result.items():
            assert "device_type" in info
            assert "device_name" in info
            assert isinstance(info["device_type"], str)
            assert isinstance(info["device_name"], str)
