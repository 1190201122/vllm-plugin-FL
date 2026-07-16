# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for vllm_fl.utils module.
"""

import os
from unittest.mock import patch

import pytest

from vllm_fl.utils import (
    VENDOR_DEVICE_MAP,
    _get_vendor_device_field,
    get_device_name,
    get_device_type,
    get_flag_gems_whitelist_blacklist,
    get_oot_blacklist,
    get_oot_whitelist,
    get_op_config,
    is_oot_enabled,
    use_flaggems,
    use_flaggems_op,
)


class TestVendorDeviceMap:
    def test_known_vendors_present(self):
        assert "nvidia" in VENDOR_DEVICE_MAP
        assert "ascend" in VENDOR_DEVICE_MAP
        assert "iluvatar" in VENDOR_DEVICE_MAP
        assert "kunlunxin" in VENDOR_DEVICE_MAP

    def test_nvidia_device_type(self):
        assert VENDOR_DEVICE_MAP["nvidia"]["device_type"] == "cuda"
        assert VENDOR_DEVICE_MAP["nvidia"]["device_name"] == "nvidia"

    def test_ascend_device_type(self):
        assert VENDOR_DEVICE_MAP["ascend"]["device_type"] == "npu"
        assert VENDOR_DEVICE_MAP["ascend"]["device_name"] == "npu"


class TestGetVendorDeviceField:
    def test_valid_vendor_and_field(self):
        assert _get_vendor_device_field("nvidia", "device_type") == "cuda"
        assert _get_vendor_device_field("ascend", "device_name") == "npu"

    def test_empty_vendor_name_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            _get_vendor_device_field("", "device_type")

    def test_whitespace_vendor_name_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            _get_vendor_device_field("   ", "device_type")

    def test_non_string_vendor_name_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            _get_vendor_device_field(None, "device_type")  # type: ignore

    def test_unknown_vendor_raises(self):
        with pytest.raises(ValueError, match="not found in VENDOR_DEVICE_MAP"):
            _get_vendor_device_field("unknown_vendor", "device_type")

    def test_missing_field_raises(self):
        with (
            patch.dict(VENDOR_DEVICE_MAP, {"test_vendor": {"device_type": "cuda"}}),
            pytest.raises(ValueError, match="missing or empty"),
        ):
            _get_vendor_device_field("test_vendor", "device_name")


class TestGetDeviceType:
    def test_nvidia(self):
        assert get_device_type("nvidia") == "cuda"

    def test_ascend(self):
        assert get_device_type("ascend") == "npu"

    def test_kunlunxin(self):
        assert get_device_type("kunlunxin") == "cuda"


class TestGetDeviceName:
    def test_nvidia(self):
        assert get_device_name("nvidia") == "nvidia"

    def test_ascend(self):
        assert get_device_name("ascend") == "npu"

    def test_iluvatar(self):
        assert get_device_name("iluvatar") == "cuda"


class TestUseFlaggems:
    def test_default_true(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("USE_FLAGGEMS", None)
            os.environ.pop("VLLM_FL_PREFER_ENABLED", None)
            os.environ.pop("VLLM_FL_PREFER", None)
            assert use_flaggems() is True

    def test_prefer_enabled_false_returns_false(self):
        with patch.dict(os.environ, {"VLLM_FL_PREFER_ENABLED": "false"}):
            os.environ.pop("USE_FLAGGEMS", None)
            os.environ.pop("VLLM_FL_PREFER", None)
            assert use_flaggems() is False

    def test_prefer_vendor_returns_false(self):
        with patch.dict(os.environ, {"VLLM_FL_PREFER": "vendor"}):
            os.environ.pop("USE_FLAGGEMS", None)
            os.environ.pop("VLLM_FL_PREFER_ENABLED", None)
            assert use_flaggems() is False

    def test_use_flaggems_explicit_true(self):
        with patch.dict(os.environ, {"USE_FLAGGEMS": "true"}):
            os.environ.pop("VLLM_FL_PREFER_ENABLED", None)
            os.environ.pop("VLLM_FL_PREFER", None)
            assert use_flaggems() is True

    def test_use_flaggems_explicit_false(self):
        with patch.dict(os.environ, {"USE_FLAGGEMS": "false"}):
            os.environ.pop("VLLM_FL_PREFER_ENABLED", None)
            os.environ.pop("VLLM_FL_PREFER", None)
            assert use_flaggems() is False

    def test_use_flaggems_explicit_one(self):
        with patch.dict(os.environ, {"USE_FLAGGEMS": "1"}):
            os.environ.pop("VLLM_FL_PREFER_ENABLED", None)
            os.environ.pop("VLLM_FL_PREFER", None)
            assert use_flaggems() is True

    def test_default_param_false(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("USE_FLAGGEMS", None)
            os.environ.pop("VLLM_FL_PREFER_ENABLED", None)
            os.environ.pop("VLLM_FL_PREFER", None)
            assert use_flaggems(default=False) is False


class TestGetFlagGemsWhitelistBlacklist:
    def test_no_env_vars_returns_none_none(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_FL_FLAGOS_WHITELIST", None)
            os.environ.pop("VLLM_FL_FLAGOS_BLACKLIST", None)
            wl, bl = get_flag_gems_whitelist_blacklist()
            # May return config-based blacklist or None
            assert wl is None

    def test_whitelist_set(self):
        with patch.dict(os.environ, {"VLLM_FL_FLAGOS_WHITELIST": "silu,rms_norm"}):
            os.environ.pop("VLLM_FL_FLAGOS_BLACKLIST", None)
            wl, bl = get_flag_gems_whitelist_blacklist()
            assert wl == ["silu", "rms_norm"]
            assert bl is None

    def test_blacklist_set(self):
        with patch.dict(os.environ, {"VLLM_FL_FLAGOS_BLACKLIST": "fused_moe,rotary"}):
            os.environ.pop("VLLM_FL_FLAGOS_WHITELIST", None)
            wl, bl = get_flag_gems_whitelist_blacklist()
            assert wl is None
            assert bl == ["fused_moe", "rotary"]

    def test_both_set_raises(self):
        with (
            patch.dict(
                os.environ,
                {
                    "VLLM_FL_FLAGOS_WHITELIST": "silu",
                    "VLLM_FL_FLAGOS_BLACKLIST": "rms_norm",
                },
            ),
            pytest.raises(ValueError, match="Cannot set both"),
        ):
            get_flag_gems_whitelist_blacklist()


_FLAGGEMS_ENABLED_ENV = {
    "VLLM_FL_PREFER_ENABLED": "True",
    "VLLM_FL_PREFER": "flagos",
    "USE_FLAGGEMS": "true",
}


class TestUseFlaggemsOp:
    def test_op_in_whitelist(self):
        with patch.dict(
            os.environ,
            {**_FLAGGEMS_ENABLED_ENV, "VLLM_FL_FLAGOS_WHITELIST": "silu,rms_norm"},
        ):
            os.environ.pop("VLLM_FL_FLAGOS_BLACKLIST", None)
            assert use_flaggems_op("silu") is True
            assert use_flaggems_op("fused_moe") is False

    def test_op_in_blacklist(self):
        with patch.dict(
            os.environ,
            {**_FLAGGEMS_ENABLED_ENV, "VLLM_FL_FLAGOS_BLACKLIST": "fused_moe"},
        ):
            os.environ.pop("VLLM_FL_FLAGOS_WHITELIST", None)
            assert use_flaggems_op("fused_moe") is False
            assert use_flaggems_op("silu") is True

    def test_no_lists_returns_default(self):
        with (
            patch.dict(
                os.environ,
                {"VLLM_FL_PREFER_ENABLED": "True", "VLLM_FL_PREFER": "flagos"},
            ),
            patch("vllm_fl.dispatch.config.get_flagos_blacklist", return_value=None),
        ):
            os.environ.pop("USE_FLAGGEMS", None)
            os.environ.pop("VLLM_FL_FLAGOS_WHITELIST", None)
            os.environ.pop("VLLM_FL_FLAGOS_BLACKLIST", None)
            assert use_flaggems_op("silu") is True
            assert use_flaggems_op("silu", default=False) is False


class TestGetOotWhitelist:
    def test_not_set_returns_none(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_FL_OOT_WHITELIST", None)
            assert get_oot_whitelist() is None

    def test_set_returns_list(self):
        with patch.dict(os.environ, {"VLLM_FL_OOT_WHITELIST": "silu_and_mul,rms_norm"}):
            result = get_oot_whitelist()
            assert result == ["silu_and_mul", "rms_norm"]

    def test_empty_string_returns_none(self):
        with patch.dict(os.environ, {"VLLM_FL_OOT_WHITELIST": ""}):
            assert get_oot_whitelist() is None


class TestGetOotBlacklist:
    def test_whitelist_set_returns_none(self):
        with patch.dict(os.environ, {"VLLM_FL_OOT_WHITELIST": "silu"}):
            os.environ.pop("VLLM_FL_OOT_BLACKLIST", None)
            assert get_oot_blacklist() is None

    def test_blacklist_set(self):
        with patch.dict(os.environ, {"VLLM_FL_OOT_BLACKLIST": "fused_moe,rotary"}):
            os.environ.pop("VLLM_FL_OOT_WHITELIST", None)
            result = get_oot_blacklist()
            assert result == ["fused_moe", "rotary"]

    def test_neither_set(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_FL_OOT_WHITELIST", None)
            os.environ.pop("VLLM_FL_OOT_BLACKLIST", None)
            # May return config-based blacklist or None
            result = get_oot_blacklist()
            assert result is None or isinstance(result, list)


class TestIsOotEnabled:
    def test_default_enabled(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_FL_OOT_ENABLED", None)
            os.environ.pop("VLLM_FL_PREFER_ENABLED", None)
            assert is_oot_enabled() is True

    def test_disabled_via_oot_enabled(self):
        with patch.dict(os.environ, {"VLLM_FL_OOT_ENABLED": "0"}):
            os.environ.pop("VLLM_FL_PREFER_ENABLED", None)
            assert is_oot_enabled() is False

    def test_disabled_via_prefer_enabled(self):
        with patch.dict(os.environ, {"VLLM_FL_PREFER_ENABLED": "false"}):
            os.environ.pop("VLLM_FL_OOT_ENABLED", None)
            assert is_oot_enabled() is False

    def test_enabled_explicit(self):
        with patch.dict(os.environ, {"VLLM_FL_OOT_ENABLED": "true"}):
            os.environ.pop("VLLM_FL_PREFER_ENABLED", None)
            assert is_oot_enabled() is True


class TestGetOpConfig:
    def test_returns_dict_or_none(self):
        result = get_op_config()
        assert result is None or isinstance(result, dict)

    def test_with_env_config(self, tmp_path):
        import json

        config = {"silu_and_mul": "flagos", "rms_norm": "vendor"}
        config_file = tmp_path / "op_config.json"
        config_file.write_text(json.dumps(config))
        with patch.dict(os.environ, {"VLLM_FL_OP_CONFIG": str(config_file)}):
            from vllm_fl import utils

            utils._load_op_config_from_env()
            result = get_op_config()
            assert result == config
            utils._OP_CONFIG = None  # cleanup
