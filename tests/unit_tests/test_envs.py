# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for vllm_fl.envs module.
"""

import os
from unittest.mock import patch

import pytest


class TestEnvsGetattr:
    def test_vllm_fl_prefer_enabled_default_true(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_FL_PREFER_ENABLED", None)
            import importlib

            import vllm_fl.envs as envs

            importlib.reload(envs)
            assert envs.VLLM_FL_PREFER_ENABLED is True

    def test_vllm_fl_prefer_enabled_explicit_false(self):
        with patch.dict(os.environ, {"VLLM_FL_PREFER_ENABLED": "false"}):
            import importlib

            import vllm_fl.envs as envs

            importlib.reload(envs)
            assert envs.VLLM_FL_PREFER_ENABLED is False

    def test_vllm_fl_prefer_enabled_explicit_zero(self):
        with patch.dict(os.environ, {"VLLM_FL_PREFER_ENABLED": "0"}):
            import importlib

            import vllm_fl.envs as envs

            importlib.reload(envs)
            assert envs.VLLM_FL_PREFER_ENABLED is False

    def test_vllm_fl_prefer_enabled_explicit_one(self):
        with patch.dict(os.environ, {"VLLM_FL_PREFER_ENABLED": "1"}):
            import importlib

            import vllm_fl.envs as envs

            importlib.reload(envs)
            assert envs.VLLM_FL_PREFER_ENABLED is True

    def test_flaggems_enable_oplist_path_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FLAGGEMS_ENABLE_OPLIST_PATH", None)
            import importlib

            import vllm_fl.envs as envs

            importlib.reload(envs)
            assert envs.FLAGGEMS_ENABLE_OPLIST_PATH == "/tmp/flaggems_enable_oplist.txt"

    def test_flaggems_enable_oplist_path_custom(self):
        with patch.dict(
            os.environ, {"FLAGGEMS_ENABLE_OPLIST_PATH": "/custom/path.txt"}
        ):
            import importlib

            import vllm_fl.envs as envs

            importlib.reload(envs)
            assert envs.FLAGGEMS_ENABLE_OPLIST_PATH == "/custom/path.txt"

    def test_getattr_unknown_raises(self):
        import vllm_fl.envs as envs

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = envs.NONEXISTENT_VAR


class TestEnvsDir:
    def test_dir_returns_variable_names(self):
        import vllm_fl.envs as envs

        result = dir(envs)
        assert "VLLM_FL_PREFER_ENABLED" in result
        assert "FLAGGEMS_ENABLE_OPLIST_PATH" in result
        assert "USE_FLAGGEMS" in result


class TestEnvsIsSet:
    def test_is_set_when_present(self):
        with patch.dict(os.environ, {"VLLM_FL_PREFER_ENABLED": "true"}):
            import importlib

            import vllm_fl.envs as envs

            importlib.reload(envs)
            assert envs.is_set("VLLM_FL_PREFER_ENABLED") is True

    def test_is_set_when_absent(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_FL_PREFER_ENABLED", None)
            import importlib

            import vllm_fl.envs as envs

            importlib.reload(envs)
            assert envs.is_set("VLLM_FL_PREFER_ENABLED") is False

    def test_is_set_unknown_raises(self):
        import vllm_fl.envs as envs

        with pytest.raises(AttributeError, match="has no attribute"):
            envs.is_set("NONEXISTENT_VAR")
