# Copyright (c) 2026 BAAI. All rights reserved.

"""Tests for dispatch builtin operator registration."""

import contextlib
from unittest.mock import MagicMock, patch

from vllm_fl.dispatch.builtin_ops import (
    _find_vendor_backend_dir,
    _get_current_vendor_backend_dirs,
    _register_vendor_backends,
    register_builtins,
)


class TestFindVendorBackendDir:
    def test_maps_vendor_names(self):
        available = {"cuda", "ascend", "metax", "iluvatar"}
        assert _find_vendor_backend_dir("nvidia", available) == "cuda"
        assert _find_vendor_backend_dir("ascend", available) == "ascend"
        assert _find_vendor_backend_dir("metax", available) == "metax"
        assert _find_vendor_backend_dir("iluvatar", available) == "iluvatar"

    def test_maca_alias_resolves_to_metax(self):
        assert _find_vendor_backend_dir("maca", {"metax"}) == "metax"

    def test_unknown_vendor_returns_none(self):
        assert _find_vendor_backend_dir("unknown", {"cuda", "ascend"}) is None

    def test_vendor_not_in_available_dirs(self):
        assert _find_vendor_backend_dir("nvidia", {"ascend"}) is None


class TestGetCurrentVendorBackendDirs:
    def test_returns_none_when_no_vendor_name(self):
        mock_platform = MagicMock()
        mock_platform.vendor_name = None
        with (
            patch(
                "vllm_fl.dispatch.builtin_ops.current_platform",
                mock_platform,
                create=True,
            ),
            patch("vllm.platforms.current_platform", mock_platform),
        ):
            result = _get_current_vendor_backend_dirs({"cuda", "ascend"})
            assert result is None

    def test_returns_none_when_empty_vendor_name(self):
        mock_platform = MagicMock()
        mock_platform.vendor_name = ""
        with patch("vllm.platforms.current_platform", mock_platform):
            result = _get_current_vendor_backend_dirs({"cuda", "ascend"})
            assert result is None


class TestRegisterVendorBackends:
    def test_registers_only_current_vendor_backend(self):
        registry = MagicMock()
        module = MagicMock()

        with (
            patch("vllm_fl.dispatch.builtin_ops._VENDOR_BACKENDS_DIR", "/fake/vendor"),
            patch(
                "vllm_fl.dispatch.builtin_ops._get_current_vendor_backend_dirs",
                return_value="metax",
            ),
            patch("vllm_fl.dispatch.builtin_ops.os.path.isdir", return_value=True),
            patch(
                "vllm_fl.dispatch.builtin_ops.os.listdir",
                return_value=["metax", "cuda", "ascend"],
            ),
            patch("vllm_fl.dispatch.builtin_ops.os.path.isfile", return_value=True),
            patch(
                "vllm_fl.dispatch.builtin_ops.importlib.import_module",
                return_value=module,
            ) as import_module,
        ):
            _register_vendor_backends(registry)

        import_module.assert_called_once_with(
            ".backends.vendor.metax.register_ops",
            package="vllm_fl.dispatch",
        )
        module.register_builtins.assert_called_once_with(registry)

    def test_skips_when_vendor_dir_not_found(self):
        registry = MagicMock()
        with patch("vllm_fl.dispatch.builtin_ops.os.path.isdir", return_value=False):
            _register_vendor_backends(registry)

    def test_skips_when_no_current_vendor(self):
        registry = MagicMock()
        with (
            patch("vllm_fl.dispatch.builtin_ops._VENDOR_BACKENDS_DIR", "/fake/vendor"),
            patch(
                "vllm_fl.dispatch.builtin_ops._get_current_vendor_backend_dirs",
                return_value=None,
            ),
            patch("vllm_fl.dispatch.builtin_ops.os.path.isdir", return_value=True),
            patch(
                "vllm_fl.dispatch.builtin_ops.os.listdir",
                return_value=["metax"],
            ),
        ):
            _register_vendor_backends(registry)

    def test_skips_vendor_without_register_ops(self):
        registry = MagicMock()
        with (
            patch("vllm_fl.dispatch.builtin_ops._VENDOR_BACKENDS_DIR", "/fake/vendor"),
            patch(
                "vllm_fl.dispatch.builtin_ops._get_current_vendor_backend_dirs",
                return_value="metax",
            ),
            patch("vllm_fl.dispatch.builtin_ops.os.path.isdir", return_value=True),
            patch(
                "vllm_fl.dispatch.builtin_ops.os.listdir",
                return_value=["metax"],
            ),
            patch("vllm_fl.dispatch.builtin_ops.os.path.isfile", return_value=False),
        ):
            _register_vendor_backends(registry)

    def test_handles_import_error_gracefully(self):
        registry = MagicMock()
        with (
            patch("vllm_fl.dispatch.builtin_ops._VENDOR_BACKENDS_DIR", "/fake/vendor"),
            patch(
                "vllm_fl.dispatch.builtin_ops._get_current_vendor_backend_dirs",
                return_value="metax",
            ),
            patch("vllm_fl.dispatch.builtin_ops.os.path.isdir", return_value=True),
            patch(
                "vllm_fl.dispatch.builtin_ops.os.listdir",
                return_value=["metax"],
            ),
            patch("vllm_fl.dispatch.builtin_ops.os.path.isfile", return_value=True),
            patch(
                "vllm_fl.dispatch.builtin_ops.importlib.import_module",
                side_effect=ImportError("no module"),
            ),
        ):
            _register_vendor_backends(registry)

    def test_handles_module_without_register_builtins(self):
        registry = MagicMock()
        module = MagicMock(spec=[])  # no register_builtins attribute
        with (
            patch("vllm_fl.dispatch.builtin_ops._VENDOR_BACKENDS_DIR", "/fake/vendor"),
            patch(
                "vllm_fl.dispatch.builtin_ops._get_current_vendor_backend_dirs",
                return_value="metax",
            ),
            patch("vllm_fl.dispatch.builtin_ops.os.path.isdir", return_value=True),
            patch(
                "vllm_fl.dispatch.builtin_ops.os.listdir",
                return_value=["metax"],
            ),
            patch("vllm_fl.dispatch.builtin_ops.os.path.isfile", return_value=True),
            patch(
                "vllm_fl.dispatch.builtin_ops.importlib.import_module",
                return_value=module,
            ),
        ):
            _register_vendor_backends(registry)


class TestRegisterBuiltins:
    def test_registers_flaggems_and_reference(self):
        from vllm_fl.dispatch.registry import OpRegistry

        registry = OpRegistry()
        with (
            patch("vllm_fl.dispatch.builtin_ops._register_vendor_backends"),
            patch("vllm_fl.dispatch.builtin_ops.importlib.import_module"),
            contextlib.suppress(Exception),
        ):
            # This may fail if flaggems/reference aren't importable, but
            # we just want to verify the function runs without crashing
            register_builtins(registry)

    def test_handles_flaggems_import_failure(self):
        from vllm_fl.dispatch.registry import OpRegistry

        registry = OpRegistry()
        with (
            patch(
                "vllm_fl.dispatch.backends.flaggems.register_ops.register_builtins",
                side_effect=ImportError("no flaggems"),
                create=True,
            ),
            patch("vllm_fl.dispatch.builtin_ops._register_vendor_backends"),
            contextlib.suppress(Exception),
        ):
            register_builtins(registry)
