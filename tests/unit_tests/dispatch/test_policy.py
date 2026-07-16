# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for dispatch policy module.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from vllm_fl.dispatch.policy import (
    PREFER_DEFAULT,
    PREFER_REFERENCE,
    PREFER_VENDOR,
    VALID_PREFER_VALUES,
    PolicyManager,
    SelectionPolicy,
    get_policy,
    policy_context,
    reset_global_policy,
    set_global_policy,
    with_preference,
    with_strict_mode,
)


class TestSelectionPolicy:
    def test_default_values(self):
        policy = SelectionPolicy()
        assert policy.prefer == PREFER_DEFAULT
        assert policy.strict is False
        assert policy.per_op_order == ()
        assert policy.deny_vendors == frozenset()
        assert policy.allow_vendors is None

    def test_invalid_prefer_value_raises(self):
        with pytest.raises(ValueError, match="Invalid prefer value"):
            SelectionPolicy(prefer="invalid")

    def test_valid_prefer_values(self):
        for prefer in VALID_PREFER_VALUES:
            policy = SelectionPolicy(prefer=prefer)
            assert policy.prefer == prefer

    def test_from_dict(self):
        policy = SelectionPolicy.from_dict(
            prefer="vendor",
            strict=True,
            per_op_order={"silu": ["vendor", "flagos"]},
            deny_vendors={"AMD"},
            allow_vendors={"CUDA"},
        )
        assert policy.prefer == "vendor"
        assert policy.strict is True
        assert policy.deny_vendors == frozenset({"AMD"})
        assert policy.allow_vendors == frozenset({"CUDA"})

    def test_get_default_order_flagos(self):
        policy = SelectionPolicy(prefer=PREFER_DEFAULT)
        order = policy.get_default_order()
        assert order == ["flagos", "vendor", "reference"]

    def test_get_default_order_vendor(self):
        policy = SelectionPolicy(prefer=PREFER_VENDOR)
        order = policy.get_default_order()
        assert order == ["vendor", "flagos", "reference"]

    def test_get_default_order_reference(self):
        policy = SelectionPolicy(prefer=PREFER_REFERENCE)
        order = policy.get_default_order()
        assert order == ["reference", "flagos", "vendor"]

    def test_is_vendor_allowed_deny_list(self):
        policy = SelectionPolicy(deny_vendors=frozenset({"AMD"}))
        assert policy.is_vendor_allowed("CUDA") is True
        assert policy.is_vendor_allowed("AMD") is False

    def test_is_vendor_allowed_allow_list(self):
        policy = SelectionPolicy(allow_vendors=frozenset({"CUDA"}))
        assert policy.is_vendor_allowed("CUDA") is True
        assert policy.is_vendor_allowed("AMD") is False

    def test_is_vendor_allowed_combined(self):
        policy = SelectionPolicy(
            allow_vendors=frozenset({"CUDA", "AMD"}),
            deny_vendors=frozenset({"AMD"}),
        )
        assert policy.is_vendor_allowed("CUDA") is True
        assert policy.is_vendor_allowed("AMD") is False

    def test_get_per_op_order(self):
        policy = SelectionPolicy.from_dict(
            per_op_order={"silu": ["vendor", "flagos"], "rms_norm": ["reference"]},
        )
        assert policy.get_per_op_order("silu") == ["vendor", "flagos"]
        assert policy.get_per_op_order("rms_norm") == ["reference"]
        assert policy.get_per_op_order("nonexistent") is None

    def test_fingerprint_uniqueness(self):
        policy1 = SelectionPolicy(prefer=PREFER_DEFAULT)
        policy2 = SelectionPolicy(prefer=PREFER_VENDOR)
        policy3 = SelectionPolicy(prefer=PREFER_DEFAULT, strict=True)

        assert policy1.fingerprint() != policy2.fingerprint()
        assert policy1.fingerprint() != policy3.fingerprint()
        assert policy2.fingerprint() != policy3.fingerprint()

    def test_frozen_dataclass(self):
        policy = SelectionPolicy()
        with pytest.raises(AttributeError):
            policy.prefer = "vendor"


class TestPolicyManager:
    @pytest.fixture(autouse=True)
    def reset_policy(self):
        """Reset global policy before and after each test."""
        with patch("vllm_fl.dispatch.config.get_config_path", return_value=None):
            reset_global_policy()
            yield
            reset_global_policy()

    def test_get_instance_singleton(self):
        manager1 = PolicyManager.get_instance()
        manager2 = PolicyManager.get_instance()
        assert manager1 is manager2

    def test_get_policy_returns_default(self):
        policy = get_policy()
        assert isinstance(policy, SelectionPolicy)
        assert policy.prefer == PREFER_DEFAULT

    def test_set_global_policy(self):
        new_policy = SelectionPolicy(prefer=PREFER_VENDOR)
        old_policy = set_global_policy(new_policy)

        current = get_policy()
        assert current.prefer == PREFER_VENDOR
        assert old_policy.prefer == PREFER_DEFAULT

    def test_reset_global_policy(self):
        set_global_policy(SelectionPolicy(prefer=PREFER_VENDOR))
        reset_global_policy()

        policy = get_policy()
        assert policy.prefer == PREFER_DEFAULT

    def test_policy_epoch_bumps(self):
        manager = PolicyManager.get_instance()
        epoch1 = manager.get_policy_epoch()

        set_global_policy(SelectionPolicy(prefer=PREFER_VENDOR))
        epoch2 = manager.get_policy_epoch()

        assert epoch2 > epoch1


class TestPolicyContext:
    @pytest.fixture(autouse=True)
    def reset_policy(self):
        with patch("vllm_fl.dispatch.config.get_config_path", return_value=None):
            reset_global_policy()
            yield
            reset_global_policy()

    def test_policy_context_override(self):
        original = get_policy()
        assert original.prefer == PREFER_DEFAULT

        override_policy = SelectionPolicy(prefer=PREFER_VENDOR)
        with policy_context(override_policy):
            inside = get_policy()
            assert inside.prefer == PREFER_VENDOR

        after = get_policy()
        assert after.prefer == PREFER_DEFAULT

    def test_with_preference(self):
        with with_preference("vendor"):
            policy = get_policy()
            assert policy.prefer == "vendor"

        policy = get_policy()
        assert policy.prefer == PREFER_DEFAULT

    def test_with_strict_mode(self):
        with with_strict_mode():
            policy = get_policy()
            assert policy.strict is True

        policy = get_policy()
        assert policy.strict is False

    def test_nested_contexts(self):
        with with_preference("vendor"):
            assert get_policy().prefer == "vendor"
            with with_strict_mode():
                policy = get_policy()
                assert policy.strict is True
            assert get_policy().prefer == "vendor"

        assert get_policy().prefer == PREFER_DEFAULT


class TestPolicyFromEnv:
    @pytest.fixture(autouse=True)
    def reset_policy(self):
        reset_global_policy()
        yield
        reset_global_policy()

    def test_policy_from_env_prefer(self):
        with patch.dict(os.environ, {"VLLM_FL_PREFER": "vendor"}):
            reset_global_policy()
            policy = get_policy()
            assert policy.prefer == "vendor"

    def test_policy_from_env_strict(self):
        with patch.dict(os.environ, {"VLLM_FL_STRICT": "1"}):
            reset_global_policy()
            policy = get_policy()
            assert policy.strict is True

    def test_policy_from_env_deny_vendors(self):
        with patch.dict(os.environ, {"VLLM_FL_DENY_VENDORS": "AMD,Intel"}):
            reset_global_policy()
            policy = get_policy()
            assert "AMD" in policy.deny_vendors
            assert "Intel" in policy.deny_vendors


class TestPolicyFromConfig:
    def test_policy_from_config_file(self):
        config_content = """
prefer: vendor
strict: true
allow_vendors:
  - CUDA
deny_vendors:
  - AMD
op_backends:
  silu:
    - vendor
    - flagos
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            f.flush()
            config_path = f.name

        try:
            from vllm_fl.dispatch.policy import policy_from_config

            policy = policy_from_config(config_path)

            assert policy.prefer == "vendor"
            assert policy.strict is True
            assert "CUDA" in policy.allow_vendors
            assert "AMD" in policy.deny_vendors
            assert policy.get_per_op_order("silu") == ["vendor", "flagos"]
        finally:
            os.unlink(config_path)

    def test_policy_from_nonexistent_config(self):
        from vllm_fl.dispatch.policy import policy_from_config

        with pytest.raises(FileNotFoundError):
            policy_from_config("/nonexistent/path/config.yaml")

    def test_policy_from_config_invalid_yaml(self):
        from vllm_fl.dispatch.policy import policy_from_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(": invalid: yaml: [")
            f.flush()
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="Failed to load"):
                policy_from_config(config_path)
        finally:
            os.unlink(config_path)

    def test_policy_from_config_string_vendors(self):
        from vllm_fl.dispatch.policy import policy_from_config

        config_content = """
prefer: flagos
deny_vendors: "AMD,Intel"
allow_vendors: "CUDA,ROCm"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            f.flush()
            config_path = f.name

        try:
            policy = policy_from_config(config_path)
            assert "AMD" in policy.deny_vendors
            assert "Intel" in policy.deny_vendors
            assert "CUDA" in policy.allow_vendors
            assert "ROCm" in policy.allow_vendors
        finally:
            os.unlink(config_path)

    def test_policy_from_config_string_op_backends(self):
        from vllm_fl.dispatch.policy import policy_from_config

        config_content = """
prefer: flagos
op_backends:
  rms_norm: "vendor|flagos|reference"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            f.flush()
            config_path = f.name

        try:
            policy = policy_from_config(config_path)
            assert policy.get_per_op_order("rms_norm") == [
                "vendor",
                "flagos",
                "reference",
            ]
        finally:
            os.unlink(config_path)

    def test_policy_from_config_invalid_prefer_defaults(self):
        from vllm_fl.dispatch.policy import policy_from_config

        config_content = """
prefer: invalid_value
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            f.flush()
            config_path = f.name

        try:
            policy = policy_from_config(config_path)
            assert policy.prefer == PREFER_DEFAULT
        finally:
            os.unlink(config_path)


class TestParseCsvSet:
    def test_empty_string(self):
        result = PolicyManager._parse_csv_set("")
        assert result == set()

    def test_single_value(self):
        result = PolicyManager._parse_csv_set("CUDA")
        assert result == {"CUDA"}

    def test_multiple_values(self):
        result = PolicyManager._parse_csv_set("CUDA,AMD,Intel")
        assert result == {"CUDA", "AMD", "Intel"}

    def test_strips_whitespace(self):
        result = PolicyManager._parse_csv_set(" CUDA , AMD , Intel ")
        assert result == {"CUDA", "AMD", "Intel"}

    def test_skips_empty_entries(self):
        result = PolicyManager._parse_csv_set("CUDA,,AMD,")
        assert result == {"CUDA", "AMD"}


class TestParsePerOp:
    def test_empty_string(self):
        result = PolicyManager._parse_per_op("")
        assert result == {}

    def test_single_op(self):
        result = PolicyManager._parse_per_op("silu=vendor|flagos")
        assert result == {"silu": ["vendor", "flagos"]}

    def test_multiple_ops(self):
        result = PolicyManager._parse_per_op("silu=vendor|flagos;rms_norm=reference")
        assert result == {"silu": ["vendor", "flagos"], "rms_norm": ["reference"]}

    def test_strips_whitespace(self):
        result = PolicyManager._parse_per_op(" silu = vendor | flagos ")
        assert result == {"silu": ["vendor", "flagos"]}

    def test_skips_invalid_parts(self):
        result = PolicyManager._parse_per_op("silu=vendor;invalid_no_equals")
        assert result == {"silu": ["vendor"]}

    def test_skips_empty_parts(self):
        result = PolicyManager._parse_per_op("silu=vendor;;rms_norm=flagos")
        assert result == {"silu": ["vendor"], "rms_norm": ["flagos"]}


class TestParseOpConfig:
    def test_valid_config(self):
        result = PolicyManager._parse_op_config(
            {"silu": "vendor", "rms_norm": "flagos"}
        )
        assert result == {"silu": ["vendor"], "rms_norm": ["flagos"]}

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unsupported backend"):
            PolicyManager._parse_op_config({"silu": "invalid_backend"})

    def test_case_insensitive(self):
        result = PolicyManager._parse_op_config({"silu": "VENDOR"})
        assert result == {"silu": ["vendor"]}


class TestPolicyFromEnvExtended:
    @pytest.fixture(autouse=True)
    def reset_policy(self):
        reset_global_policy()
        yield
        reset_global_policy()

    def test_allow_vendors_from_env(self):
        with (
            patch.dict(os.environ, {"VLLM_FL_ALLOW_VENDORS": "CUDA,ROCm"}),
            patch("vllm_fl.dispatch.config.get_config_path", return_value=None),
        ):
            reset_global_policy()
            policy = get_policy()
            assert policy.allow_vendors == frozenset({"CUDA", "ROCm"})

    def test_per_op_from_env(self):
        with (
            patch.dict(
                os.environ, {"VLLM_FL_PER_OP": "silu=vendor|flagos;rms=reference"}
            ),
            patch("vllm_fl.dispatch.config.get_config_path", return_value=None),
            patch("vllm_fl.utils.get_op_config", return_value=None),
        ):
            reset_global_policy()
            policy = get_policy()
            assert policy.get_per_op_order("silu") == ["vendor", "flagos"]
            assert policy.get_per_op_order("rms") == ["reference"]

    def test_invalid_strict_value_defaults_false(self):
        with (
            patch.dict(os.environ, {"VLLM_FL_STRICT": "invalid"}),
            patch("vllm_fl.dispatch.config.get_config_path", return_value=None),
        ):
            reset_global_policy()
            policy = get_policy()
            assert policy.strict is False

    def test_config_file_override(self):
        config_content = "prefer: vendor\nstrict: true\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            f.flush()
            config_path = f.name

        try:
            with patch.dict(os.environ, {"VLLM_FL_CONFIG": config_path}):
                reset_global_policy()
                policy = get_policy()
                assert policy.prefer == "vendor"
                assert policy.strict is True
        finally:
            os.unlink(config_path)


class TestConvenienceContextManagers:
    @pytest.fixture(autouse=True)
    def reset_policy(self):
        with patch("vllm_fl.dispatch.config.get_config_path", return_value=None):
            reset_global_policy()
            yield
            reset_global_policy()

    def test_with_allowed_vendors(self):
        from vllm_fl.dispatch.policy import with_allowed_vendors

        with with_allowed_vendors("CUDA", "ROCm"):
            policy = get_policy()
            assert policy.allow_vendors == frozenset({"CUDA", "ROCm"})
            assert policy.is_vendor_allowed("CUDA") is True
            assert policy.is_vendor_allowed("AMD") is False

        policy = get_policy()
        assert policy.allow_vendors is None

    def test_with_denied_vendors(self):
        from vllm_fl.dispatch.policy import with_denied_vendors

        with with_denied_vendors("AMD", "Intel"):
            policy = get_policy()
            assert "AMD" in policy.deny_vendors
            assert "Intel" in policy.deny_vendors

        policy = get_policy()
        assert "AMD" not in policy.deny_vendors


class TestSelectionPolicyExtended:
    def test_fingerprint_with_allow_vendors(self):
        policy = SelectionPolicy(allow_vendors=frozenset({"CUDA", "ROCm"}))
        fp = policy.fingerprint()
        assert "allow=" in fp
        assert "CUDA" in fp

    def test_fingerprint_with_deny_vendors(self):
        policy = SelectionPolicy(deny_vendors=frozenset({"AMD"}))
        fp = policy.fingerprint()
        assert "deny=AMD" in fp

    def test_fingerprint_with_per_op(self):
        policy = SelectionPolicy.from_dict(per_op_order={"silu": ["vendor"]})
        fp = policy.fingerprint()
        assert "per=" in fp
        assert "silu" in fp

    def test_per_op_order_dict(self):
        policy = SelectionPolicy.from_dict(
            per_op_order={"silu": ["vendor", "flagos"], "rms": ["reference"]}
        )
        d = policy.per_op_order_dict
        assert d == {"rms": ["reference"], "silu": ["vendor", "flagos"]}

    def test_hash_equality(self):
        p1 = SelectionPolicy(prefer="vendor")
        p2 = SelectionPolicy(prefer="vendor")
        assert hash(p1) == hash(p2)

    def test_hash_inequality(self):
        p1 = SelectionPolicy(prefer="vendor")
        p2 = SelectionPolicy(prefer="flagos")
        assert hash(p1) != hash(p2)
