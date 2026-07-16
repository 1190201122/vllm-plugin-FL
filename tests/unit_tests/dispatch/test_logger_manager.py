# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for vllm_fl.dispatch.logger_manager module.
"""

import logging

from vllm_fl.dispatch.logger_manager import _loggers, get_logger, set_log_level


class TestGetLogger:
    def setup_method(self):
        _loggers.clear()

    def test_returns_logger_instance(self):
        logger = get_logger("test.logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.logger"

    def test_default_name(self):
        logger = get_logger()
        assert logger.name == "vllm_fl.dispatch"

    def test_caches_logger(self):
        logger1 = get_logger("test.cache")
        logger2 = get_logger("test.cache")
        assert logger1 is logger2

    def test_has_handler(self):
        logger = get_logger("test.handler")
        assert len(logger.handlers) >= 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_propagate_disabled(self):
        logger = get_logger("test.propagate")
        assert logger.propagate is False

    def test_different_names_different_loggers(self):
        logger1 = get_logger("test.a")
        logger2 = get_logger("test.b")
        assert logger1 is not logger2


class TestSetLogLevel:
    def setup_method(self):
        _loggers.clear()

    def test_set_level_specific_logger(self):
        logger = get_logger("test.level")
        set_log_level("DEBUG", name="test.level")
        assert logger.level == logging.DEBUG

    def test_set_level_all_loggers(self):
        logger1 = get_logger("test.all1")
        logger2 = get_logger("test.all2")
        set_log_level("WARNING")
        assert logger1.level == logging.WARNING
        assert logger2.level == logging.WARNING

    def test_set_level_nonexistent_name(self):
        set_log_level("DEBUG", name="nonexistent")

    def test_invalid_level_defaults_to_info(self):
        logger = get_logger("test.invalid")
        set_log_level("INVALID_LEVEL", name="test.invalid")
        assert logger.level == logging.INFO
