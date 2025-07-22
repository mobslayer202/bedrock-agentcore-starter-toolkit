"""Tests for the centralized logging configuration module."""

import logging
from unittest.mock import Mock, patch

import pytest

from bedrock_agentcore_starter_toolkit.utils.logging_config import (
    _setup_cli_logging,
    _setup_sdk_logging,
    is_logging_configured,
    reset_logging_config,
    setup_toolkit_logging,
)


class TestSetupToolkitLogging:
    """Test the main setup_toolkit_logging function."""

    def setup_method(self):
        """Reset logging state before each test."""
        reset_logging_config()
        # Clear any existing handlers
        toolkit_logger = logging.getLogger("bedrock_agentcore_starter_toolkit")
        for handler in toolkit_logger.handlers[:]:
            toolkit_logger.removeHandler(handler)
        # Reset root logger handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_setup_cli_mode(self):
        """Test explicit CLI mode setup."""
        with patch("bedrock_agentcore_starter_toolkit.utils.logging_config._setup_cli_logging") as mock_cli:
            setup_toolkit_logging(mode="cli")
            mock_cli.assert_called_once()
            assert is_logging_configured()

    def test_setup_sdk_mode(self):
        """Test explicit SDK mode setup."""
        with patch("bedrock_agentcore_starter_toolkit.utils.logging_config._setup_sdk_logging") as mock_sdk:
            setup_toolkit_logging(mode="sdk")
            mock_sdk.assert_called_once()
            assert is_logging_configured()

    def test_duplicate_setup_prevention(self):
        """Test that duplicate setup calls are ignored."""
        with patch("bedrock_agentcore_starter_toolkit.utils.logging_config._setup_sdk_logging") as mock_sdk:
            setup_toolkit_logging(mode="sdk")
            setup_toolkit_logging(mode="sdk")  # Second call should be ignored
            mock_sdk.assert_called_once()  # Should only be called once

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid logging mode: invalid"):
            setup_toolkit_logging(mode="invalid")

    def test_default_mode_is_sdk(self):
        """Test that default mode is sdk."""
        with patch("bedrock_agentcore_starter_toolkit.utils.logging_config._setup_sdk_logging") as mock_sdk:
            setup_toolkit_logging()  # No mode specified
            mock_sdk.assert_called_once()


class TestCliLoggingSetup:
    """Test CLI logging setup functionality."""

    def setup_method(self):
        """Reset logging state before each test."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    @patch("rich.logging.RichHandler")
    @patch("bedrock_agentcore_starter_toolkit.cli.common.console")
    def test_cli_logging_setup_with_rich(self, mock_console, mock_rich_handler):
        """Test CLI logging setup with RichHandler."""
        mock_handler = Mock()
        mock_rich_handler.return_value = mock_handler

        _setup_cli_logging()

        mock_rich_handler.assert_called_once_with(
            show_time=False, show_path=False, show_level=False, console=mock_console
        )

    @patch("rich.logging.RichHandler", side_effect=ImportError)
    @patch("bedrock_agentcore_starter_toolkit.utils.logging_config._setup_basic_logging")
    def test_cli_logging_fallback_without_rich(self, mock_basic_logging, mock_rich_handler):
        """Test CLI logging fallback when RichHandler is not available."""
        _setup_cli_logging()
        mock_basic_logging.assert_called_once()


class TestSdkLoggingSetup:
    """Test SDK logging setup functionality."""

    def setup_method(self):
        """Reset logging state before each test."""
        toolkit_logger = logging.getLogger("bedrock_agentcore_starter_toolkit")
        for handler in toolkit_logger.handlers[:]:
            toolkit_logger.removeHandler(handler)

    def test_sdk_logging_setup(self):
        """Test SDK logging setup with StreamHandler."""
        _setup_sdk_logging()

        toolkit_logger = logging.getLogger("bedrock_agentcore_starter_toolkit")
        assert len(toolkit_logger.handlers) == 1
        assert isinstance(toolkit_logger.handlers[0], logging.StreamHandler)
        assert toolkit_logger.level == logging.INFO

    def test_sdk_logging_no_duplicate_handlers(self):
        """Test that SDK logging doesn't add duplicate handlers."""
        _setup_sdk_logging()
        _setup_sdk_logging()  # Call again

        toolkit_logger = logging.getLogger("bedrock_agentcore_starter_toolkit")
        assert len(toolkit_logger.handlers) == 1


class TestLoggingStateManagement:
    """Test logging state management functions."""

    def setup_method(self):
        """Reset logging state before each test."""
        reset_logging_config()

    def test_is_logging_configured_initial_state(self):
        """Test initial state of logging configuration."""
        assert is_logging_configured() is False

    def test_is_logging_configured_after_setup(self):
        """Test logging configuration state after setup."""
        setup_toolkit_logging(mode="sdk")
        assert is_logging_configured() is True

    def test_reset_logging_config(self):
        """Test resetting logging configuration state."""
        setup_toolkit_logging(mode="sdk")
        assert is_logging_configured() is True

        reset_logging_config()
        assert is_logging_configured() is False


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""

    def setup_method(self):
        """Reset logging state before each test."""
        reset_logging_config()
        # Clear any existing handlers
        toolkit_logger = logging.getLogger("bedrock_agentcore_starter_toolkit")
        for handler in toolkit_logger.handlers[:]:
            toolkit_logger.removeHandler(handler)

    def test_cli_then_sdk_no_duplication(self):
        """Test that CLI setup prevents SDK setup from adding duplicate handlers."""
        # First setup CLI
        with patch("bedrock_agentcore_starter_toolkit.utils.logging_config._setup_cli_logging"):
            setup_toolkit_logging(mode="cli")

        # Then try SDK setup - should be ignored due to _LOGGING_CONFIGURED flag
        with patch("bedrock_agentcore_starter_toolkit.utils.logging_config._setup_sdk_logging") as mock_sdk:
            setup_toolkit_logging(mode="sdk")
            mock_sdk.assert_not_called()

    def test_sdk_then_cli_no_duplication(self):
        """Test that SDK setup prevents CLI setup from adding duplicate handlers."""
        # First setup SDK
        with patch("bedrock_agentcore_starter_toolkit.utils.logging_config._setup_sdk_logging"):
            setup_toolkit_logging(mode="sdk")

        # Then try CLI setup - should be ignored due to _LOGGING_CONFIGURED flag
        with patch("bedrock_agentcore_starter_toolkit.utils.logging_config._setup_cli_logging") as mock_cli:
            setup_toolkit_logging(mode="cli")
            mock_cli.assert_not_called()

    def test_multiple_sdk_setups(self):
        """Test multiple SDK setups don't cause issues."""
        with patch("bedrock_agentcore_starter_toolkit.utils.logging_config._setup_sdk_logging") as mock_sdk:
            setup_toolkit_logging()  # First SDK setup (default)
            setup_toolkit_logging()  # Second SDK setup
            setup_toolkit_logging()  # Third SDK setup

            mock_sdk.assert_called_once()  # Should only be called once

    def test_actual_logging_output_sdk(self, caplog):
        """Test that actual logging output works correctly in SDK mode."""
        reset_logging_config()
        setup_toolkit_logging(mode="sdk")

        # Get the toolkit logger and test actual logging
        logger = logging.getLogger("bedrock_agentcore_starter_toolkit.test")

        # Use caplog to capture log records
        with caplog.at_level(logging.INFO, logger="bedrock_agentcore_starter_toolkit"):
            logger.info("Test message")

        # Check that the message was logged
        assert "Test message" in caplog.text

    def test_logger_hierarchy(self):
        """Test that child loggers inherit the configuration."""
        setup_toolkit_logging(mode="sdk")

        # Test that child loggers work
        child_logger = logging.getLogger("bedrock_agentcore_starter_toolkit.operations.test")
        parent_logger = logging.getLogger("bedrock_agentcore_starter_toolkit")

        # Child logger should inherit from parent
        assert child_logger.parent == parent_logger
        assert len(parent_logger.handlers) == 1
