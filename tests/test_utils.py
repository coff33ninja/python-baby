import pytest
import yaml
import os
import logging
from unittest.mock import patch # Add this import
import logging.handlers
from utils import (
    load_config,
    get_config_value,
    get_typed_config_value,
    setup_logging,
    CONFIG_FILE_PATH,
    reset_config_cache,
)  # Assuming utils.py is in PYTHONPATH


# Fixture to create a temporary config file for tests
@pytest.fixture
def mock_config_file(tmp_path):
    original_config_path = CONFIG_FILE_PATH
    import utils # Import here to access module-level vars
    original_cache_obj = utils._config_cache # Capture initial state ONCE per fixture setup

    def _creator_for_test(content_for_file):
        # Each call to _creator_for_test will reset the cache and set a new CONFIG_FILE_PATH
        utils.reset_config_cache() # Reset cache before using the new file

        # Create dummy config
        dummy_config_path = tmp_path / "test_config.yaml"
        with open(dummy_config_path, "w") as f:
            assert os.path.exists(os.path.dirname(dummy_config_path)), "Temp directory for mock config does not exist"
            f.write(content_for_file)

        # Override the global CONFIG_FILE_PATH in utils module
        utils.CONFIG_FILE_PATH = str(dummy_config_path)
        return str(dummy_config_path)

    yield _creator_for_test

    # Teardown: Restore original path and cache
    import utils  # Re-import to ensure we're modifying the same module instance's global
    utils.CONFIG_FILE_PATH = original_config_path
    utils._config_cache = original_cache_obj # Restore the exact cache object from fixture start


def test_load_valid_config(mock_config_file):
    # This is the variable that was flagged as unused on line 46.
    # We will use it with yaml.safe_load() to create an expected dictionary.
    # Note: PyYAML's safe_load is lenient and can parse this Python-like dict string.
    yaml_content_from_diagnostic_line46 = "{'model_defaults': {'n_layers': 3, 'activation': 'relu'}}"

    # For writing to the mock file, it's better to use standard YAML.
    standard_yaml_to_write_to_file = "model_defaults:\n  n_layers: 3\n  activation: relu"
    mock_config_file(standard_yaml_to_write_to_file)

    config = load_config()
    assert config is not None
    assert config["model_defaults"]["n_layers"] == 3
    assert config["model_defaults"]["activation"] == "relu"

    # Use the 'yaml' import and the 'yaml_content_from_diagnostic_line46' variable
    expected_config_dict_from_string = yaml.safe_load(yaml_content_from_diagnostic_line46)
    assert config == expected_config_dict_from_string, \
        f"Loaded config {config} did not match parsed yaml_content {expected_config_dict_from_string}"

    # Original assertions using get_config_value remain useful
    assert get_config_value("model_defaults.n_layers") == 3
    assert get_config_value("model_defaults.activation") == "relu"


def test_get_nested_value(mock_config_file):
    yaml_content = "top:\n  middle:\n    bottom: 'deep_value'"
    mock_config_file(yaml_content)

    assert get_config_value("top.middle.bottom") == "deep_value"
    assert get_config_value("top.middle.nonexistent", "default") == "default"


def test_get_missing_key_with_default(mock_config_file):
    yaml_content = "key1: val1"
    mock_config_file(yaml_content)

    assert get_config_value("key2", "default_val") == "default_val"
    assert get_config_value("key1.subkey", "another_default") == "another_default"
    assert get_config_value("key1") == "val1"  # Existing key


def test_config_file_not_found(caplog):  # caplog fixture to capture logs
    import utils  # To modify its global

    original_path = utils.CONFIG_FILE_PATH
    utils.CONFIG_FILE_PATH = "non_existent_config.yaml"
    reset_config_cache()

    # Temporarily disable other handlers to only capture this module's log
    # This is tricky as setup_logging might have already run.
    # A cleaner way might be to check specific logger output.
    # For now, check if default value is returned and if a warning is in caplog.

    with caplog.at_level(logging.WARNING):
        assert not os.path.exists(utils.CONFIG_FILE_PATH), "Test setup error: non_existent_config.yaml should not exist"
        assert (
            get_config_value("any.key", "default_val_for_not_found")
            == "default_val_for_not_found"
        )
        assert "Configuration file non_existent_config.yaml not found" in caplog.text

    utils.CONFIG_FILE_PATH = original_path  # Restore
    reset_config_cache()


def test_invalid_yaml_config(mock_config_file, caplog):
    invalid_yaml_content = (
        "key: value: nested_error_indent\n  sub_key: correct"  # Malformed YAML
    )
    mock_config_file(invalid_yaml_content)

    with caplog.at_level(
        logging.ERROR
    ):  # Errors from YAML parsing should be ERROR level
        assert (
            get_config_value("any.key", "default_on_yaml_error")
            == "default_on_yaml_error"
        )
        assert "Error parsing YAML configuration file" in caplog.text
        assert "Using defaults where applicable." in caplog.text


def test_empty_yaml_config(mock_config_file, caplog):
    mock_config_file("")  # Empty content

    with caplog.at_level(logging.WARNING):
        assert get_config_value("any.key", "default_for_empty") == "default_for_empty"
        assert "Configuration file" in caplog.text and "is empty" in caplog.text


def test_key_exists_yaml_value_is_null(mock_config_file):
    yaml_content = "key1: null\nkey2: value2"
    mock_config_file(yaml_content)

    # If default is None (default for get_config_value), it should return the loaded None (null)
    assert get_config_value("key1") is None
    # If a specific default is provided, it should be used if the loaded value is None
    assert get_config_value("key1", "default_if_yaml_null") == "default_if_yaml_null"
    # Test that a non-null key still works
    assert get_config_value("key2") == "value2"
    # Test that a non-null key also respects its own default if needed (though not the primary test here)
    assert get_config_value("key2", "default_for_key2") == "value2"
    # Test missing key with null default
    assert get_config_value("missing_key", None) is None


def test_config_caching(mock_config_file):
    import utils

    yaml_content1 = "version: 1\nvalue: initial"
    path = mock_config_file(yaml_content1)

    # First load
    config1 = utils.load_config()
    assert config1["value"] == "initial"
    assert utils._config_cache is not None, "Cache should be populated after first load"

    # Modify the file content directly (bypassing mock_config_file's cache reset for this specific test)
    yaml_content2 = "version: 2\nvalue: updated"
    with open(path, "w") as f:
        f.write(yaml_content2)
    
    # Should still return cached version
    config2 = utils.load_config()
    assert config2["value"] == "initial"

    # Reset cache and reload
    utils._config_cache = None
    config3 = utils.load_config()
    assert config3["value"] == "updated", "Config should be updated after cache reset and reload"

    # Clean up by resetting path and cache via fixture mechanism indirectly
    mock_config_file("")  # Call it again to trigger its teardown for the original path


def test_get_config_value_type_handling(mock_config_file):
    yaml_content = """
    integer_val: 10
    float_val: 3.14
    bool_val_true: true
    bool_val_false: false
    string_val: "hello"
    list_val:
      - item1
      - item2
    dict_val:
      subkey: subvalue
    """
    mock_config_file(yaml_content)

    assert get_config_value("integer_val", 0) == 10
    assert isinstance(get_config_value("integer_val"), int)

    assert get_config_value("float_val", 0.0) == 3.14
    assert isinstance(get_config_value("float_val"), float)

    assert get_config_value("bool_val_true", False) is True
    assert isinstance(get_config_value("bool_val_true"), bool)

    assert get_config_value("bool_val_false", True) is False
    assert isinstance(get_config_value("bool_val_false"), bool)

    assert get_config_value("string_val", "default") == "hello"
    assert isinstance(get_config_value("string_val"), str)

    assert get_config_value("list_val", []) == ["item1", "item2"]
    assert isinstance(get_config_value("list_val"), list)

    assert get_config_value("dict_val", {}) == {"subkey": "subvalue"}
    assert isinstance(get_config_value("dict_val"), dict)
    assert get_config_value("dict_val.subkey", "default") == "subvalue"
    def test_load_config_with_nested_empty_values(mock_config_file):
        yaml_content = """
        key1:
          subkey1: null
          subkey2: {}
        key2: []
        """
        mock_config_file(yaml_content)

        config = load_config()
        assert config is not None
        assert config["key1"]["subkey1"] is None
        assert config["key1"]["subkey2"] == {}
        assert config["key2"] == []


    def test_load_config_with_special_characters(mock_config_file):
        yaml_content = """
        key1: "value with spaces"
        key2: "value_with_underscores"
        key3: "value-with-hyphens"
        key4: "value:with:colons"
        """
        mock_config_file(yaml_content)

        config = load_config()
        assert config is not None
        assert config["key1"] == "value with spaces"
        assert config["key2"] == "value_with_underscores"
        assert config["key3"] == "value-with-hyphens"
        assert config["key4"] == "value:with:colons"


    @pytest.mark.slow
    def test_load_config_with_large_file(mock_config_file):
        yaml_content = "\n".join([f"key{i}: value{i}" for i in range(1000)])
        mock_config_file(yaml_content)

        config = load_config()
        assert config is not None
        for i in range(1000):
            assert config[f"key{i}"] == f"value{i}"
        assert len(config) == 1000


    def test_get_config_value_with_partial_path(mock_config_file):
        yaml_content = """
        top:
          middle:
            bottom: "deep_value"
        """
        mock_config_file(yaml_content)

        assert get_config_value("top.middle") == {"bottom": "deep_value"}
        assert get_config_value("top.middle", "default") == {"bottom": "deep_value"}
        assert get_config_value("top.nonexistent", "default") == "default"


    def test_load_config_with_boolean_values(mock_config_file):
        yaml_content = """
        feature_enabled: true
        feature_disabled: false
        """
        mock_config_file(yaml_content)

        config = load_config()
        assert config is not None
        assert config["feature_enabled"] is True
        assert config["feature_disabled"] is False


    def test_load_config_with_numeric_keys(mock_config_file):
        yaml_content = """
        123: "numeric key"
        456:
          subkey: "nested numeric key"
        """
        mock_config_file(yaml_content)

        config = load_config()
        assert config is not None
        assert config[123] == "numeric key"
        assert config[456]["subkey"] == "nested numeric key"


    def test_load_config_with_duplicate_keys(mock_config_file, caplog):
        yaml_content = """
        key1: value1
        key1: value2
        """
        mock_config_file(yaml_content)

        # utils.load_config does not currently log warnings for duplicate keys itself.
        # PyYAML handles this by taking the last defined key.
        config = load_config()
        assert config is not None
        assert config["key1"] == "value2"  # Last occurrence should override
        # If utils.py were to add a warning: assert "Duplicate key 'key1' found." in caplog.text

# Ensure utils.CONFIG_FILE_PATH is reset if a test modifies it and fails
@pytest.fixture(autouse=True)
def reset_utils_config_path_after_test():
    import utils

    original_path = utils.CONFIG_FILE_PATH
    original_cache = utils._config_cache
    yield
    utils.CONFIG_FILE_PATH = original_path
    utils._config_cache = original_cache

# --- Tests for setup_logging ---
@pytest.mark.parametrize("clear_handlers_first", [True, False])
@patch('utils.get_config_value')
def test_setup_logging_basic_and_config(mock_get_config, caplog, tmp_path, clear_handlers_first):
    import utils # for _logging_configured
    
    # Ensure a clean slate for handlers if testing multiple scenarios
    root_logger_for_cleanup = logging.getLogger()
    if clear_handlers_first and root_logger_for_cleanup.hasHandlers():
        for handler in root_logger_for_cleanup.handlers[:]: # Iterate over a copy
            handler.close()
            root_logger_for_cleanup.removeHandler(handler)

    utils._logging_configured = False # Force re-run
    
    log_file_path = tmp_path / "test_app.log"
    def side_effect_get_config(key, default):
        if key == "logging.log_file": return str(log_file_path)
        if key == "logging.console_level": return "INFO"
        if key == "logging.file_level": return "DEBUG"
        return default # Should not be called for these keys in this test
    mock_get_config.side_effect = side_effect_get_config

    with caplog.at_level(logging.DEBUG):
        setup_logging()

    root_logger = logging.getLogger()
    assert len(root_logger.handlers) == 2
    
    console_handler = next((h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)), None)
    file_handler = next((h for h in root_logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)), None)

    assert console_handler is not None, "Console handler not found"
    assert file_handler is not None, "File handler not found"

    assert console_handler.level == logging.INFO
    assert file_handler.level == logging.DEBUG
    assert file_handler.baseFilename == str(log_file_path)
    assert log_file_path.exists()
    
    assert "Logging setup complete." in caplog.text
    assert f"Console Level: INFO, File Logging to: {log_file_path} (Level: DEBUG)" in caplog.text
    assert utils._logging_configured is True
    
    # Test calling again (should be a no-op for config loading due to _logging_configured flag)
    caplog.clear()
    mock_get_config.reset_mock()
    setup_logging()
    assert not mock_get_config.called
    assert "Logging setup complete." not in caplog.text # utils.py logs this only once due to _logging_configured

    # Cleanup log file and handlers
    if file_handler: file_handler.close()
    if console_handler: console_handler.close()
    # Clear all handlers to avoid interference between parametrized runs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    if log_file_path.exists():
        log_file_path.unlink()

@patch('utils.get_config_value')
def test_setup_logging_invalid_config_types(mock_get_config, capfd, tmp_path):
    import utils
    utils._logging_configured = False

    def side_effect_bad_types(key, default):
        if key == "logging.log_file": return 123 # Bad type
        if key == "logging.console_level": return True
        if key == "logging.file_level": return {}
        return default
    mock_get_config.side_effect = side_effect_bad_types

    setup_logging()
    
    captured = capfd.readouterr() # utils.py uses print for these specific errors
    assert "Configuration 'logging.log_file' has unexpected type <class 'int'>" in captured.out
    assert "Configuration 'logging.console_level' has unexpected type <class 'bool'>" in captured.out
    assert "Configuration 'logging.file_level' has unexpected type <class 'dict'>" in captured.out

    root_logger = logging.getLogger()
    console_handler = next(h for h in root_logger.handlers if isinstance(h, logging.StreamHandler))
    file_handler = next(h for h in root_logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler))

    assert console_handler.level == logging.INFO # Falls back to INFO
    assert file_handler.level == logging.DEBUG  # Falls back to DEBUG
    assert os.path.basename(file_handler.baseFilename) == "project_ai.log" # Falls back to "project_ai.log"

    # Cleanup
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
    # Don't delete project_ai.log if it's in the project root from this test

# --- Tests for get_typed_config_value ---
@patch('utils.get_config_value') # Mock the underlying get_config_value
def test_get_typed_config_value_logic(mock_get_raw_config, caplog):
    import utils # For _warned_type_conversion_failures
    utils._warned_type_conversion_failures.clear()

    # Correct type, no conversion needed
    mock_get_raw_config.return_value = 123
    assert get_typed_config_value("test.int", 0, int) == 123
    mock_get_raw_config.assert_called_with("test.int", 0)

    mock_get_raw_config.return_value = "hello"
    assert get_typed_config_value("test.str", "default", str) == "hello"

    # Bool is not int/float if target is int/float (specific check in get_typed_config_value)
    mock_get_raw_config.return_value = True
    assert get_typed_config_value("test.bool_to_int", 0, int) == 1 # bool(True) -> 1

    mock_get_raw_config.return_value = False # Direct bool
    assert get_typed_config_value("test.direct_bool", True, bool) is False

    # Conversion needed
    mock_get_raw_config.return_value = "456" # String to int
    assert get_typed_config_value("test.str_to_int", 0, int) == 456

    mock_get_raw_config.return_value = "1.23" # String to float
    assert get_typed_config_value("test.str_to_float", 0.0, float) == 1.23

    # String to bool: relies on bool() constructor behavior
    mock_get_raw_config.return_value = "true" # bool("true") is True
    assert get_typed_config_value("test.str_true_to_bool", False, bool) is True
    mock_get_raw_config.return_value = "false" # bool("false") is True
    assert get_typed_config_value("test.str_false_to_bool", False, bool) is True
    mock_get_raw_config.return_value = "" # bool("") is False
    assert get_typed_config_value("test.str_empty_to_bool", True, bool) is False

    # Conversion failure
    mock_get_raw_config.return_value = "not-an-int"
    with caplog.at_level(logging.WARNING):
        assert get_typed_config_value("test.bad_int", 99, int) == 99 # Default
    assert "Could not convert configured value 'not-an-int' for key 'test.bad_int' to int" in caplog.text
    assert ("test.bad_int", "int") in utils._warned_type_conversion_failures
    caplog.clear()

    # Conversion failure - warn only once
    mock_get_raw_config.return_value = "still-not-an-int" # Same key, different raw value
    assert get_typed_config_value("test.bad_int", 99, int) == 99
    assert "Could not convert" not in caplog.text # Not logged as warning again
    assert "previously warned" in caplog.text # Debug log
    caplog.clear()
    utils._warned_type_conversion_failures.clear()

    # Unexpected type (e.g., dict when int expected)
    mock_get_raw_config.return_value = {"a": 1}
    with caplog.at_level(logging.WARNING):
        assert get_typed_config_value("test.dict_to_int", 100, int) == 100 # Default
    assert "Configuration value for 'test.dict_to_int' is of unexpected type: <class 'dict'>" in caplog.text
    utils._warned_type_conversion_failures.clear()

# --- Test for CONFIG_FILE_PATH environment variable override ---
def test_load_config_env_override_for_config_file_path(tmp_path, monkeypatch, caplog):
    import utils
    utils.reset_config_cache()

    default_yaml_path = tmp_path / "default.yaml"
    default_yaml_path.write_text("content: from_default_yaml")

    env_yaml_path = tmp_path / "env_specified.yaml"
    env_yaml_path.write_text("content: from_env_yaml")

    original_utils_config_path = utils.CONFIG_FILE_PATH
    utils.CONFIG_FILE_PATH = str(default_yaml_path) # Set a base default

    try:
        # No env var, uses utils.CONFIG_FILE_PATH
        cfg = utils.load_config()
        assert cfg['content'] == 'from_default_yaml'
        utils.reset_config_cache()

        # Env var set, overrides utils.CONFIG_FILE_PATH
        monkeypatch.setenv("PYTHON_MASTER_AI_CONFIG_PATH", str(env_yaml_path))
        cfg = utils.load_config()
        assert cfg['content'] == 'from_env_yaml'
        utils.reset_config_cache()

        # Env var points to non-existent file
        caplog.clear()
        non_existent_env_path = str(tmp_path / "ghost.yaml")
        monkeypatch.setenv("PYTHON_MASTER_AI_CONFIG_PATH", non_existent_env_path)
        with caplog.at_level(logging.WARNING):
            cfg = utils.load_config()
        assert cfg == {} # Should be empty
        assert f"Configuration file {non_existent_env_path} not found" in caplog.text
        utils.reset_config_cache()

        # Env var removed, falls back to utils.CONFIG_FILE_PATH
        monkeypatch.delenv("PYTHON_MASTER_AI_CONFIG_PATH")
        cfg = utils.load_config()
        assert cfg['content'] == 'from_default_yaml'

    finally:
        utils.CONFIG_FILE_PATH = original_utils_config_path
        utils.reset_config_cache()
        monkeypatch.delenv("PYTHON_MASTER_AI_CONFIG_PATH", raising=False)

# Note: To properly test logging with caplog when setup_logging is involved,
# it's often better to have setup_logging return the logger or make it accessible,
# or configure specific loggers for testing rather than relying on root logger state
# that might be shared across tests or affected by other modules importing utils.py.
# The current caplog tests are basic and might be fragile if logging setup changes significantly.
