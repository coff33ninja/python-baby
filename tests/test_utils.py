import pytest
import yaml
import os
import logging
from utils import (
    load_config,
    get_config_value,
    CONFIG_FILE_PATH,
)  # Assuming utils.py is in PYTHONPATH


# Fixture to create a temporary config file for tests
@pytest.fixture
def mock_config_file(tmp_path):
    original_config_path = CONFIG_FILE_PATH
    original_cache = None  # To store utils._config_cache if needed, though we reset it

    def _create_mock(content):
        # Store original cache state and clear it
        nonlocal original_cache
        # This import is to access the global _config_cache in utils
        import utils

        original_cache = utils._config_cache
        utils._config_cache = None

        # Create dummy config
        dummy_config_path = tmp_path / "test_config.yaml"
        with open(dummy_config_path, "w") as f:
            assert os.path.exists(os.path.dirname(dummy_config_path)), "Temp directory for mock config does not exist"
            f.write(content)

        # Override the global CONFIG_FILE_PATH in utils module
        utils.CONFIG_FILE_PATH = str(dummy_config_path)
        return str(dummy_config_path)

    yield _create_mock

    # Teardown: Restore original path and cache
    import utils  # Re-import to ensure we're modifying the same module instance's global

    utils.CONFIG_FILE_PATH = original_config_path
    utils._config_cache = original_cache


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
    utils._config_cache = None  # Reset cache

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
    utils._config_cache = None  # Reset cache


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
    assert utils._config_cache is not None  # Cache should be populated

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
    assert config3["value"] == "updated"

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


    def test_load_config_with_large_file(mock_config_file):
        yaml_content = "\n".join([f"key{i}: value{i}" for i in range(1000)])
        mock_config_file(yaml_content)

        config = load_config()
        assert config is not None
        for i in range(1000):
            assert config[f"key{i}"] == f"value{i}"


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

        with caplog.at_level(logging.WARNING):
            config = load_config()
            assert config is not None
            assert config["key1"] == "value2"  # Last occurrence should override
            assert "Duplicate key" in caplog.text or "overwritten" in caplog.text

# Ensure utils.CONFIG_FILE_PATH is reset if a test modifies it and fails
@pytest.fixture(autouse=True)
def reset_utils_config_path_after_test():
    import utils

    original_path = utils.CONFIG_FILE_PATH
    original_cache = utils._config_cache
    yield
    utils.CONFIG_FILE_PATH = original_path
    utils._config_cache = original_cache


# Note: To properly test logging with caplog when setup_logging is involved,
# it's often better to have setup_logging return the logger or make it accessible,
# or configure specific loggers for testing rather than relying on root logger state
# that might be shared across tests or affected by other modules importing utils.py.
# The current caplog tests are basic and might be fragile if logging setup changes significantly.
