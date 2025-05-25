import yaml
import os
import sys
import logging  # Added for logging
import logging.handlers  # Added for logging

CONFIG_FILE_PATH = "config.yaml"
_config_cache: dict | None = None
# Initialize logger at module level for use in load_config
logger = logging.getLogger(__name__)



def load_config():
    global _config_cache
    if _config_cache is None:
        # Allow CONFIG_FILE_PATH to be overridden by an environment variable
        actual_config_path = os.getenv('PYTHON_MASTER_AI_CONFIG_PATH', CONFIG_FILE_PATH)
        logger.info(f"Attempting to load configuration from: {actual_config_path}")

        if os.path.exists(actual_config_path):
            try:
                with open(actual_config_path, "r") as f:
                    _config_cache = yaml.safe_load(f)
                    if _config_cache is None:
                        _config_cache = {} # Ensure it's a dict even if empty
                        logger.warning( # Changed from print
                            f"Configuration file {actual_config_path} is empty. Using defaults where applicable."
                        )
                    else:
                        logger.info(f"Successfully loaded configuration from {actual_config_path}.")
                        # Optionally, validate the loaded configuration
                        # validate_config_schema(_config_cache)
            except yaml.YAMLError as e:
                logger.error( # Changed from print
                    f"Error parsing YAML configuration file {actual_config_path}: {e}. Using defaults where applicable."
                )
                _config_cache = {}
        else:
            logger.warning( # Changed from print
                f"Configuration file {actual_config_path} not found. Using defaults where applicable."
            )
            _config_cache = {}
    return _config_cache

def get_config_value(key_path: str, default=None):
    # 1. Check environment variables first
    # Convert dot.separated.path to UPPER_SNAKE_CASE for env var lookup
    env_var_name = key_path.upper().replace('.', '_')
    env_value = os.getenv(env_var_name)

    if env_value is not None:
        # Attempt to infer type for common cases (bool, int, float)
        if env_value.lower() in ['true', 'false']:
            logger.debug(f"Using environment variable override for '{key_path}' (as bool): {env_value.lower() == 'true'}")
            return env_value.lower() == 'true'
        try:
            logger.debug(f"Using environment variable override for '{key_path}' (as int): {int(env_value)}")
            return int(env_value)
        except ValueError:
            try:
                logger.debug(f"Using environment variable override for '{key_path}' (as float): {float(env_value)}")
                return float(env_value)
            except ValueError:
                logger.debug(f"Using environment variable override for '{key_path}' (as string): '{env_value}'")
                return env_value # Return as string if not bool, int, or float

    # 2. If no env var, check the YAML config
    config = load_config()
    keys = key_path.split(".")
    val = config
    try:
        for key in keys:
            val = val[key]
        return val if val is not None or default is None else default
    except (KeyError, TypeError):
        return default


_logging_configured = False


def validate_config_schema(config_data: dict):
    """
    Placeholder for validating the structure and types of the loaded configuration.
    This can be expanded with a schema validation library like jsonschema or Pydantic.
    """
    required_top_level_keys = ["model_defaults", "training_defaults", "logging", "evaluation", "scraper", "gui_settings"]
    for key in required_top_level_keys:
        if key not in config_data:
            logger.warning(f"Configuration validation: Missing expected top-level key '{key}'.")

    if "logging" in config_data and not isinstance(config_data["logging"], dict):
        logger.warning("Configuration validation: 'logging' section should be a dictionary.")
    # Add more specific checks as needed

def setup_logging():
    global _logging_configured
    if _logging_configured:
        return

    # Load logging configuration and ensure correct types
    log_file_config = get_config_value("logging.log_file", "project_ai.log")
    if not isinstance(log_file_config, str):
        print( # Use print here as logger might not be fully set up for this specific warning path
            f"Configuration 'logging.log_file' has unexpected type {type(log_file_config)}. Using default 'project_ai.log'."
        )
        log_file = "project_ai.log"
    else:
        log_file = log_file_config

    log_level_console_config = get_config_value("logging.console_level", "INFO")
    if not isinstance(log_level_console_config, str):
        print(
            f"Configuration 'logging.console_level' has unexpected type {type(log_level_console_config)}. Using default 'INFO'."
        )
        log_level_console_str = "INFO"
    else:
        log_level_console_str = log_level_console_config

    log_level_file_config = get_config_value("logging.file_level", "DEBUG")
    if not isinstance(log_level_file_config, str):
        print(
            f"Configuration 'logging.file_level' has unexpected type {type(log_level_file_config)}. Using default 'DEBUG'."
        )
        log_level_file_str = "DEBUG"
    else:
        log_level_file_str = log_level_file_config

    # Ensure log file directory exists if log_file is specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    root_logger = logging.getLogger()  # Root logger, renamed from 'logger'
    root_logger.setLevel(
        logging.DEBUG
    )  # Set root logger level to the lowest of all handlers

    # Prevent duplicate messages if this function is called multiple times (e.g., in tests)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)  # Explicitly use sys.stdout
    try:
        console_level = getattr(logging, log_level_console_str.upper(), logging.INFO)
    except AttributeError:
        console_level = logging.INFO  # Default to INFO if string is invalid
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File Handler (Rotating)
    if (
        log_file and log_file.lower() != "none" and log_file.lower() != "null"
    ):  # Check if file logging is enabled
        try:
            file_level = getattr(logging, log_level_file_str.upper(), logging.DEBUG)
        except AttributeError:
            file_level = logging.DEBUG  # Default to DEBUG if string is invalid

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",  # 5MB per file, 3 backups
        )
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(filename)s:%(lineno)d - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        root_logger.info( # Use root_logger here
            "Logging setup complete. Console Level: %s, File Logging to: %s (Level: %s)",
            log_level_console_str,
            log_file,
            log_level_file_str,
        )
    else:
        root_logger.info( # Use root_logger here
            "Logging setup complete. Console Level: %s. File logging is disabled via config.",
            log_level_console_str,
        )

    _logging_configured = True


# Initial call to setup logging when utils.py is first imported.
# This ensures that any module importing from utils will have logging configured.
# However, main scripts should still call it to ensure it happens early.
# This initial call can use hardcoded defaults or rely on config being available.
# For robustness, let it rely on its internal config loading.
# setup_logging() # Removing this initial call to let main scripts control it explicitly.
# This avoids issues if config is not yet fully processed or if a script wants to delay/override.
# The first logger = logging.getLogger(__name__) in a script will get a basic logger if not configured.
# The main scripts' call to setup_logging() will then properly configure it.
# Adding import sys for StreamHandler
