import yaml
import os
import sys
import logging  # Added for logging
import logging.handlers  # Added for logging

CONFIG_FILE_PATH = "config.yaml"
_config_cache = None
# Initialize logger at module level for use in load_config
logger = logging.getLogger(__name__)



def load_config():
    global _config_cache
    if _config_cache is None:
        if os.path.exists(CONFIG_FILE_PATH):
            try:
                with open(CONFIG_FILE_PATH, "r") as f:
                    _config_cache = yaml.safe_load(f)
                    if _config_cache is None:
                        _config_cache = {} # Ensure it's a dict even if empty
                        logger.warning( # Changed from print
                            f"Warning: Configuration file {CONFIG_FILE_PATH} is empty. Using defaults where applicable."
                        )
                    # else:
                    # print(f"Loaded configuration from {CONFIG_FILE_PATH}") # This will be logged by setup_logging
            except yaml.YAMLError as e:
                logger.error( # Changed from print
                    f"Error parsing YAML configuration file {CONFIG_FILE_PATH}: {e}. Using defaults where applicable."
                )
                _config_cache = {}
        else:
            logger.warning( # Changed from print
                f"Warning: Configuration file {CONFIG_FILE_PATH} not found. Using defaults where applicable."
            )
            _config_cache = {}
    return _config_cache

def get_config_value(key_path: str, default=None):
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


def setup_logging():
    global _logging_configured
    if _logging_configured:
        return

    # Load logging configuration using get_config_value
    log_file = get_config_value(
        "logging.log_file", "project_ai.log"
    )  # Default to project_ai.log
    log_level_console_str = get_config_value("logging.console_level", "INFO")
    log_level_file_str = get_config_value("logging.file_level", "DEBUG")

    # Ensure log file directory exists if log_file is specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger()  # Root logger
    logger.setLevel(
        logging.DEBUG
    )  # Set root logger level to the lowest of all handlers

    # Prevent duplicate messages if this function is called multiple times (e.g., in tests)
    if logger.hasHandlers():
        logger.handlers.clear()

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
    logger.addHandler(console_handler)

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
        logger.addHandler(file_handler)

        logger.info(
            "Logging setup complete. Console Level: %s, File Logging to: %s (Level: %s)",
            log_level_console_str,
            log_file,
            log_level_file_str,
        )
    else:
        logger.info(
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
