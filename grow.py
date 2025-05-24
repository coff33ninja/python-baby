# grow.py
import torch
from python_master_ai import PythonMasterAI
import requests
from torch.optim import Adam
import os
import logging  # Added for logging
from typing import TypeVar, Type # Added for helper
from utils import get_config_value, setup_logging  # Added for config and logging

# --- Initialize logger for this module ---
logger = logging.getLogger(__name__)

MASTER_KEY = PythonMasterAI.MASTER_KEY

# --- Helper function for typed config values (local to this module) ---
_T_HELPER = TypeVar('_T_HELPER', float, int, str, bool)

def _get_typed_config_value(key: str, default_value: _T_HELPER, target_type: Type[_T_HELPER]) -> _T_HELPER:
    val = get_config_value(key, default_value) # get_config_value from utils returns Any
    # If val is already the exact target type (and not a bool masquerading as int/float if target is int/float)
    if isinstance(val, target_type) and not (target_type in (int, float) and isinstance(val, bool)):
        return val
    # If val is a type that can be directly converted (int, float, str, bool)
    if isinstance(val, (int, float, str, bool)):
        try:
            return target_type(val) # Attempt conversion
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Could not convert configured value '{str(val)[:100]}' for key '{key}' to {target_type.__name__}: {e}. "
                f"Using default value: {default_value}"
            )
            return default_value
    else: # val is some other unexpected type (e.g., dict, list)
        logger.warning(
            f"Configuration value for '{key}' is of unexpected type: {type(val)} (value: '{str(val)[:100]}'). "
            f"Using default value: {default_value}"
        )
        return default_value
CHECKPOINT_DIR = _get_typed_config_value("checkpointing.checkpoint_dir", "checkpoints", str)
# logger.info(f"Using checkpoint directory: {CHECKPOINT_DIR}") # Logged in main if run directly


def grow_model(current_model: PythonMasterAI):
    """
    Attempts to grow the model by creating a new instance with increased complexity
    (e.g., more layers) and transferring compatible weights from the current_model.
    Saves an initial checkpoint for the newly grown model.
    """
    if not current_model.assess_performance()["needs_growth"]:
        logger.info(
            f"Model at stage '{current_model.stage}' does not meet criteria for growth or growth not needed."
        )
        return current_model, None

    # --- Master Approval ---
    try:
        response = requests.post(
            "http://localhost:8000/master/auth",
            json={"key": MASTER_KEY, "command": "APPROVE_GROWTH"},
        )
        response.raise_for_status()
        approval_data = response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during master approval: {e}", exc_info=True)
        raise  # Re-raise the network exception as expected by the test
    except Exception as e:
        logger.error(f"Error during master approval: {e}", exc_info=True)
        # For other errors (like non-200 status that raise_for_status might cause, or JSON decode errors)
        raise Exception(f"Master approval failed: {e}") from e

    if approval_data.get("action") != "grow":
        logger.warning("Master approval required, but 'grow' action not granted by master service.")
        raise Exception("Master approval required and action must be 'grow'.")
    logger.info("Master approval received. Proceeding with model growth...")

    # --- Determine New Configuration ---
    new_n_layers = current_model.n_layers + 1
    logger.info(
        f"Current model: {current_model.n_layers} layers. New model will have: {new_n_layers} layers."
    )

    # --- Create New Model Instance with Weight Transfer ---
    logger.info(
        f"Creating new grown model, attempting to transfer weights from current model (Config ID: {current_model.configuration_id})..."
    )

    previous_config = current_model.get_config_dict()
    logger.debug(f"Previous model config for seeding: {previous_config}")

    # PythonMasterAI's __init__ will use its own config defaults if specific values are not passed
    grown_model = PythonMasterAI(
        # vocab_size, n_heads, hidden_size, dropout, activation are taken from config by default in PythonMasterAI.__init__
        # Only specify parameters that are changing or need to be explicitly copied if they are not default.
        # Here, we assume the new model largely inherits architecture except for n_layers.
        # If those were also configurable per-growth-step, they'd be fetched from config here.
        vocab_size=current_model.vocab_size,  # Explicitly copy from old model
        n_layers=new_n_layers,  # This is the change for growth
        n_heads=current_model.n_heads,  # Explicitly copy
        hidden_size=current_model.hidden_size,  # Explicitly copy
        dropout=current_model.dropout,  # Explicitly copy
        dim_feedforward=current_model.dim_feedforward,  # Explicitly copy
        activation=current_model.activation,  # Explicitly copy
        previous_model_state_dict=current_model.state_dict(),
        previous_model_config=previous_config,
    )
    logger.info(
        f"New grown model created. Initial Configuration ID: {grown_model.configuration_id}"
    )

    grown_model.update_stage()
    logger.info(
        f"Grown model is now at stage: '{grown_model.stage}' with {sum(p.numel() for p in grown_model.parameters()):,} parameters."
    )

    # --- Save Initial Checkpoint for the Grown Model ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Optimizer learning rate from config for the new optimizer
    default_lr = _get_typed_config_value("training_defaults.learning_rate", 1e-4, float)
    new_optimizer = Adam(grown_model.parameters(), lr=default_lr)
    logger.info(
        f"Created new Adam optimizer for grown model with learning rate: {default_lr}"
    )

    checkpoint_data = {
        "epoch": 0,
        "model_state_dict": grown_model.state_dict(),
        "optimizer_state_dict": new_optimizer.state_dict(),
        "loss": None,
        "ai_state": grown_model.get_state_for_checkpoint(),
    }

    epoch0_filename = f"model_stage_{grown_model.stage}_config_{grown_model.configuration_id}_epoch_0.pt"
    latest_filename = f"model_stage_{grown_model.stage}_config_{grown_model.configuration_id}_latest.pt"

    epoch0_filepath = os.path.join(CHECKPOINT_DIR, epoch0_filename)
    latest_filepath = os.path.join(CHECKPOINT_DIR, latest_filename)

    try:
        torch.save(checkpoint_data, epoch0_filepath)
        torch.save(checkpoint_data, latest_filepath)
        logger.info(
            f"Saved initial checkpoint for grown model (Epoch 0) to {epoch0_filepath} and as {latest_filepath}"
        )
    except Exception as e:
        logger.error(
            f"Failed to save initial checkpoint for grown model: {e}", exc_info=True
        )

    return grown_model, new_optimizer


if __name__ == "__main__":
    # Setup logging as early as possible
    # setup_logging() will read from config itself.
    setup_logging()

    logger.info(f"Using checkpoint directory: {CHECKPOINT_DIR}")
    logger.info("--- Initializing base model ---")

    base_model = PythonMasterAI()
    logger.info(
        f"Base model initialized. Stage: {base_model.stage}, Config ID: {base_model.configuration_id}, Layers: {base_model.n_layers}"
    )
    logger.info(f"Base model device: {base_model.device}")

    # Simulate that the base model meets growth criteria for testing
    base_model.performance_log.append(("loss", 0.1))
    base_model.task_progress["unit_test_accuracy"] = 0.9
    if base_model.stage in base_model.growth_tasks:
        for task in base_model.growth_tasks[base_model.stage]:
            if isinstance(base_model.growth_tasks[base_model.stage][task], int):
                base_model.task_progress[task] = base_model.growth_tasks[
                    base_model.stage
                ][task]

    logger.info(
        "\n--- Attempting to grow model (Master approval will be simulated via HTTP request to localhost:8000) ---"
    )
    grown_model_instance, grown_optimizer = grow_model(base_model)

    if grown_optimizer:
        logger.info("\n--- Growth Occurred ---")
        logger.info(
            f"Original model - Stage: {base_model.stage}, Config ID: {base_model.configuration_id}, Layers: {base_model.n_layers}"
        )
        logger.info(
            f"Grown model    - Stage: {grown_model_instance.stage}, Config ID: {grown_model_instance.configuration_id}, Layers: {grown_model_instance.n_layers}"
        )
        logger.info(f"Grown model device: {grown_model_instance.device}")
    else:
        logger.info("\n--- Growth Did Not Occur (or was not approved/needed) ---")
        logger.info(
            f"Current model remains - Stage: {base_model.stage}, Config ID: {base_model.configuration_id}, Layers: {base_model.n_layers}"
        )
