# grow.py
import torch
# import torch.nn as nn # No longer directly creating layers here
from python_master_ai import PythonMasterAI
import requests
# import copy # No longer deepcopying layers here
from torch.optim import Adam
import os # For checkpointing

MASTER_KEY = PythonMasterAI.MASTER_KEY
CHECKPOINT_DIR = "checkpoints" # For saving the new grown model's checkpoint

def grow_model(current_model: PythonMasterAI):
    """
    Attempts to grow the model by creating a new instance with increased complexity
    (e.g., more layers) and transferring compatible weights from the current_model.
    Saves an initial checkpoint for the newly grown model.
    """
    if not current_model.assess_performance()["needs_growth"]:
        print(f"Model at stage '{current_model.stage}' does not meet criteria for growth or growth not needed.")
        return current_model, None # Return current model and no new optimizer

    # --- Master Approval ---
    try:
        response = requests.post("http://localhost:8000/master/auth", json={"key": MASTER_KEY, "command": "APPROVE_GROWTH"})
        response.raise_for_status()
        approval_data = response.json()
        if approval_data.get("action") != "grow":
            print("Master approval required, but 'grow' action not granted.")
            return current_model, None
    except requests.exceptions.RequestException as e:
        print(f"Network error during master approval: {e}")
        return current_model, None
    except Exception as e:
        print(f"Error during master approval: {e}")
        return current_model, None

    print("Master approval received. Proceeding with model growth...")

    # --- Determine New Configuration ---
    new_n_layers = current_model.n_layers + 1 # Example: Increment layers
    # Other parameters are inherited from the current model
    # Note: vocab_size, n_heads, hidden_size etc., are kept same for direct weight transfer for many layers.
    # If these were to change, weight transfer would be more complex or impossible for some layers.
    print(f"Current model: {current_model.n_layers} layers. New model will have: {new_n_layers} layers.")

    # --- Create New Model Instance with Weight Transfer ---
    print(f"Creating new grown model, attempting to transfer weights from current model (Config ID: {current_model.configuration_id})...")
    grown_model = PythonMasterAI(
        vocab_size=current_model.vocab_size,
        n_layers=new_n_layers,
        n_heads=current_model.n_heads,
        hidden_size=current_model.hidden_size,
        dropout=current_model.dropout,
        dim_feedforward=current_model.dim_feedforward,
        activation=current_model.activation,
        previous_model_state_dict=current_model.state_dict() # Pass weights
    )
    print(f"New grown model created. Initial Configuration ID: {grown_model.configuration_id}")

    # --- Update Stage and State for the New Model ---
    # The grown_model's __init__ already sets its initial stage based on its config.
    # Call update_stage() to potentially promote it further based on its new parameters.
    grown_model.update_stage() # This will print promotion if it happens
    print(f"Grown model is now at stage: '{grown_model.stage}' with {sum(p.numel() for p in grown_model.parameters()):,} parameters.")
    # Note: grown_model starts with fresh task_progress, knowledge_gaps, etc.

    # --- Save Initial Checkpoint for the Grown Model ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    new_optimizer = Adam(grown_model.parameters(), lr=1e-4) # Create optimizer for the new model

    checkpoint_data = {
        'epoch': 0,  # Initial checkpoint for the newly grown model
        'model_state_dict': grown_model.state_dict(),
        'optimizer_state_dict': new_optimizer.state_dict(),
        'loss': None,  # No training loss yet
        'ai_state': grown_model.get_state_for_checkpoint() # Get all other AI state
    }

    epoch0_filename = f"model_stage_{grown_model.stage}_config_{grown_model.configuration_id}_epoch_0.pt"
    latest_filename = f"model_stage_{grown_model.stage}_config_{grown_model.configuration_id}_latest.pt"
    
    epoch0_filepath = os.path.join(CHECKPOINT_DIR, epoch0_filename)
    latest_filepath = os.path.join(CHECKPOINT_DIR, latest_filename)

    try:
        torch.save(checkpoint_data, epoch0_filepath)
        torch.save(checkpoint_data, latest_filepath)
        print(f"Saved initial checkpoint for grown model (Epoch 0) to {epoch0_filepath} and as {latest_filepath}")
    except Exception as e:
        print(f"ERROR: Failed to save initial checkpoint for grown model: {e}")
        # Depending on policy, might want to raise this or handle differently

    return grown_model, new_optimizer


if __name__ == "__main__":
    print("--- Initializing base model ---")
    # This will try to load the latest checkpoint for its default config if one exists
    base_model = PythonMasterAI() 
    print(f"Base model initialized. Stage: {base_model.stage}, Config ID: {base_model.configuration_id}, Layers: {base_model.n_layers}")
    print(f"Base model device: {base_model.device}")

    # Simulate that the base model meets growth criteria for testing
    # In a real scenario, this would be based on actual performance metrics
    base_model.performance_log.append(("loss", 0.1)) # Simulate low loss
    base_model.task_progress["unit_test_accuracy"] = 0.9 # Simulate high accuracy
    # Ensure all tasks for the current stage are met (simplified)
    if base_model.stage in base_model.growth_tasks:
        for task in base_model.growth_tasks[base_model.stage]:
            if isinstance(base_model.growth_tasks[base_model.stage][task], int):
                 base_model.task_progress[task] = base_model.growth_tasks[base_model.stage][task]
    
    print(f"\n--- Attempting to grow model (Master approval will be simulated via HTTP request to localhost:8000) ---")
    # Note: For this test to pass the approval step, the master_key.py server must be running.
    grown_model_instance, grown_optimizer = grow_model(base_model)

    if grown_optimizer: # Check if growth actually happened and a new optimizer was returned
        print("\n--- Growth Occurred ---")
        print(f"Original model - Stage: {base_model.stage}, Config ID: {base_model.configuration_id}, Layers: {base_model.n_layers}")
        print(f"Grown model    - Stage: {grown_model_instance.stage}, Config ID: {grown_model_instance.configuration_id}, Layers: {grown_model_instance.n_layers}")
        print(f"Grown model device: {grown_model_instance.device}")
        # Further actions with grown_model_instance and grown_optimizer (e.g., start training)
    else:
        print("\n--- Growth Did Not Occur (or was not approved/needed) ---")
        print(f"Current model remains - Stage: {base_model.stage}, Config ID: {base_model.configuration_id}, Layers: {base_model.n_layers}")
