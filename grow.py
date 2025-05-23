# grow.py
import torch
import torch.nn as nn
from python_master_ai import PythonMasterAI
import requests
import copy
from torch.optim import Adam

MASTER_KEY = PythonMasterAI.MASTER_KEY

def grow_model(model):
    if model.assess_performance()["needs_growth"]:
        try:
            response = requests.post("http://localhost:8000/master/auth", json={"key": MASTER_KEY, "command": "APPROVE_GROWTH"})
            response.raise_for_status() # Check for HTTP errors
            approval_data = response.json()
            if approval_data.get("action") != "grow":
                raise Exception("Master approval required and action must be 'grow'.")
        except requests.exceptions.RequestException as e:
            print(f"Network error during master approval: {e}")
            raise
        except Exception as e: # Catches JSONDecodeError or the custom one above
            print(f"Error during master approval: {e}")
            raise

        print("Master approval received. Growing model...")

        if model.transformer.encoder.layers: # Check if there are existing layers in the live model
            print("Growing from existing layers.")
            last_layer_config = model.transformer.encoder.layers[-1]
            # Create a new layer, deepcopying the configuration and state from the last one
            new_layer = copy.deepcopy(last_layer_config)
            # Scale down weights of the new layer
            with torch.no_grad(): # Ensure operations don't track gradients
                for param in new_layer.parameters():
                    param.mul_(0.1) # Ensure small initial impact
            model.transformer.encoder.layers.append(new_layer)
        else:
            print("No existing layers in encoder. Creating a new layer from model defaults.")
            new_layer = nn.TransformerEncoderLayer(
                d_model=model.hidden_size,
                nhead=model.n_heads,
                dim_feedforward=getattr(model, 'dim_feedforward', model.hidden_size * 4),
                dropout=getattr(model, 'dropout', 0.1),
                activation=getattr(model, 'activation', "relu"),
                batch_first=True  # Crucial for consistency
            )
            model.transformer.encoder.layers.append(new_layer)

        model.n_layers += 1
        model.update_stage()
        params = sum(p.numel() for p in model.parameters())
        print(f"{model.stage.capitalize()} grown to {params:,} parameters")
        return model, Adam(model.parameters(), lr=1e-4)

    return model, None

if __name__ == "__main__":
    model = PythonMasterAI()
    model, optimizer = grow_model(model)
