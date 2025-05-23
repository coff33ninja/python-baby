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
        response = requests.post("http://localhost:8000/master/auth", json={"key": MASTER_KEY, "command": "APPROVE_GROWTH"})
        if response.status_code == 200 and response.json().get("action") == "grow":
            # Take a snapshot of the layers before modification
            old_layers = copy.deepcopy(model.transformer.encoder.layers)
            new_layer = nn.TransformerEncoderLayer(d_model=model.hidden_size, nhead=model.n_heads)

            if old_layers: # If there were previous layers
                # Initialize new_layer's weights from the last layer in the snapshot
                new_layer.load_state_dict(old_layers[-1].state_dict())

            # Scale down the new layer's parameters (applies to copied or default initialized weights)
            with torch.no_grad():
                for param in new_layer.parameters():
                    param.mul_(0.1) # Ensure small initial impact

            model.transformer.encoder.layers.append(new_layer)
            model.n_layers += 1
            model.update_stage()
            params = sum(p.numel() for p in model.parameters())
            print(f"{model.stage.capitalize()} grown to {params:,} parameters")
            return model, Adam(model.parameters(), lr=1e-4)
        raise Exception("Master approval required")
    return model, None

if __name__ == "__main__":
    model = PythonMasterAI()
    model, optimizer = grow_model(model)
