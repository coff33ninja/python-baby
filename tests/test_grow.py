# d:\Scripts\python-baby\tests\test_grow.py
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import copy
import requests # For requests.RequestException

# Adjust import path based on how pytest discovers modules
# This assumes 'python-baby' is in PYTHONPATH or tests are run from 'python-baby' root
from python_master_ai import PythonMasterAI
from grow import grow_model, MASTER_KEY
from torch.optim import Adam

# Helper to create a mock response for requests.post
def mock_requests_post_response(status_code, json_data=None, raise_exception=None):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    if json_data is not None:
        mock_resp.json.return_value = json_data
    # If the post call itself raises an exception (e.g. ConnectionError)
    # it's usually passed as side_effect to the patch directly.
    # This helper is more for mocking the Response object's behavior.
    if raise_exception:
        mock_resp.raise_for_status.side_effect = raise_exception
    return mock_resp

@pytest.fixture
def base_model():
    """Provides a fresh PythonMasterAI model for each test."""
    model = PythonMasterAI()
    # Default n_layers is 2, hidden_size=256, n_heads=4
    assert len(model.transformer.encoder.layers) == model.n_layers
    return model

def test_grow_model_no_growth_needed(base_model):
    """Test that grow_model does nothing if needs_growth is False."""
    with patch.object(base_model, 'assess_performance', return_value={"needs_growth": False}):
        with patch('grow.requests.post') as mock_post: # Patch requests.post in the 'grow' module
            initial_n_layers = base_model.n_layers
            
            grown_model, optimizer = grow_model(base_model)
            
            assert grown_model is base_model # Should be the same instance
            assert optimizer is None
            assert base_model.n_layers == initial_n_layers
            mock_post.assert_not_called()

def test_grow_model_master_approval_network_failure(base_model):
    """Test grow_model when master auth call fails (network error)."""
    with patch.object(base_model, 'assess_performance', return_value={"needs_growth": True}):
        with patch('grow.requests.post', side_effect=requests.exceptions.ConnectionError("Network fail")) as mock_post:
            initial_n_layers = base_model.n_layers
            
            with pytest.raises(requests.exceptions.ConnectionError):
                grow_model(base_model)
            
            assert base_model.n_layers == initial_n_layers
            mock_post.assert_called_once_with(
                "http://localhost:8000/master/auth",
                json={"key": MASTER_KEY, "command": "APPROVE_GROWTH"}
            )

def test_grow_model_master_approval_denied_status_code(base_model):
    """Test grow_model when master auth returns non-200 status."""
    mock_response = mock_requests_post_response(status_code=403, json_data={"detail": "Invalid key"})
    
    with patch.object(base_model, 'assess_performance', return_value={"needs_growth": True}):
        with patch('grow.requests.post', return_value=mock_response) as mock_post:
            initial_n_layers = base_model.n_layers
            
            with pytest.raises(Exception, match="Master approval required"):
                grow_model(base_model)
            
            assert base_model.n_layers == initial_n_layers
            mock_post.assert_called_once_with(
                "http://localhost:8000/master/auth",
                json={"key": MASTER_KEY, "command": "APPROVE_GROWTH"}
            )

def test_grow_model_master_approval_denied_action(base_model):
    """Test grow_model when master auth returns 200 but action is not 'grow'."""
    mock_response = mock_requests_post_response(status_code=200, json_data={"action": "deny"})
    
    with patch.object(base_model, 'assess_performance', return_value={"needs_growth": True}):
        with patch('grow.requests.post', return_value=mock_response) as mock_post:
            initial_n_layers = base_model.n_layers
            
            with pytest.raises(Exception, match="Master approval required"):
                grow_model(base_model)
            
            assert base_model.n_layers == initial_n_layers
            mock_post.assert_called_once_with(
                "http://localhost:8000/master/auth",
                json={"key": MASTER_KEY, "command": "APPROVE_GROWTH"}
            )

def test_grow_model_success_with_existing_layers(base_model):
    """Test successful model growth when encoder has existing layers."""
    mock_response = mock_requests_post_response(status_code=200, json_data={"action": "grow"})
    
    with patch.object(base_model, 'assess_performance', return_value={"needs_growth": True}):
        with patch('grow.requests.post', return_value=mock_response) as mock_post:
            with patch.object(base_model, 'update_stage') as mock_update_stage:
                initial_n_layers = base_model.n_layers
                assert initial_n_layers > 0, "Model should have initial layers for this test"
                
                last_layer_before_growth_state_dict = copy.deepcopy(base_model.transformer.encoder.layers[-1].state_dict())

                grown_model, optimizer = grow_model(base_model)

                assert grown_model is base_model
                assert isinstance(optimizer, Adam)
                assert base_model.n_layers == initial_n_layers + 1
                assert len(base_model.transformer.encoder.layers) == initial_n_layers + 1
                mock_post.assert_called_once()
                mock_update_stage.assert_called_once()

                newly_added_layer = base_model.transformer.encoder.layers[-1]
                for param_name, param_tensor_before in last_layer_before_growth_state_dict.items():
                    expected_tensor_after_scaling = param_tensor_before * 0.1
                    actual_tensor_in_new_layer = newly_added_layer.state_dict()[param_name]
                    assert torch.allclose(actual_tensor_in_new_layer, expected_tensor_after_scaling, atol=1e-7), \
                        f"Parameter {param_name} in new layer not scaled correctly."

@patch('grow.nn.TransformerEncoderLayer') # Patch where nn.TransformerEncoderLayer is used in grow.py
def test_grow_model_success_initially_no_layers_in_encoder(mock_transformer_encoder_layer_constructor, base_model):
    """Test successful growth when encoder initially has no layers (testing the 'if old_layers:' branch)."""
    base_model.transformer.encoder.layers = nn.ModuleList() # Manually empty layers
    initial_n_layers_attr = base_model.n_layers # Store original attribute value

    # Configure the mock TransformerEncoderLayer that grow_model will create
    mock_created_layer_instance = MagicMock(spec=nn.TransformerEncoderLayer)
    
    # Mock parameters for the created layer
    # We'll mock two parameters to check the scaling logic.
    mock_param1_data = MagicMock()
    mock_param2_data = MagicMock()
    mock_param1 = MagicMock(data=mock_param1_data)
    mock_param2 = MagicMock(data=mock_param2_data)
    mock_created_layer_instance.parameters.return_value = [mock_param1, mock_param2]
    
    mock_transformer_encoder_layer_constructor.return_value = mock_created_layer_instance

    mock_response = mock_requests_post_response(status_code=200, json_data={"action": "grow"})
    
    with patch.object(base_model, 'assess_performance', return_value={"needs_growth": True}):
        with patch('grow.requests.post', return_value=mock_response) as mock_post:
            with patch.object(base_model, 'update_stage') as mock_update_stage:
                grown_model, optimizer = grow_model(base_model)
    
                assert grown_model is base_model
                assert isinstance(optimizer, Adam)
                assert base_model.n_layers == initial_n_layers_attr + 1 
                assert len(base_model.transformer.encoder.layers) == 1 # One new layer added
                # Ensure the layer added to the model is our mocked instance
                assert base_model.transformer.encoder.layers[0] is mock_created_layer_instance
    
                mock_post.assert_called_once()
                mock_update_stage.assert_called_once()
    
                # Assert that TransformerEncoderLayer was called with correct args from base_model
                # (assuming grow_model uses these attributes from the model)
                mock_transformer_encoder_layer_constructor.assert_called_once_with(
                    d_model=base_model.hidden_size,
                    nhead=base_model.n_heads,
                    dim_feedforward=base_model.dim_feedforward,
                    dropout=base_model.dropout, # Assuming attribute name is 'dropout' in PythonMasterAI
                    activation=base_model.activation, # Assuming attribute name is 'activation'
                    batch_first=False # Consistent with the UserWarning implying batch_first is False
                )
                
                # Assert that the parameters of the (mocked) new layer were scaled
                mock_created_layer_instance.parameters.assert_called_once()
                mock_param1_data.mul_.assert_called_once_with(0.1)
                mock_param2_data.mul_.assert_called_once_with(0.1)