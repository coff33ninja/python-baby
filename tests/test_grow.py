# d:\Scripts\python-baby\tests\test_grow.py
import pytest
import torch
import torch.nn as nn # Import nn for nn.ModuleList
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

# Minimal nn.Module subclass to act as the layer returned by the mocked constructor.
# Its 'parameters' method will be replaced by a MagicMock on the instance.
class MockEncoderLayerModule(nn.Module):
    def __init__(self, d_model=None, nhead=None, dim_feedforward=None, dropout=None, activation=None, batch_first=None, layer_norm_eps=None, norm_first=None): # Match expected signature
        super().__init__()
        # It can be empty or have dummy nn.Parameters if needed for other reasons,
        # but for this test, we'll mock its 'parameters' method directly on the instance.
        pass # No specific parameters needed here if we mock the method

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        return src


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
            
            with pytest.raises(Exception, match="Master approval required and action must be 'grow'\\."):
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
            
            with pytest.raises(Exception, match="Master approval required and action must be 'grow'\\."):
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

@patch('grow.nn.TransformerEncoderLayer', autospec=True) # Patch with autospec
def test_grow_model_success_initially_no_layers_in_encoder(mock_transformer_encoder_layer_constructor, base_model):
    """Test successful growth when encoder initially has no layers (testing the 'if old_layers:' branch)."""
    # Replace the ModuleList with a new empty one.
    # PyTorch's nn.Module handles reassignment of nn.Module attributes by updating _modules.
    base_model.transformer.encoder.layers = nn.ModuleList()
    # Set the model's n_layers attribute to 0 to reflect that we want to test the "no initial layers" state
    base_model.n_layers = 0 
    initial_n_layers_attr = base_model.n_layers # This will now be 0

    # Configure the mock constructor to return an instance of our nn.Module mock
    # This instance will have its own 'parameters' MagicMock method.
    # This is the actual nn.Module instance that will be added to the model
    actual_module_instance = MockEncoderLayerModule() 
    mock_transformer_encoder_layer_constructor.return_value = actual_module_instance

    # We need to mock the 'parameters' *method* of this specific instance.
    # Create a MagicMock that will replace the 'parameters' method.
    mock_parameters_method = MagicMock(name="parameters_method_mock")
    
    # Configure what this mocked 'parameters' method will return (an iterator of mock nn.Parameters)
    mock_param1 = MagicMock(spec=nn.Parameter) # Mock an nn.Parameter
    mock_param1.numel.return_value = 10 
    mock_param2 = MagicMock(spec=nn.Parameter) # Mock another nn.Parameter
    mock_param2.numel.return_value = 20
    mock_parameters_method.return_value = iter([mock_param1, mock_param2])

    # Replace the 'parameters' method of our specific module instance with our MagicMock
    actual_module_instance.parameters = mock_parameters_method

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
                assert base_model.transformer.encoder.layers[0] is actual_module_instance

                mock_post.assert_called_once()
                mock_update_stage.assert_called_once()

                # Assert that TransformerEncoderLayer was called with correct args from base_model
                # (assuming grow_model uses these attributes from the model)
                # For parameters not directly on base_model, we assume grow_model uses
                # defaults consistent with PythonMasterAI's typical setup or explicit values.

                # Expected values based on PythonMasterAI defaults or grow_model logic
                expected_dim_feedforward = getattr(base_model, 'dim_feedforward', base_model.hidden_size * 4)
                expected_dropout = getattr(base_model, 'dropout', 0.1)
                expected_activation = getattr(base_model, 'activation', "relu")

                mock_transformer_encoder_layer_constructor.assert_called_once_with(
                    d_model=base_model.hidden_size,
                    nhead=base_model.n_heads,
                    dim_feedforward=expected_dim_feedforward,
                    dropout=expected_dropout,
                    activation=expected_activation,
                    batch_first=True # Asserting that grow_model now passes this
                )
                
                # Assert that the parameters of the (mocked) new layer are accessed (e.g., by the optimizer)
                # Scaling by mul_(0.1) is not expected here as there's no prior layer to scale from.
                mock_parameters_method.assert_called_once()