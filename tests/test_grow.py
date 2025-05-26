# d:\Scripts\python-baby\tests\test_grow.py
import pytest  # type: ignore
import torch
import torch.nn as nn # Import nn for nn.ModuleList
from unittest.mock import patch, MagicMock
import copy
import requests # For requests.RequestException

# Adjust import path based on how pytest discovers modules
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

            with pytest.raises(requests.exceptions.ConnectionError, match="Network fail"):
               grow_model(base_model)

            # Model should not have been modified if growth failed early
            mock_post.assert_called_once_with(
                "http://localhost:8000/master/auth",
                json={"key": MASTER_KEY, "command": "APPROVE_GROWTH"}
            )

def test_grow_model_master_approval_denied_status_code(base_model):
    """Test grow_model when master auth returns non-200 status."""
    mock_response = mock_requests_post_response(status_code=403, json_data={"detail": "Invalid key"})
    # Make raise_for_status() on the mock response object raise an HTTPError
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("403 Client Error")

    with patch.object(base_model, 'assess_performance', return_value={"needs_growth": True}):
        with patch('grow.requests.post', return_value=mock_response) as mock_post:

            with pytest.raises(requests.exceptions.HTTPError, match="403 Client Error"):
                grow_model(base_model)

            mock_post.assert_called_once_with(
                "http://localhost:8000/master/auth",
                json={"key": MASTER_KEY, "command": "APPROVE_GROWTH"}
            )

def test_grow_model_master_approval_denied_action(base_model):
    """Test grow_model when master auth returns 200 but action is not 'grow'."""
    mock_response = mock_requests_post_response(status_code=200, json_data={"action": "deny"})

    with patch.object(base_model, 'assess_performance', return_value={"needs_growth": True}):
        with patch('grow.requests.post', return_value=mock_response) as mock_post:

            with pytest.raises(Exception, match="Master approval required and action must be 'grow'\\."): # Match the new exception message
                grow_model(base_model)

            mock_post.assert_called_once_with(
                "http://localhost:8000/master/auth",
                json={"key": MASTER_KEY, "command": "APPROVE_GROWTH"}
            )

@patch('python_master_ai.PythonMasterAI.update_stage') # Patch at class level
@patch('utils.get_typed_config_value') # To mock config access for checkpoint_dir
def test_grow_model_success_with_existing_layers(mock_get_typed_config, mock_update_stage_on_new_model, base_model, tmp_path):
    """Test successful model growth when encoder has existing layers."""
    mock_response = mock_requests_post_response(status_code=200, json_data={"action": "grow"})

    # --- Arrange ---
    # Mock config for CHECKPOINT_DIR to use tmp_path
    mock_checkpoint_dir = tmp_path / "checkpoints"
    def config_side_effect(key, default=None, type_hint=None): # Match expected signature
        if key == "checkpointing.checkpoint_dir":
            return str(mock_checkpoint_dir)
        elif key == "training.growth_strategy_factor": # Used by grow_model
            return 2.0
        elif key == "training.growth_increment": # Used by grow_model
            return 1
        return default
    mock_get_typed_config.side_effect = config_side_effect

    with patch.object(base_model, 'assess_performance', return_value={"needs_growth": True}):
        with patch('grow.requests.post', return_value=mock_response) as mock_post:
            initial_n_layers = base_model.n_layers
            initial_model_config_id = base_model.configuration_id
            assert initial_n_layers > 0, "Model should have initial layers for this test"

            last_layer_state_dict_from_old_model = copy.deepcopy(base_model.transformer.encoder.layers[-1].state_dict())

            grown_model_instance, optimizer = grow_model(base_model)

            assert grown_model_instance is not base_model # The new model is returned
            assert isinstance(optimizer, Adam)
            assert grown_model_instance.n_layers == initial_n_layers + 1 # Check new model
            assert grown_model_instance.configuration_id != initial_model_config_id
            assert len(grown_model_instance.transformer.encoder.layers) == initial_n_layers + 1 # Check new model
            mock_post.assert_called_once()
            mock_update_stage_on_new_model.assert_called_once() # update_stage is called on the new grown_model instance

            # Check scaling of the newly added layer (it's seeded from the old model's last layer)
            newly_added_layer_in_grown_model = grown_model_instance.transformer.encoder.layers[-1]
            for param_name, param_tensor_from_old_model_last_layer in last_layer_state_dict_from_old_model.items():
                expected_tensor_after_scaling = param_tensor_from_old_model_last_layer * 0.5 # Scaling factor is 0.5
                actual_tensor_in_new_layer = newly_added_layer_in_grown_model.state_dict()[param_name]
                assert torch.allclose(actual_tensor_in_new_layer, expected_tensor_after_scaling, atol=1e-7), \
                    f"Parameter {param_name} in new layer not scaled correctly."

            # Verify checkpoint saving
            # grow_model calls grown_model.update_stage("growing") before saving
            expected_epoch0_filename = f"model_stage_growing_config_{grown_model_instance.configuration_id}_epoch_0.pt"
            expected_latest_filename = f"model_stage_growing_config_{grown_model_instance.configuration_id}_latest.pt"
            epoch0_filepath = mock_checkpoint_dir / expected_epoch0_filename
            latest_filepath = mock_checkpoint_dir / expected_latest_filename

            assert epoch0_filepath.exists(), "Epoch 0 checkpoint file should be saved"
            assert latest_filepath.exists(), "Latest checkpoint file should be saved"

# Patch torch.nn.TransformerEncoderLayer where it's used by torch.nn.Transformer
@patch('utils.get_typed_config_value') # To mock config access for checkpoint_dir
@patch('torch.nn.modules.transformer.TransformerEncoderLayer')
@patch('python_master_ai.PythonMasterAI.update_stage')
def test_grow_model_success_initially_no_layers_in_encoder(mock_update_stage_on_new_model, mock_torch_encoder_layer_constructor, mock_get_typed_config, base_model, tmp_path):
    """Test successful growth when encoder initially has no layers (testing the 'if old_layers:' branch)."""
    # --- Arrange ---
    # Replace the ModuleList with a new empty one.
    # PyTorch's nn.Module handles reassignment of nn.Module attributes by updating _modules.
    # Mock config for CHECKPOINT_DIR to use tmp_path
    mock_checkpoint_dir = tmp_path / "checkpoints"
    def config_side_effect(key, default=None, type_hint=None): # Match expected signature
        if key == "checkpointing.checkpoint_dir":
            return str(mock_checkpoint_dir)
        elif key == "training.growth_strategy_factor": return 2.0
        elif key == "training.growth_increment": return 1
        return default
    mock_get_typed_config.side_effect = config_side_effect

    base_model.transformer.encoder.layers = nn.ModuleList()
    # Set the model's n_layers attribute to 0 to reflect that we want to test the "no initial layers" state
    base_model.n_layers = 0
    initial_n_layers_attr = base_model.n_layers # This will now be 0

    # Make the mock_torch_encoder_layer_constructor (which replaces the class) behave like a type for isinstance
    # and ensure it's callable to return our mock instance.
    mock_torch_encoder_layer_constructor.__bases__ = (torch.nn.TransformerEncoderLayer,) # Make the mock class a "subclass"

    # Configure the mock constructor to return an instance of our nn.Module mock
    # This instance will have its own 'parameters' MagicMock method.
    # This is the actual nn.Module instance that will be added to the model
    actual_module_instance = MockEncoderLayerModule( # As per the prompt's refined version
        d_model=base_model.hidden_size,
        nhead=base_model.n_heads,
        dim_feedforward=base_model.dim_feedforward,
        dropout=base_model.dropout,
        activation=base_model.activation,
        batch_first=True
    )
    mock_torch_encoder_layer_constructor.return_value = actual_module_instance
    # We need to mock the 'named_parameters' *method* of this specific instance.
    # Adam optimizer calls model.parameters(), which internally calls named_parameters() recursively.
    mock_named_parameters_method = MagicMock(name="named_parameters_method_mock")

    # Configure what this mocked 'named_parameters' method will return
    # (an iterator of (name, mock nn.Parameter) tuples)
    mock_param1 = MagicMock(spec=nn.Parameter) # Mock an nn.Parameter
    mock_param1.requires_grad = True # Optimizers usually only care about params that require grad
    mock_param2 = MagicMock(spec=nn.Parameter) # Mock another nn.Parameter
    mock_param2.requires_grad = True
    # As per the prompt's refined version, mock parameters() directly
    actual_module_instance.parameters = MagicMock(return_value=iter([mock_param1, mock_param2]))

    mock_response = mock_requests_post_response(status_code=200, json_data={"action": "grow"})
    initial_model_config_id = base_model.configuration_id

    with patch.object(base_model, 'assess_performance', return_value={"needs_growth": True}):
        with patch('grow.requests.post', return_value=mock_response) as mock_post:
            # --- Act ---
            grown_model_instance, optimizer = grow_model(base_model)

            # --- Assert ---
            assert grown_model_instance is not base_model
            assert isinstance(optimizer, Adam)
            assert grown_model_instance.n_layers == initial_n_layers_attr + 1
            assert len(grown_model_instance.transformer.encoder.layers) == 1 # One new layer added
            # The TransformerEncoder uses copy.deepcopy(encoder_layer), so we can't use 'is'.
            # Instead, check the type and that the (deepcopied) mocked 'parameters' method was called.
            added_layer = grown_model_instance.transformer.encoder.layers[0]
            assert isinstance(added_layer, MockEncoderLayerModule), \
            "Layer added is not of the expected mock type MockEncoderLayerModule."
                
            # The 'named_parameters' attribute of actual_module_instance was replaced by mock_named_parameters_method.
            # When deepcopied, added_layer.named_parameters becomes a deepcopy of mock_named_parameters_method.
            # This deepcopied mock should have been called by the Adam optimizer.
            # As per the prompt's refined version, assert parameters() was called
            added_layer.parameters.assert_called_once()

            mock_post.assert_called_once()
            mock_update_stage_on_new_model.assert_called_once()

            # Assert that TransformerEncoderLayer was called with correct args from grown_model_instance
            # (grow_model uses attributes from the grown_model_instance for the new layer)

            expected_dim_feedforward = grown_model_instance.dim_feedforward
            expected_dropout = grown_model_instance.dropout
            expected_activation = grown_model_instance.activation

            mock_torch_encoder_layer_constructor.assert_called_once_with(
                d_model=grown_model_instance.hidden_size,
                nhead=grown_model_instance.n_heads,
                dim_feedforward=expected_dim_feedforward,
                dropout=expected_dropout,
                activation=expected_activation,
                batch_first=True
            )
            assert grown_model_instance.configuration_id != initial_model_config_id

            # Verify checkpoint saving
            expected_epoch0_filename = f"model_stage_growing_config_{grown_model_instance.configuration_id}_epoch_0.pt"
            expected_latest_filename = f"model_stage_growing_config_{grown_model_instance.configuration_id}_latest.pt"
            epoch0_filepath = mock_checkpoint_dir / expected_epoch0_filename
            latest_filepath = mock_checkpoint_dir / expected_latest_filename

            assert epoch0_filepath.exists(), "Epoch 0 checkpoint file should be saved for growth from no layers"
            assert latest_filepath.exists(), "Latest checkpoint file should be saved for growth from no layers"


@patch('python_master_ai.PythonMasterAI.update_stage')
@patch('utils.get_typed_config_value')
@patch('grow.Adam') # Mock torch.optim.Adam as imported in grow.py
@patch('grow.requests.post')
def test_grow_model_optimizer_creation_failure(
    mock_post, mock_adam, mock_get_typed_config, mock_update_stage, base_model, tmp_path
):
    """Test that grow_model handles failure during optimizer creation gracefully."""
    # --- Arrange ---
    # Mock master server approval
    mock_post.return_value = mock_requests_post_response(status_code=200, json_data={"action": "grow"})

    # Mock assess_performance to trigger growth
    base_model.assess_performance = MagicMock(return_value={"needs_growth": True})

    # Configure mock_get_typed_config for checkpoint_dir and other grow_model needs
    mock_checkpoint_dir = tmp_path / "checkpoints"
    def config_side_effect(key, default=None, type_hint=None):
        if key == "checkpointing.checkpoint_dir":
            return str(mock_checkpoint_dir)
        elif key == "training.growth_strategy_factor": return 2.0
        elif key == "training.growth_increment": return 1
        return default
    mock_get_typed_config.side_effect = config_side_effect

    # Simulate optimizer creation failure
    mock_adam.side_effect = Exception("Optimizer creation failed")

    # --- Act ---
    grown_model, grown_optimizer = grow_model(base_model)

    # --- Assert ---
    assert grown_model is not base_model # A new model instance should have been created
    assert grown_model.n_layers == base_model.n_layers + 1 # Model growth configuration should have occurred

    assert grown_optimizer is None, "Optimizer should be None due to creation failure"

    mock_post.assert_called_once() # Ensure master approval was sought
    mock_adam.assert_called_once() # Ensure Adam optimizer creation was attempted
    mock_update_stage.assert_called_once_with("growing") # Ensure model stage was updated

    # Check that checkpoint files are still created for the model itself,
    # as save_checkpoint in PythonMasterAI can handle optimizer=None.
    expected_epoch0_filename = f"model_stage_growing_config_{grown_model.configuration_id}_epoch_0.pt"
    expected_latest_filename = f"model_stage_growing_config_{grown_model.configuration_id}_latest.pt"
    assert (mock_checkpoint_dir / expected_epoch0_filename).exists(), "Epoch 0 checkpoint should be saved even if optimizer failed"
    assert (mock_checkpoint_dir / expected_latest_filename).exists(), "Latest checkpoint should be saved even if optimizer failed"
