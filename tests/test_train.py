import pytest
from unittest.mock import MagicMock
import types
from train import train

# test_train.py

# Absolute import for train

@pytest.fixture
def mock_dependencies(monkeypatch):
    # Patch PythonMasterAI
    mock_model = MagicMock()
    mock_model.stage = "stage1"
    mock_model.hidden_size = 8
    mock_model.device = "cpu"
    mock_model.parameters.return_value = [MagicMock()]
    mock_model.assess_performance.return_value = {"needs_research": False}
    mock_model.prioritize_scraping.return_value = []
    mock_model.known_sources = {}
    mock_model.get_latest_dataset_path.return_value = "/tmp/dataset"
    mock_model.get_state_for_checkpoint.return_value = {"dummy": True}
    mock_model.state_dict.return_value = {"weights": [1, 2, 3]}
    mock_model.configuration_id = "test"
    mock_model.log_performance = MagicMock()
    mock_model.log_task_progress = MagicMock()
    mock_model.process_scraped_research_data = MagicMock()
    mock_model.generate_for_evaluation.return_value = "def foo():\n    return 42"
    mock_model.current_dataset_version = None
    mock_model.define_growth_tasks.return_value = {"stage1": {}, "stage2": {}}
    mock_model.get_research_scrape_targets.return_value = []
    mock_model.discover_new_sources = MagicMock()

    monkeypatch.setattr("train.model", mock_model)

    # Patch tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "<|pad|>"
    mock_tokenizer.eos_token = "<|eos|>"
    mock_tokenizer.return_value = {
        "input_ids": MagicMock(to=MagicMock(return_value=MagicMock())),
        "attention_mask": MagicMock(to=MagicMock(return_value=MagicMock()))
    }
    monkeypatch.setattr("train.tokenizer", mock_tokenizer)

    # Patch scrape_data
    monkeypatch.setattr("train.scrape_data", MagicMock())

    # Patch load_dataset to return a dummy dataset
    class DummyDataset:
        def __len__(self): return 2
        def __getitem__(self, idx): return {"text": "hello world"}
        def __iter__(self): return iter([{"text": "hello world"}, {"text": "bye world"}])
    monkeypatch.setattr("train.load_dataset", MagicMock(return_value=DummyDataset()))

    # Patch DataLoader to just iterate over the dataset
    monkeypatch.setattr("train.DataLoader", lambda ds, batch_size: [ds[0], ds[1]])

    # Patch torch.save and os.makedirs
    monkeypatch.setattr("train.torch.save", MagicMock())
    monkeypatch.setattr("train.os.makedirs", MagicMock())

    # Patch glob.glob to return dummy txt files
    monkeypatch.setattr("train.glob.glob", MagicMock(return_value=["/tmp/dataset/file1.txt", "/tmp/dataset/file2.txt"]))

    # Patch os.path.exists to True for dataset and manifest
    monkeypatch.setattr("train.os.path.exists", MagicMock(side_effect=lambda p: "manifest.json" in p or "dataset" in p))

    # Patch open for manifest.json
    def fake_open(file, mode="r", *args, **kwargs):
        if "manifest.json" in file:
            return types.SimpleNamespace(__enter__=lambda s: s, __exit__=lambda s, exc_type, exc_val, exc_tb: None, read=lambda: '{"excluded_files": []}', __iter__=lambda s: iter(['{"excluded_files": []}']))
        else:
            raise FileNotFoundError
    monkeypatch.setattr("builtins.open", fake_open)

    # Patch json.load to return empty exclusions
    monkeypatch.setattr("train.json.load", MagicMock(return_value={"excluded_files": []}))

    # Patch get_typed_config_value and get_config_value
    monkeypatch.setattr("train.get_typed_config_value", lambda k, d, t: d)
    monkeypatch.setattr("train.get_config_value", lambda k, d=None: d if d is not None else "checkpoints")

    # Patch torch.nn.CrossEntropyLoss
    class DummyLoss:
        def __call__(self, a, b): return MagicMock(item=lambda: 0.5, backward=MagicMock())
    monkeypatch.setattr("train.nn.CrossEntropyLoss", DummyLoss)

    # Patch torch.nn.utils.clip_grad_norm_
    monkeypatch.setattr("train.torch.nn.utils.clip_grad_norm_", MagicMock())

    # Patch StepLR
    class DummyScheduler:
        def step(self): pass
        def get_last_lr(self): return [0.001]
    monkeypatch.setattr("train.StepLR", lambda opt, step_size, gamma: DummyScheduler())

    # Patch TensorBoard SummaryWriter
    monkeypatch.setattr("train.writer", None)
    monkeypatch.setattr("train.SummaryWriter", MagicMock())

    return mock_model

def test_train_runs_successfully(mock_dependencies, caplog):
    # Should run without error and log expected messages
    with caplog.at_level("INFO"):
        train("stage1")
    assert any("Train function initiated for stage: stage1" in r.message for r in caplog.records)
    assert any("Training using dataset version:" in r.message for r in caplog.records)
    assert any("Starting training for" in r.message for r in caplog.records)
    assert any("Saved checkpoint for epoch" in r.message for r in caplog.records)

def test_train_aborts_if_no_dataset_path(monkeypatch, caplog):
    # Patch model.get_latest_dataset_path to return None
    mock_model = MagicMock()
    mock_model.stage = "stage1"
    monkeypatch.setattr("train.model", mock_model)
    monkeypatch.setattr("train.get_config_value", lambda k, d=None: d if d is not None else "checkpoints")
    monkeypatch.setattr("train.get_typed_config_value", lambda k, d, t: d)
    monkeypatch.setattr("train.logger", MagicMock())
    mock_model.get_latest_dataset_path.return_value = None
    mock_model.assess_performance.return_value = {"needs_research": False}
    mock_model.prioritize_scraping.return_value = []
    mock_model.known_sources = {}
    mock_model.process_scraped_research_data = MagicMock()
    with caplog.at_level("ERROR"):
        train("stage1")
    assert any("Could not determine latest dataset path" in r.message for r in caplog.records)

def test_train_aborts_if_no_data_files(monkeypatch, mock_dependencies, caplog):
    # Patch glob.glob to return only excluded files
    monkeypatch.setattr("train.glob.glob", MagicMock(return_value=["/tmp/dataset/manifest.json"]))
    # Patch json.load to exclude all files
    monkeypatch.setattr("train.json.load", MagicMock(return_value={"excluded_files": ["file1.txt", "file2.txt"]}))
    with caplog.at_level("ERROR"):
        train("stage1")
    assert any("No data files remaining for training" in r.message for r in caplog.records)

def test_train_handles_manifest_exclusions(monkeypatch, mock_dependencies, caplog):
    # Patch glob.glob to return two files, one excluded
    monkeypatch.setattr("train.glob.glob", MagicMock(return_value=["/tmp/dataset/file1.txt", "/tmp/dataset/file2.txt"]))
    monkeypatch.setattr("train.json.load", MagicMock(return_value={"excluded_files": ["file2.txt"]}))
    with caplog.at_level("INFO"):
        train("stage1")
    assert any("Excluding files from training based on manifest" in r.message for r in caplog.records)
    assert any("Skipping excluded file: file2.txt" in r.message for r in caplog.records)

def test_train_tensorboard_writer(monkeypatch, mock_dependencies):
    # Patch get_config_value to enable tensorboard
    monkeypatch.setattr("train.get_config_value", lambda k, d=None: "runs/test" if "tensorboard_log_dir" in k else (d if d is not None else "checkpoints"))
    mock_writer = MagicMock()
    monkeypatch.setattr("train.SummaryWriter", lambda log_dir: mock_writer)
    monkeypatch.setattr("train.writer", None)
    train("stage1")
    assert mock_writer.add_scalar.called
    assert mock_writer.close.called