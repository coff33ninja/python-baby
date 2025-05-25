import pytest
from unittest.mock import patch, MagicMock, mock_open
from python_master_ai import PythonMasterAI
import os

# test_python_master_ai.py


@pytest.fixture
def ai(monkeypatch, tmp_path):
    # Patch logger to avoid real logging
    monkeypatch.setattr("python_master_ai.logger", MagicMock())

    # Patch os.path.join to behave normally
    monkeypatch.setattr("python_master_ai.os.path.join", os.path.join)

    # Patch os.path.isdir to always True for dataset path
    monkeypatch.setattr("python_master_ai.os.path.isdir", lambda path: True)

    # Patch open for reading latest.txt and research data files
    def open_side_effect(file, mode="r", *args, **kwargs):
        if file.endswith("latest.txt"):
            m = mock_open(read_data="20240101").return_value
            m.__iter__.return_value = iter(["20240101"])
            return m
        elif file.endswith(".txt"):
            m = mock_open(read_data="some python content\nmore lines").return_value
            m.__iter__.return_value = iter(["some python content\nmore lines"])
            return m
        else:
            raise FileNotFoundError
    monkeypatch.setattr("builtins.open", open_side_effect)

    # Patch os.path.exists to True for dataset and research files
    monkeypatch.setattr("python_master_ai.os.path.exists", lambda path: True)

    # Patch get_latest_dataset_path to return a dummy dataset path
    monkeypatch.setattr(PythonMasterAI, "get_latest_dataset_path", lambda self, stage: os.path.join("data", stage, "20240101"))

    # Patch validate_research_data to always True
    monkeypatch.setattr(PythonMasterAI, "validate_research_data", lambda self, content, source: True)

    # Patch log_task_progress and log_research to just record calls
    monkeypatch.setattr(PythonMasterAI, "log_task_progress", MagicMock())
    monkeypatch.setattr(PythonMasterAI, "log_research", MagicMock())

    # Patch evaluate_source to always return added=True
    monkeypatch.setattr(PythonMasterAI, "evaluate_source", MagicMock(return_value={"added": True}))

    # Patch search_for_sources to return dummy sources
    monkeypatch.setattr(PythonMasterAI, "search_for_sources", MagicMock(return_value=[("src1", "http://example.com")]))
    # Patch initial_source_categories to a simple dict
    ai = PythonMasterAI()
    ai.initial_source_categories = {"cat1": ["search_query:foo", "https://docs.python.org/3/"]}
    ai.known_sources = {
        "http://example.com": {"name": "src1", "category_hint": "cat1", "url": "http://example.com"},
        "https://docs.python.org/3/": {"name": "docs", "category_hint": "cat1", "url": "https://docs.python.org/3/"},
    }
    ai.stage = "baby"
    ai.growth_tasks = ai.define_growth_tasks()
    ai.task_progress = {}
    ai.knowledge_gaps = ["foo"]
    return ai

def test_process_scraped_research_data_dynamic_and_initial(monkeypatch, ai):
    # Patch formulate_research_queries to return a query
    monkeypatch.setattr(PythonMasterAI, "formulate_research_queries", lambda self: ["foo"])
    ai.process_scraped_research_data("baby")
    # Should call search_for_sources and evaluate_source for dynamic queries
    assert ai.search_for_sources.called
    assert ai.evaluate_source.called
    # Should call log_task_progress for research and initial category
    assert ai.log_task_progress.call_count >= 2
    # Should call log_research for resolved queries
    assert ai.log_research.called

def test_process_scraped_research_data_no_queries(monkeypatch, ai):
    # Patch formulate_research_queries to return empty
    monkeypatch.setattr(PythonMasterAI, "formulate_research_queries", lambda self: [])
    ai.process_scraped_research_data("baby")
    # Should not call search_for_sources
    assert not ai.search_for_sources.called

def test_process_scraped_research_data_missing_dataset(monkeypatch, ai):
    # Patch get_latest_dataset_path to return None
    monkeypatch.setattr(PythonMasterAI, "get_latest_dataset_path", lambda self, stage: None)
    ai.known_sources = {"http://example.com": {"name": "src1", "category_hint": "cat1", "url": "http://example.com"}}
    ai.knowledge_gaps = ["foo"]
    ai.formulate_research_queries = lambda: ["foo"]
    ai.process_scraped_research_data("baby")
    # Should exit early and not call log_research
    assert not ai.log_research.called

def test_process_scraped_research_data_invalid_research_data(monkeypatch, ai):
    # Patch validate_research_data to return False
    monkeypatch.setattr(PythonMasterAI, "validate_research_data", lambda self, content, source: False)
    ai.formulate_research_queries = lambda: ["foo"]
    ai.process_scraped_research_data("baby")
    # Should not call log_research for invalid data
    assert not ai.log_research.called

def test_process_scraped_research_data_knowledge_gap_resolution(monkeypatch, ai):
    # Patch validate_research_data to return True
    monkeypatch.setattr(PythonMasterAI, "validate_research_data", lambda self, content, source: True)
    ai.knowledge_gaps = ["foo"]
    ai.formulate_research_queries = lambda: ["foo"]
    ai.process_scraped_research_data("baby")
    # Should resolve the knowledge gap
    assert ai.knowledge_gaps == []

def test_process_scraped_research_data_no_initial_categories(monkeypatch, ai):
    ai.initial_source_categories = {}
    ai.formulate_research_queries = lambda: []
    ai.process_scraped_research_data("baby")
    # Should log "No active research queries to process post-scraping."
    # (logger is patched, so just ensure no error)