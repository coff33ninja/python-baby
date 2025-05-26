import pytest
from unittest.mock import patch, MagicMock, mock_open
from python_master_ai import PythonMasterAI
import os
from collections import defaultdict


@pytest.fixture
def ai(monkeypatch, tmp_path):
    # Patch logger to avoid real logging
    monkeypatch.setattr("python_master_ai.logger", MagicMock())
    monkeypatch.setattr("python_master_ai.os.path.join", os.path.join)
    monkeypatch.setattr("python_master_ai.os.path.isdir", lambda path: True)

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
    monkeypatch.setattr("python_master_ai.os.path.exists", lambda path: True)
    monkeypatch.setattr(
        PythonMasterAI,
        "get_latest_dataset_path",
        lambda self, stage: os.path.join("data", stage, "20240101"),
    )
    monkeypatch.setattr(
        PythonMasterAI, "validate_research_data", lambda self, content, source: True
    )
    monkeypatch.setattr(PythonMasterAI, "log_task_progress", MagicMock())
    monkeypatch.setattr(PythonMasterAI, "log_research", MagicMock())
    monkeypatch.setattr(
        PythonMasterAI, "evaluate_source", MagicMock(return_value={"added": True})
    )
    monkeypatch.setattr(
        PythonMasterAI,
        "search_for_sources",
        MagicMock(return_value=[("src1", "http://example.com")]),
    )

    ai = PythonMasterAI()
    ai.initial_source_categories = {
        "cat1": ["search_query:foo", "https://docs.python.org/3/"]
    }
    ai.known_sources = {
        "http://example.com": {
            "name": "src1",
            "category_hint": "cat1",
            "url": "http://example.com",
        },
        "https://docs.python.org/3/": {
            "name": "docs",
            "category_hint": "cat1",
            "url": "https://docs.python.org/3/",
        },
    }
    ai.stage = "baby"
    ai.growth_tasks = ai.define_growth_tasks()
    ai.task_progress = defaultdict(float)  # Standardized to defaultdict(float)
    ai.knowledge_gaps = ["foo"]
    return ai


# --- Existing tests (kept as is unless modified below) ---


def test_process_scraped_research_data_logs_progress_for_initial_categories(
    monkeypatch, ai
):
    ai.initial_source_categories = {
        "cat1": ["https://docs.python.org/3/"],
        "cat2": ["https://pypi.org/project/pytest/"],
    }
    ai.formulate_research_queries = lambda: []
    ai.process_scraped_research_data("baby")
    assert ai.log_task_progress.call_count >= 2


def test_process_scraped_research_data_skips_empty_string_seed(monkeypatch, ai):
    ai.initial_source_categories = {"cat1": [""]}
    ai.formulate_research_queries = lambda: []
    with patch("python_master_ai.logger") as logger_mock:
        ai.process_scraped_research_data("baby")
        logger_mock.warning.assert_any_call(
            "Seed item '' for category 'cat1' is not a recognized URL, search_query, or PyPI package. Skipping."
        )


def test_process_scraped_research_data_handles_mixed_valid_and_invalid_seeds(
    monkeypatch, ai
):
    ai.initial_source_categories = {
        "cat1": ["https://docs.python.org/3/", 123, None, ""]
    }
    ai.formulate_research_queries = lambda: []
    with patch("python_master_ai.logger") as logger_mock, patch.object(
        ai, "evaluate_source", wraps=ai.evaluate_source
    ) as eval_mock:
        ai.process_scraped_research_data("baby")
        assert eval_mock.call_count == 1  # Only valid URL processed
        logger_mock.warning.assert_any_call(
            "Seed item '123' for category 'cat1' is not a recognized URL, search_query, or PyPI package. Skipping."
        )
        logger_mock.warning.assert_any_call(
            "Seed item 'None' for category 'cat1' is not a recognized URL, search_query, or PyPI package. Skipping."
        )
        logger_mock.warning.assert_any_call(
            "Seed item '' for category 'cat1' is not a recognized URL, search_query, or PyPI package. Skipping."
        )


def test_process_scraped_research_data_logs_no_active_queries(monkeypatch, ai):
    # Consolidated test for empty initial categories and queries
    ai.initial_source_categories = {}
    ai.formulate_research_queries = lambda: []
    with patch("python_master_ai.logger") as logger_mock:
        ai.process_scraped_research_data("baby")
        logger_mock.info.assert_any_call(
            "No active research queries to process post-scraping."
        )


def test_process_scraped_research_data_logs_progress_for_each_unique_query(
    monkeypatch, ai
):
    ai.knowledge_gaps = ["foo", "bar", "baz"]
    ai.formulate_research_queries = lambda: ["foo", "bar", "baz"]
    ai.process_scraped_research_data("baby")
    assert ai.log_task_progress.call_count >= 3


def test_process_scraped_research_data_skips_query_if_not_in_knowledge_gaps(
    monkeypatch, ai
):
    ai.knowledge_gaps = []
    ai.formulate_research_queries = lambda: ["foo"]
    ai.process_scraped_research_data("baby")
    assert not ai.log_research.called


def test_process_scraped_research_data_handles_all_invalid_initial_seeds(
    monkeypatch, ai
):
    ai.initial_source_categories = {"cat1": [None, 123, ""]}
    ai.formulate_research_queries = lambda: []
    with patch("python_master_ai.logger") as logger_mock:
        ai.process_scraped_research_data("baby")
        logger_mock.warning.assert_any_call(
            "Seed item 'None' for category 'cat1' is not a recognized URL, search_query, or PyPI package. Skipping."
        )
        logger_mock.warning.assert_any_call(
            "Seed item '123' for category 'cat1' is not a recognized URL, search_query, or PyPI package. Skipping."
        )
        logger_mock.warning.assert_any_call(
            "Seed item '' for category 'cat1' is not a recognized URL, search_query, or PyPI package. Skipping."
        )


def test_process_scraped_research_data_handles_duplicate_queries_and_initial(
    monkeypatch, ai
):
    ai.knowledge_gaps = ["foo", "bar"]
    ai.initial_source_categories = {
        "cat1": ["https://docs.python.org/3/", "https://docs.python.org/3/"]
    }
    ai.formulate_research_queries = lambda: ["foo", "foo", "bar"]
    with patch.object(
        ai, "search_for_sources", wraps=ai.search_for_sources
    ) as search_mock:
        ai.process_scraped_research_data("baby")
        assert search_mock.call_count == 2


def test_process_scraped_research_data_handles_file_not_found_for_initial(
    monkeypatch, ai
):
    monkeypatch.setattr("python_master_ai.os.path.exists", lambda path: False)
    ai.initial_source_categories = {"cat1": ["https://docs.python.org/3/"]}
    ai.formulate_research_queries = lambda: []
    ai.process_scraped_research_data("baby")
    assert not ai.log_research.called


def test_process_scraped_research_data_handles_file_not_found_for_query(
    monkeypatch, ai
):
    monkeypatch.setattr("python_master_ai.os.path.exists", lambda path: False)
    ai.knowledge_gaps = ["foo"]
    ai.formulate_research_queries = lambda: ["foo"]
    ai.process_scraped_research_data("baby")
    assert not ai.log_research.called


def test_process_scraped_research_data_duplicate_search_queries(monkeypatch, ai):
    ai.formulate_research_queries = lambda: ["foo", "foo"]
    with patch.object(
        ai, "search_for_sources", wraps=ai.search_for_sources
    ) as search_mock:
        ai.process_scraped_research_data("baby")
        assert search_mock.call_count == 1


def test_process_scraped_research_data_partial_file_not_found(monkeypatch, ai):
    def exists_side_effect(path):
        return "foo" in path

    monkeypatch.setattr("python_master_ai.os.path.exists", exists_side_effect)
    ai.knowledge_gaps = ["foo", "bar"]
    ai.formulate_research_queries = lambda: ["foo", "bar"]
    ai.process_scraped_research_data("baby")
    assert ai.knowledge_gaps == ["bar"]


def test_process_scraped_research_data_handles_large_number_of_queries(monkeypatch, ai):
    queries = [f"gap_{i}" for i in range(20)]
    ai.knowledge_gaps = queries.copy()
    ai.formulate_research_queries = lambda: queries
    ai.process_scraped_research_data("baby")
    assert ai.knowledge_gaps == []


def test_process_scraped_research_data_handles_empty_txt_file(monkeypatch, ai):
    def open_side_effect(file, mode="r", *args, **kwargs):
        if file.endswith(".txt"):
            m = mock_open(read_data="").return_value
            m.__iter__.return_value = iter([""])
            return m
        else:
            raise FileNotFoundError

    monkeypatch.setattr("builtins.open", open_side_effect)
    ai.formulate_research_queries = lambda: ["foo"]
    ai.process_scraped_research_data("baby")
    assert ai.log_research.called


def test_process_scraped_research_data_skips_duplicate_initial_urls(monkeypatch, ai):
    ai.initial_source_categories = {
        "cat1": ["https://docs.python.org/3/", "https://docs.python.org/3/"]
    }
    ai.formulate_research_queries = lambda: []
    with patch.object(ai, "evaluate_source", wraps=ai.evaluate_source) as eval_mock:
        ai.process_scraped_research_data("baby")
        assert eval_mock.call_count == 1


def test_process_scraped_research_data_skips_unrecognized_seed(monkeypatch, ai):
    ai.initial_source_categories = {"cat1": ["not_a_url_or_query"]}
    ai.formulate_research_queries = lambda: []
    with patch("python_master_ai.logger") as logger_mock:
        ai.process_scraped_research_data("baby")
        logger_mock.warning.assert_any_call(
            "Seed item 'not_a_url_or_query' for category 'cat1' is not a recognized URL, search_query, or PyPI package. Skipping."
        )


def test_process_scraped_research_data_handles_pypi_seed(monkeypatch, ai):
    ai.initial_source_categories = {"pypi_cat": ["requests"]}
    ai.formulate_research_queries = lambda: []
    with patch.object(ai, "evaluate_source", wraps=ai.evaluate_source) as eval_mock:
        ai.process_scraped_research_data("baby")
        assert eval_mock.called


def test_process_scraped_research_data_dynamic_and_initial(monkeypatch, ai):
    monkeypatch.setattr(
        PythonMasterAI, "formulate_research_queries", lambda self: ["foo"]
    )
    ai.process_scraped_research_data("baby")
    assert ai.search_for_sources.called
    assert ai.evaluate_source.called
    assert ai.log_task_progress.call_count >= 2
    assert ai.log_research.called


def test_process_scraped_research_data_no_queries(monkeypatch, ai):
    monkeypatch.setattr(PythonMasterAI, "formulate_research_queries", lambda self: [])
    ai.process_scraped_research_data("baby")
    assert not ai.search_for_sources.called


def test_process_scraped_research_data_missing_dataset(monkeypatch, ai):
    monkeypatch.setattr(
        PythonMasterAI, "get_latest_dataset_path", lambda self, stage: None
    )
    ai.known_sources = {
        "http://example.com": {
            "name": "src1",
            "category_hint": "cat1",
            "url": "http://example.com",
        }
    }
    ai.knowledge_gaps = ["foo"]
    ai.formulate_research_queries = lambda: ["foo"]
    ai.process_scraped_research_data("baby")
    assert not ai.log_research.called


def test_process_scraped_research_data_invalid_research_data(monkeypatch, ai):
    monkeypatch.setattr(
        PythonMasterAI, "validate_research_data", lambda self, content, source: False
    )
    ai.formulate_research_queries = lambda: ["foo"]
    ai.process_scraped_research_data("baby")
    assert not ai.log_research.called


def test_process_scraped_research_data_knowledge_gap_resolution(monkeypatch, ai):
    monkeypatch.setattr(
        PythonMasterAI, "validate_research_data", lambda self, content, source: True
    )
    ai.knowledge_gaps = ["foo"]
    ai.formulate_research_queries = lambda: ["foo"]
    ai.process_scraped_research_data("baby")
    assert ai.knowledge_gaps == []


def test_process_scraped_research_data_initial_category_and_query(monkeypatch, ai):
    ai.initial_source_categories = {"cat1": ["https://docs.python.org/3/"]}
    ai.knowledge_gaps = ["foo"]
    ai.formulate_research_queries = lambda: ["foo"]
    ai.process_scraped_research_data("baby")
    assert ai.log_task_progress.call_count >= 2
    assert ai.log_research.called


def test_process_scraped_research_data_updates_task_progress(monkeypatch, ai):
    # New test to verify task_progress updates
    ai.knowledge_gaps = ["foo"]
    ai.initial_source_categories = {"cat1": ["https://docs.python.org/3/"]}
    ai.formulate_research_queries = lambda: ["foo"]
    ai.process_scraped_research_data("baby")
    assert (
        ai.task_progress["research"] > 0
    )  # Verify progress is updated for research task
    assert (
        ai.task_progress["cat1"] > 0
    )  # Verify progress is updated for initial category
