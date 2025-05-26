import pytest
from unittest.mock import MagicMock, patch, mock_open
from scrapy.http import HtmlResponse, Response # Added Response for non-HTML
import os
import json
from datetime import datetime # For checking timestamp format
import requests # For requests.exceptions.RequestException

# Ensure the scrape_data module can be imported.
# This might require PYTHONPATH adjustments depending on execution context.
# For now, assume it's discoverable.
from scrape_data import PythonSpider, SaveToFilePipeline, _load_api_config_for_scraper, discover_urls_from_query, fetch_pypi_updates, API_CONFIG_FILE_PATH_SCRAPER

@pytest.fixture
def spider_instance(tmp_path):
    """Provides a PythonSpider instance for testing."""
    dummy_version_data_dir = str(tmp_path / "spider_output")
    os.makedirs(dummy_version_data_dir, exist_ok=True) # Ensure dir exists for file path logic

    spider = PythonSpider(
        source="test_source", # Default, can be overridden
        url="http://example.com/testpage", # Default
        version_data_dir=dummy_version_data_dir,
        stage="test_stage"
    )
    # Replace the logger with a MagicMock to inspect calls
    spider.logger = MagicMock()
    return spider

# --- Test Cases ---

# 1. Default Criteria Tests
def test_parse_default_criteria_pass_no_fallback(spider_instance):
    spider_instance.source = "default_source" # Ensure it uses default criteria
    html_content = "<html><body><p>This content is long enough to pass default criteria.</p></body></html>"
    mock_response = HtmlResponse(url=spider_instance.url, body=html_content, encoding='utf-8')

    results = list(spider_instance.parse(mock_response))
    item = results[0]

    assert item['content'] == "This content is long enough to pass default criteria."  # nosec B101
    assert item['parser_used'] == "Scrapy CSS Selector"  # nosec B101
    spider_instance.logger.info.assert_any_call(f"Successfully parsed {spider_instance.url} (source: '{spider_instance.source}') using Scrapy CSS selectors, meeting specific criteria.")
    # Check that no warning about fallback was logged
    assert not any("Falling back to BeautifulSoup" in c[0][0] for c in spider_instance.logger.warning.call_args_list)  # nosec B101

def test_parse_default_criteria_fail_min_length_fallback_bs_finds_nothing_meaningful(spider_instance):
    spider_instance.source = "default_source_short"
    # Scrapy p::text finds "Short." (fails min_length: 20 for default)
    # BeautifulSoup p::text also finds "Short." (which would also fail if checked by same criteria)
    html_content = "<html><body><p>Short.</p></body></html>"
    mock_response = HtmlResponse(url=spider_instance.url, body=html_content, encoding='utf-8')

    results = list(spider_instance.parse(mock_response))
    item = results[0]

    # Based on current spider logic: if BS content is empty or also fails, original Scrapy content is kept.
    assert item['content'] == "Short."  # nosec B101
    # And parser_used indicates that BS was attempted but its result wasn't used (or BS result was also "Short.")
    # Spider logic: if extracted_content_bs is empty, parser_used is NOT updated to "BeautifulSoup".
    # If extracted_content_bs is "Short.", it would be assigned, and parser_used would be "BeautifulSoup".
    # Let's assume BS also finds "Short."
    assert item['parser_used'] == "BeautifulSoup"  # nosec B101

    spider_instance.logger.warning.assert_any_call(f"Scrapy CSS selector parsing for '{spider_instance.source}' ({spider_instance.url}) failed source-specific content criteria. Falling back to BeautifulSoup.")
    # This debug log comes from _check_content_meaningfulness
    spider_instance.logger.debug.assert_any_call(f"Content from source '{spider_instance.source}' (length: 6) failed min_length check of 20. Criteria: {{'min_length': 20}}")


def test_parse_default_criteria_fail_min_length_fallback_bs_finds_different_content(spider_instance):
    spider_instance.source = "default_source_complex"
    # Scrapy p::text finds "Short." (fails min_length: 20)
    # BS p::text finds "Short. Also another paragraph." (which could pass if combined, or if BS logic is different)
    # Current BS logic in spider: soup.find_all('p') then join.
    html_content = "<html><body><p>Short.</p> <article><p>Also another paragraph.</p></article></body></html>"
    mock_response = HtmlResponse(url=spider_instance.url, body=html_content, encoding='utf-8')

    results = list(spider_instance.parse(mock_response))
    item = results[0]

    assert item['content'] == "Short. Also another paragraph." # BS combines text from all <p>  # nosec B101
    assert item['parser_used'] == "BeautifulSoup"  # nosec B101
    spider_instance.logger.warning.assert_any_call(f"Scrapy CSS selector parsing for '{spider_instance.source}' ({spider_instance.url}) failed source-specific content criteria. Falling back to BeautifulSoup.")
    spider_instance.logger.info.assert_any_call(f"Successfully parsed {spider_instance.url} (source: '{spider_instance.source}') using BeautifulSoup after Scrapy CSS selector fallback.")


# 2. Source-Specific Criteria Tests ('study_guides')
def test_parse_study_guides_criteria_pass_no_fallback(spider_instance):
    spider_instance.source = "study_guides"
    html_content = "<html><body><p>This is a study guide that is definitely longer than fifty characters to meet the specific criteria.</p></body></html>"
    mock_response = HtmlResponse(url=spider_instance.url, body=html_content, encoding='utf-8')

    results = list(spider_instance.parse(mock_response))
    item = results[0]

    assert item['content'] == "This is a study guide that is definitely longer than fifty characters to meet the specific criteria."  # nosec B101
    assert item['parser_used'] == "Scrapy CSS Selector"  # nosec B101
    spider_instance.logger.info.assert_any_call(f"Successfully parsed {spider_instance.url} (source: '{spider_instance.source}') using Scrapy CSS selectors, meeting specific criteria.")

def test_parse_study_guides_fail_min_length_fallback_succeeds(spider_instance):
    spider_instance.source = "study_guides"
    # Scrapy p::text finds content < 50 chars for study_guides.
    # BS p::text finds the same short content.
    html_content = "<html><body><p>Short study guide.</p></body></html>" # Length 19
    mock_response = HtmlResponse(url=spider_instance.url, body=html_content, encoding='utf-8')

    expected_criteria_sg = spider_instance.SOURCE_SPECIFIC_CONTENT_CRITERIA["study_guides"]

    results = list(spider_instance.parse(mock_response))
    item = results[0]

    assert item['content'] == "Short study guide." # BS found same content  # nosec B101
    assert item['parser_used'] == "BeautifulSoup" # BS was used  # nosec B101
    spider_instance.logger.warning.assert_any_call(f"Scrapy CSS selector parsing for '{spider_instance.source}' ({spider_instance.url}) failed source-specific content criteria. Falling back to BeautifulSoup.")
    spider_instance.logger.debug.assert_any_call(f"Content from source '{spider_instance.source}' (length: 19) failed min_length check of {expected_criteria_sg['min_length']}. Criteria: {expected_criteria_sg}")

def test_parse_study_guides_fail_forbidden_keyword_fallback_succeeds(spider_instance):
    spider_instance.source = "study_guides"
    forbidden_keyword = "placeholder content"
    html_content = f"<html><body><p>This is some {forbidden_keyword} that is long enough but should be rejected due to keywords.</p></body></html>"
    mock_response = HtmlResponse(url=spider_instance.url, body=html_content, encoding='utf-8')

    expected_criteria_sg = spider_instance.SOURCE_SPECIFIC_CONTENT_CRITERIA["study_guides"]

    results = list(spider_instance.parse(mock_response))
    item = results[0]

    # BS will also find the forbidden keyword, so its content is used.
    assert forbidden_keyword in item['content']  # nosec B101
    assert item['parser_used'] == "BeautifulSoup"  # nosec B101
    spider_instance.logger.warning.assert_any_call(f"Scrapy CSS selector parsing for '{spider_instance.source}' ({spider_instance.url}) failed source-specific content criteria. Falling back to BeautifulSoup.")
    spider_instance.logger.debug.assert_any_call(f"Content from source '{spider_instance.source}' failed must_not_contain_any check. Found: ['{forbidden_keyword}']. Criteria: {expected_criteria_sg}")

# 3. Non-HTML Content (GitHub JSON)
def test_parse_github_json_no_html_fallback(spider_instance):
    spider_instance.source = "github_beginner" # Or any source handled by the github logic
    json_data = {
        "items": [
            {"description": "Description 1 for repo."},
            {"description": "Description 2 another repo."}
        ]
    }
    # Scrapy's JsonResponse is not directly available, use generic Response
    # and spider's logic should call response.json()
    mock_response = Response(url=spider_instance.url, body=b'{"items": [{"description": "Description 1 for repo."}, {"description": "Description 2 another repo."}]}', headers={'Content-Type': 'application/json'})

    # Mock response.json() if Response doesn't automatically provide it based on headers
    # However, Scrapy's actual Response object used in spiders typically handles this.
    # For this test, let's assume the spider calls response.json() which works on a Response object with JSON body.
    # If `response.json()` method is part of the base `Response` class and works with `_body`, this is fine.
    # Let's add a mock for `response.json()` to be safe and explicit.
    mock_response.json = MagicMock(return_value=json_data)


    results = list(spider_instance.parse(mock_response))
    item = results[0]

    assert item['content'] == "Description 1 for repo.\nDescription 2 another repo."  # nosec B101
    assert item['parser_used'] == "JSON API"  # nosec B101
    # Ensure no HTML parsing logs are present for this source
    assert not any("CSS selector" in c[0][0] for c in spider_instance.logger.debug.call_args_list)  # nosec B101
    assert not any("BeautifulSoup" in c[0][0] for c in spider_instance.logger.warning.call_args_list)  # nosec B101


# 4. Empty/Error Responses
def test_parse_empty_response_text_triggers_fallback_yields_empty(spider_instance):
    spider_instance.source = "empty_source_default"
    html_content = "<html><body></body></html>" # No <p> tags
    mock_response = HtmlResponse(url=spider_instance.url, body=html_content, encoding='utf-8')

    results = list(spider_instance.parse(mock_response))
    item = results[0]

    assert item['content'] == ""  # nosec B101
    # Scrapy finds nothing, fails criteria. BS finds nothing.
    # Original Scrapy content (empty) is used. parser_used is "Scrapy CSS Selector"
    # because BS didn't provide alternative content.
    assert item['parser_used'] == "Scrapy CSS Selector"  # nosec B101
    spider_instance.logger.warning.assert_any_call(f"Scrapy CSS selector parsing for '{spider_instance.source}' ({spider_instance.url}) failed source-specific content criteria. Falling back to BeautifulSoup.")
    spider_instance.logger.warning.assert_any_call(f"BeautifulSoup parsing also yielded no meaningful content for {spider_instance.url} (source: '{spider_instance.source}'). Original Scrapy content (if any) will be used or content will be empty.")


@patch('scrape_data.BeautifulSoup') # Patch BeautifulSoup where it's imported in scrape_data
def test_parse_exception_during_bs_fallback(mock_bs_constructor, spider_instance):
    spider_instance.source = "bs_exception_source"
    html_content = "<html><body><p>Short</p></body></html>" # To trigger fallback
    mock_response = HtmlResponse(url=spider_instance.url, body=html_content, encoding='utf-8')

    # Make BeautifulSoup constructor or a method call on its instance raise an error
    mock_bs_constructor.side_effect = Exception("BS Critical Error")

    results = list(spider_instance.parse(mock_response))
    item = results[0]

    assert "Error processing bs_exception_source. See logs." in item['content']  # nosec B101
    # The parser_used at time of error would be after Scrapy failed, and BS was being attempted.  # nosec B101
    # The spider sets parser_used to "Scrapy CSS Selector" initially. It's not updated before BS call.
    # So, if BS itself errors out, parser_used would still be "Scrapy CSS Selector".
    assert item['parser_used'] == "Scrapy CSS Selector"  # nosec B101

    spider_instance.logger.error.assert_called_once()
    args, kwargs = spider_instance.logger.error.call_args
    assert "Error parsing" in args[0]
    assert "BS Critical Error" in str(kwargs.get('exc_info'))


def test_parse_response_css_raises_exception(spider_instance):
    spider_instance.source = "css_error_source"
    html_content = "<html><body><p>Some content</p></body></html>"
    mock_response = HtmlResponse(url=spider_instance.url, body=html_content, encoding='utf-8')

    # Mock response.css() to raise an error
    mock_response.css = MagicMock(side_effect=Exception("CSS Selector Engine Failed"))

    results = list(spider_instance.parse(mock_response))
    item = results[0]

    assert "Error processing css_error_source. See logs." in item['content']  # nosec B101
    assert item['parser_used'] == "Scrapy CSS Selector" # Initial value before error  # nosec B101
    spider_instance.logger.error.assert_called_once()
    args, kwargs = spider_instance.logger.error.call_args
    assert "Error parsing" in args[0]  # nosec B101
    assert "CSS Selector Engine Failed" in str(kwargs.get('exc_info'))

# --- Section I: Tests for SaveToFilePipeline ---

@pytest.fixture
def pipeline_instance():
    return SaveToFilePipeline()

def test_pipeline_saves_metadata_for_text_item(pipeline_instance, spider_instance, tmp_path):
    test_file_path = tmp_path / "test_output.txt"
    item = {
        "file": str(test_file_path),
        "content": "This is test content.",
        "source": "test_source_metadata",
        "source_url": "http://example.com/metadata-test",
        "parser_used": "Scrapy CSS Selector"
    }

    pipeline_instance.process_item(item, spider_instance)

    assert test_file_path.exists()  # nosec B101
    with open(test_file_path, "r") as f:
        assert "This is test content." in f.read()  # nosec B101

    meta_file_path = tmp_path / "test_output.txt.meta.json"
    assert meta_file_path.exists()  # nosec B101
    with open(meta_file_path, "r") as f:
        meta_data = json.load(f)

    assert meta_data["source_url"] == "http://example.com/metadata-test"  # nosec B101
    assert "scraped_timestamp" in meta_data  # nosec B101
    try:
        datetime.fromisoformat(meta_data["scraped_timestamp"])
    except ValueError:
        pytest.fail("scraped_timestamp is not a valid ISO format datetime string")

def test_pipeline_archives_raw_content(pipeline_instance, spider_instance, tmp_path):
    archive_item = {
        "source_url": "http://example.com/path/to/page.html",
        "source_name": "example_archive", # Used by spider, not directly by pipeline for path
        "version_data_dir": str(tmp_path),
        "is_archive_item": True,
        "raw_response_body": b"<html>Test archive content</html>",
        "response_headers": {"Content-Type": "text/html"}
    }

    pipeline_instance.process_item(archive_item, spider_instance)

    expected_archive_path = tmp_path / "archived_sites" / "example_com" / "path" / "to" / "page.html"
    assert expected_archive_path.exists()  # nosec B101
    with open(expected_archive_path, "rb") as f:
        assert f.read() == b"<html>Test archive content</html>"  # nosec B101
    spider_instance.logger.info.assert_any_call(f"Archived raw content from http://example.com/path/to/page.html to {expected_archive_path}")


def test_pipeline_archives_raw_content_no_extension_infer_from_content_type(pipeline_instance, spider_instance, tmp_path):
    archive_item = {
        "source_url": "http://example.com/api/data",
        "source_name": "example_api",
        "version_data_dir": str(tmp_path),
        "is_archive_item": True,
        "raw_response_body": b'{"key": "value"}',
        "response_headers": {"Content-Type": "application/json; charset=utf-8"} # Include charset
    }
    pipeline_instance.process_item(archive_item, spider_instance)
    expected_archive_path = tmp_path / "archived_sites" / "example_com" / "api" / "data.json" # data part becomes filename
    assert expected_archive_path.exists()  # nosec B101
    with open(expected_archive_path, "rb") as f:
        assert f.read() == b'{"key": "value"}'  # nosec B101

def test_pipeline_archives_content_type_missing_uses_default_name(pipeline_instance, spider_instance, tmp_path):
    archive_item = {
        "source_url": "http://example.com/api/endpoint",
        "source_name": "example_endpoint",
        "version_data_dir": str(tmp_path),
        "is_archive_item": True,
        "raw_response_body": b"raw data",
        "response_headers": {} # No Content-Type
    }
    pipeline_instance.process_item(archive_item, spider_instance)
    # The logic is: path is /api/endpoint. 'endpoint' has no extension.
    # Content-Type is missing. So it defaults to 'index.dat' under the 'endpoint' directory.
    expected_archive_path = tmp_path / "archived_sites" / "example_com" / "api" / "endpoint" / "index.dat"
    assert expected_archive_path.exists()  # nosec B101
    with open(expected_archive_path, "rb") as f:
        assert f.read() == b"raw data"  # nosec B101

# --- Section II: Tests for API-based Source Discovery ---

# Part 1: _load_api_config_for_scraper
@patch('scrape_data.os.path.exists')
@patch('scrape_data.open', new_callable=mock_open)
def test_load_api_config_success(mock_open_file, mock_exists):
    mock_exists.return_value = True
    mock_open_file.return_value.read.return_value = '{"api_keys": {"google_custom_search_api_key": "test_key"}}'

    config = _load_api_config_for_scraper()

    assert config == {"api_keys": {"google_custom_search_api_key": "test_key"}}  # nosec B101
    mock_open_file.assert_called_once_with(API_CONFIG_FILE_PATH_SCRAPER, "r")

@patch('scrape_data.os.path.exists')
def test_load_api_config_file_not_found(mock_exists, caplog):
    mock_exists.return_value = False

    config = _load_api_config_for_scraper()

    assert config == {}  # nosec B101
    assert f"API configuration file {API_CONFIG_FILE_PATH_SCRAPER} not found" in caplog.text  # nosec B101

@patch('scrape_data.os.path.exists')
@patch('scrape_data.open', new_callable=mock_open)
def test_load_api_config_invalid_json(mock_open_file, mock_exists, caplog):
    mock_exists.return_value = True
    mock_open_file.return_value.read.return_value = '{"bad json'

    config = _load_api_config_for_scraper()

    assert config == {}  # nosec B101
    assert f"Error reading or parsing API config file {API_CONFIG_FILE_PATH_SCRAPER}" in caplog.text  # nosec B101


# Part 2: discover_urls_from_query
@pytest.fixture
def api_config_google_only():
    return {
        "api_keys": {"google_custom_search_api_key": "fake_google_key", "google_custom_search_cx_id": "fake_cx_id"},
        "api_endpoints": {"google_custom_search": "https://fakeapi.google.com/customsearch/v1"},
        "search_parameters": {"google_search_default_num_results": "1"}
    }

@pytest.fixture
def api_config_github_only():
    return {
        "api_keys": {"github_pat": "fake_github_pat"},
        "api_endpoints": {"github_search_repositories": "https://fakeapi.github.com/search/repositories"},
        "search_parameters": {"github_repo_default_query_params": "sort=stars&order=desc"}
    }

@patch('scrape_data.requests.get')
def test_discover_urls_google_success(mock_requests_get, api_config_google_only):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"items": [{"title": "Test Google Result", "link": "http://google-result.com/test"}]}
    mock_requests_get.return_value = mock_response

    discovered = discover_urls_from_query("test query", api_config_google_only)

    expected_url = "https://fakeapi.google.com/customsearch/v1?key=fake_google_key&cx=fake_cx_id&q=test query&num=1"
    mock_requests_get.assert_called_once_with(expected_url, timeout=10)
    assert len(discovered) == 1  # nosec B101
    source_name, url = discovered[0]
    assert url == "http://google-result.com/test"  # nosec B101
    assert "web_discovered_google_result_com_Test_Google_Result" in source_name  # nosec B101


@patch('scrape_data.requests.get')
def test_discover_urls_github_success(mock_requests_get, api_config_github_only):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"items": [{"full_name": "user/repo", "html_url": "http://github.com/user/repo"}]}
    mock_requests_get.return_value = mock_response

    discovered = discover_urls_from_query("test query", api_config_github_only)

    expected_url = "https://fakeapi.github.com/search/repositories?q=test query python&sort=stars&order=desc"
    mock_requests_get.assert_called_once_with(expected_url, headers={"Authorization": "token fake_github_pat", "Accept": "application/vnd.github.v3+json"}, timeout=10)
    assert len(discovered) == 1  # nosec B101
    source_name, url = discovered[0]
    assert url == "http://github.com/user/repo"  # nosec B101
    assert source_name == "github_discovered_user_repo"  # nosec B101

@patch('scrape_data.requests.get')
def test_discover_urls_api_error(mock_requests_get, api_config_google_only, caplog):
    mock_requests_get.side_effect = requests.exceptions.RequestException("API down")

    discovered = discover_urls_from_query("test query", api_config_google_only)

    assert discovered == []  # nosec B101
    assert "Google Custom Search API failed for query 'test query': API down" in caplog.text  # nosec B101

def test_discover_urls_no_config(caplog): # Removed spider_instance as it's not used
    discovered = discover_urls_from_query("test query", {})
    assert discovered == []  # nosec B101
    assert "API config is empty. Cannot perform dynamic discovery." in caplog.text  # nosec B101

# --- Section III: Tests for fetch_pypi_updates ---

@patch('scrape_data.requests.get')
def test_fetch_pypi_saves_raw_json(mock_requests_get, tmp_path, caplog):
    mock_response_json = {"info": {"summary": "Test summary", "description": "Test Readme", "version": "1.0"}}

    mock_api_response = MagicMock()
    mock_api_response.status_code = 200
    mock_api_response.json.return_value = mock_response_json
    mock_requests_get.return_value = mock_api_response

    package_name = "testpackage"
    result = fetch_pypi_updates(package_name, str(tmp_path))

    raw_json_filename = f"pypi_package_{package_name}_raw.json"
    expected_raw_json_path = tmp_path / raw_json_filename

    assert expected_raw_json_path.exists()  # nosec B101
    with open(expected_raw_json_path, "r") as f:
        saved_data = json.load(f)
    assert saved_data == mock_response_json  # nosec B101

    assert result["raw_json_file_saved"] == raw_json_filename  # nosec B101
    assert result["summary_file_saved"] == f"pypi_package_{package_name}.txt" # Check other keys too  # nosec B101
    assert result["readme_file_saved"] == f"pypi_package_{package_name}_README.txt" # Default ext if no content type for desc  # nosec B101

@patch('scrape_data.requests.get')
@patch('scrape_data.json.dump') # Patch json.dump directly for this test
def test_fetch_pypi_raw_json_save_error(mock_json_dump, mock_requests_get, tmp_path, caplog):
    mock_response_json = {"info": {"summary": "Test summary", "description": "Test Readme", "version": "1.0"}}

    mock_api_response = MagicMock()
    mock_api_response.status_code = 200
    mock_api_response.json.return_value = mock_response_json
    mock_requests_get.return_value = mock_api_response

    # Make json.dump raise an error when attempting to write the raw json file
    # This is a bit broad, but simpler than patching 'open' for just one write.
    # We rely on the call order: summary, then raw_json, then readme.
    # The first call to json.dump will be for the raw json file.
    mock_json_dump.side_effect = IOError("Disk full")

    package_name = "testpackage_json_fail"
    result = fetch_pypi_updates(package_name, str(tmp_path))

    assert result["raw_json_file_saved"] is None  # nosec B101
    assert "Error saving raw PyPI JSON" in caplog.text  # nosec B101
    assert "Disk full" in caplog.text  # nosec B101
    # Summary should still be saved as it's written before raw JSON attempt.
    # However, json.dump is globally patched. This test needs refinement.
    # For simplicity, let's assume the error is specific to raw json.
    # A better patch would be:
    # with patch('builtins.open', new_callable=mock_open) as m_open:
    #     m_open.side_effect = [mock_open().return_value, # for summary
    #                           IOError("Disk full"),     # for raw json
    #                           mock_open().return_value] # for readme
    # This is complex to set up with multiple open calls.
    # The current patch on json.dump will affect all json.dump calls in the function.
    # Given the current structure of fetch_pypi_updates, the raw JSON is saved first.
    # The provided snippet saves raw JSON *after* response.json() but *before* summary/readme.
    # Let's re-verify the snippet: "Save the raw JSON response" is right after data = response.json().
    # So if json.dump for raw json fails, summary/readme saving might not occur or also fail.
    # The test prompt's snippet for fetch_pypi_updates has raw JSON saving *after* response.json().
    # Let's adjust the expectation based on current `fetch_pypi_updates` structure from previous turn.
    # If `json.dump` is the first `json.dump` in the try block for raw json, this test is okay.

    # Re-checking the provided snippet:
    # data = response.json()
    # raw_json_filename = ...
    # try: json.dump(data, ...) <--- This is the one we are targetting
    # except: raw_json_filename = None
    # summary_content = ...
    # with open(output_filepath, "a", encoding="utf-8") as f: f.write(summary_content + "\n") <-- this is text write.
    # So, json.dump is only for the raw JSON. The summary and readme are plain text writes.
    # Thus, patching json.dump is specific enough.

    # Summary and readme are *not* saved using json.dump, they use f.write.
    # So this patch is fine.
    assert result["summary_file_saved"] is not None # Summary should be saved as it's a text write.  # nosec B101
    assert result["readme_file_saved"] is not None # Readme also a text write.  # nosec B101
