import pytest
from unittest.mock import MagicMock, patch, call
from scrapy.http import HtmlResponse, Response # Added Response for non-HTML
import os

# Ensure the scrape_data module can be imported.
# This might require PYTHONPATH adjustments depending on execution context.
# For now, assume it's discoverable.
from scrape_data import PythonSpider

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

    assert item['content'] == "This content is long enough to pass default criteria."
    assert item['parser_used'] == "Scrapy CSS Selector"
    spider_instance.logger.info.assert_any_call(f"Successfully parsed {spider_instance.url} (source: '{spider_instance.source}') using Scrapy CSS selectors, meeting specific criteria.")
    # Check that no warning about fallback was logged
    assert not any("Falling back to BeautifulSoup" in c[0][0] for c in spider_instance.logger.warning.call_args_list)

def test_parse_default_criteria_fail_min_length_fallback_bs_finds_nothing_meaningful(spider_instance):
    spider_instance.source = "default_source_short"
    # Scrapy p::text finds "Short." (fails min_length: 20 for default)
    # BeautifulSoup p::text also finds "Short." (which would also fail if checked by same criteria)
    html_content = "<html><body><p>Short.</p></body></html>"
    mock_response = HtmlResponse(url=spider_instance.url, body=html_content, encoding='utf-8')

    results = list(spider_instance.parse(mock_response))
    item = results[0]
    
    # Based on current spider logic: if BS content is empty or also fails, original Scrapy content is kept.
    assert item['content'] == "Short." 
    # And parser_used indicates that BS was attempted but its result wasn't used (or BS result was also "Short.")
    # Spider logic: if extracted_content_bs is empty, parser_used is NOT updated to "BeautifulSoup".
    # If extracted_content_bs is "Short.", it would be assigned, and parser_used would be "BeautifulSoup".
    # Let's assume BS also finds "Short."
    assert item['parser_used'] == "BeautifulSoup" 

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

    assert item['content'] == "Short. Also another paragraph." # BS combines text from all <p>
    assert item['parser_used'] == "BeautifulSoup"
    spider_instance.logger.warning.assert_any_call(f"Scrapy CSS selector parsing for '{spider_instance.source}' ({spider_instance.url}) failed source-specific content criteria. Falling back to BeautifulSoup.")
    spider_instance.logger.info.assert_any_call(f"Successfully parsed {spider_instance.url} (source: '{spider_instance.source}') using BeautifulSoup after Scrapy CSS selector fallback.")


# 2. Source-Specific Criteria Tests ('study_guides')
def test_parse_study_guides_criteria_pass_no_fallback(spider_instance):
    spider_instance.source = "study_guides"
    html_content = "<html><body><p>This is a study guide that is definitely longer than fifty characters to meet the specific criteria.</p></body></html>"
    mock_response = HtmlResponse(url=spider_instance.url, body=html_content, encoding='utf-8')

    results = list(spider_instance.parse(mock_response))
    item = results[0]

    assert item['content'] == "This is a study guide that is definitely longer than fifty characters to meet the specific criteria."
    assert item['parser_used'] == "Scrapy CSS Selector"
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

    assert item['content'] == "Short study guide." # BS found same content
    assert item['parser_used'] == "BeautifulSoup" # BS was used
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
    assert forbidden_keyword in item['content']
    assert item['parser_used'] == "BeautifulSoup"
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

    assert item['content'] == "Description 1 for repo.\nDescription 2 another repo."
    assert item['parser_used'] == "JSON API"
    # Ensure no HTML parsing logs are present for this source
    assert not any("CSS selector" in c[0][0] for c in spider_instance.logger.debug.call_args_list)
    assert not any("BeautifulSoup" in c[0][0] for c in spider_instance.logger.warning.call_args_list)


# 4. Empty/Error Responses
def test_parse_empty_response_text_triggers_fallback_yields_empty(spider_instance):
    spider_instance.source = "empty_source_default"
    html_content = "<html><body></body></html>" # No <p> tags
    mock_response = HtmlResponse(url=spider_instance.url, body=html_content, encoding='utf-8')

    results = list(spider_instance.parse(mock_response))
    item = results[0]

    assert item['content'] == ""
    # Scrapy finds nothing, fails criteria. BS finds nothing.
    # Original Scrapy content (empty) is used. parser_used is "Scrapy CSS Selector"
    # because BS didn't provide alternative content.
    assert item['parser_used'] == "Scrapy CSS Selector" 
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

    assert "Error processing bs_exception_source. See logs." in item['content']
    # The parser_used at time of error would be after Scrapy failed, and BS was being attempted.
    # The spider sets parser_used to "Scrapy CSS Selector" initially. It's not updated before BS call.
    # So, if BS itself errors out, parser_used would still be "Scrapy CSS Selector".
    assert item['parser_used'] == "Scrapy CSS Selector" 
    
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

    assert "Error processing css_error_source. See logs." in item['content']
    assert item['parser_used'] == "Scrapy CSS Selector" # Initial value before error
    spider_instance.logger.error.assert_called_once()
    args, kwargs = spider_instance.logger.error.call_args
    assert "Error parsing" in args[0]
    assert "CSS Selector Engine Failed" in str(kwargs.get('exc_info'))
