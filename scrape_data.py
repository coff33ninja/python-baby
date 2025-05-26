# scrape_data.py
import requests
import scrapy
from scrapy.crawler import CrawlerProcess
from bs4 import BeautifulSoup
import os
import time
import logging  # Standard logging
from datetime import datetime
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    RetryError,
    retry_if_exception_type,
)  # For retries
from utils import (
    get_config_value,
    setup_logging,
    get_typed_config_value,
)  # For config and logging setup

# --- Initialize logger for this module ---
logger = logging.getLogger(__name__)

# Scrapy's own logging is configured via settings, but we can reduce verbosity
# of some of its components if they are too noisy at the global level.
# However, setup_logging() will configure the root logger, which Scrapy also uses.
# We might need to adjust Scrapy's specific loggers if they become problematic.
# For now, let setup_logging() handle the overall configuration.
# logging.getLogger('scrapy.addons').setLevel(logging.WARNING)
# logging.getLogger('scrapy.extensions.telnet').setLevel(logging.WARNING)
# logging.getLogger('scrapy.middleware').setLevel(logging.WARNING)


class SaveToFilePipeline:
    def process_item(self, item, spider):  # spider.logger is Scrapy's logger
        file_path = item.get("file")
        content = item.get("content")

        if not file_path:
            spider.logger.warning(
                f"Missing 'file' key in item from {spider.name} for source {item.get('source', 'unknown')}. Item will not be saved. Item: {item}"
            )
            return item
        if content is None:
            spider.logger.warning(
                f"Missing 'content' in item for {file_path}. Item: {item}"
            )
            content = ""

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        try:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(content + "\n")
            spider.logger.debug(
                f"Appended content to {file_path} for source {item.get('source')}"
            )
        except Exception as e:
            spider.logger.error(
                f"Error writing to file {file_path}: {e}", exc_info=True
            )
        return item


class PythonSpider(scrapy.Spider):
    name = "python_spider"

    def __init__(self, source, url, version_data_dir, stage="baby", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source = source
        self.stage = stage
        self.version_data_dir = version_data_dir
        self.start_urls = [url] if url else []
        self.use_playwright = kwargs.pop("use_playwright", False)
        # self.logger is Scrapy's logger, automatically available.

    async def start_requests(self):
        if not self.start_urls:
            self.logger.info(
                f"No start_urls defined for source {self.source}, spider will not make requests."
            )
            return
        for url in self.start_urls:
            meta = {}
            if self.use_playwright:
                self.logger.info(f"Using Playwright for URL: {url}")
                meta["playwright"] = True
            yield scrapy.Request(url, callback=self.parse, meta=meta)

    def parse(self, response):
        # Helper function to check if content is meaningful
        def is_content_meaningful(content_str):
            return content_str and content_str.strip()

        output_filename = f"{self.source}.txt"
        output_filepath = os.path.join(self.version_data_dir, output_filename)
        extracted_content = ""
        parser_used = "Scrapy CSS Selector" # Default parser

        try:
            if "github" in self.source:
                # github logic remains the same, assuming JSON response doesn't need this HTML fallback
                data = response.json()
                descriptions = [item["description"] for item in data.get("items", [])[:5] if item.get("description")]
                extracted_content = "\n".join(descriptions)
                parser_used = "JSON API"
            
            elif self.source == "study_guides":
                self.logger.debug(f"Attempting to parse {response.url} for source '{self.source}' using Scrapy CSS selectors.")
                # Primary attempt: Scrapy CSS selectors
                text_list = response.css("p::text").getall()
                extracted_content = " ".join(text_list).strip()

                if not is_content_meaningful(extracted_content):
                    self.logger.warning(f"Scrapy CSS selector parsing yielded no meaningful content for {response.url} (source: {self.source}). Falling back to BeautifulSoup.")
                    soup = BeautifulSoup(response.text, 'lxml')
                    paragraphs = soup.find_all('p')
                    text_content_bs = [p.get_text(separator=' ', strip=True) for p in paragraphs]
                    extracted_content = " ".join(text_content_bs).strip()
                    parser_used = "BeautifulSoup"
                    if is_content_meaningful(extracted_content):
                        self.logger.info(f"Successfully parsed {response.url} (source: {self.source}) using BeautifulSoup after Scrapy CSS selector fallback.")
                    else:
                        self.logger.warning(f"BeautifulSoup parsing also yielded no meaningful content for {response.url} (source: {self.source}).")
                else:
                    self.logger.info(f"Successfully parsed {response.url} (source: {self.source}) using Scrapy CSS selectors.")

            # Add other elif blocks for different sources here, applying the same fallback pattern if they parse HTML
            # elif self.source == "another_html_source":
            #    # ... primary attempt with response.css() ...
            #    # ... if not is_content_meaningful(extracted_content): ...
            #    # ... fallback to BeautifulSoup ...
            
            else: # Generic HTML parsing fallback for other sources
                self.logger.debug(f"Attempting to parse {response.url} for source '{self.source}' using generic Scrapy CSS selectors (p tags).")
                # Primary attempt: Scrapy CSS selectors
                text_list = response.css("p::text").getall()
                extracted_content = " ".join(text_list).strip()

                if not is_content_meaningful(extracted_content):
                    self.logger.warning(f"Generic Scrapy CSS selector parsing (p tags) yielded no meaningful content for {response.url} (source: {self.source}). Falling back to BeautifulSoup (p tags).")
                    soup = BeautifulSoup(response.text, 'lxml')
                    paragraphs = soup.find_all('p') # Basic p tag extraction for generic case
                    text_content_bs = [p.get_text(separator=' ', strip=True) for p in paragraphs]
                    extracted_content = " ".join(text_content_bs).strip()
                    parser_used = "BeautifulSoup (generic p tags)"
                    if is_content_meaningful(extracted_content):
                        self.logger.info(f"Successfully parsed {response.url} (source: {self.source}) using BeautifulSoup (generic p tags) after Scrapy CSS selector fallback.")
                    else:
                        self.logger.warning(f"BeautifulSoup parsing (generic p tags) also yielded no meaningful content for {response.url} (source: {self.source}).")
                else:
                    self.logger.info(f"Successfully parsed {response.url} (source: {self.source}) using generic Scrapy CSS selectors (p tags).")

            # Common yield statement
            # Ensure content is truncated AFTER deciding which parser's content to use.
            yield {
                "content": extracted_content[:1000] if extracted_content else "", # Truncate after extraction
                "file": output_filepath,
                "source": self.source,
                "parser_used": parser_used # For logging/debugging
            }

        except Exception as e:
            self.logger.error(
                f"Error parsing {response.url} for source {self.source}. Parser used at time of error: {parser_used}. Error: {e}",
                exc_info=True,
            )
            yield {
                "content": f"Error processing {self.source}. See logs.",
                "file": output_filepath,
                "source": self.source,
                "parser_used": parser_used, # Log parser even in case of error
            }


@retry(
    stop=stop_after_attempt(
        get_typed_config_value("scraper.pypi_fetch.retry_attempts", 3, int)
    ),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(
        requests.exceptions.RequestException
    ),  # Only retry on requests exceptions
)
def fetch_pypi_updates(package_name: str, version_data_dir: str):
    """
    Fetches package information from PyPI, including summary and description (README).
    Retries on failure.
    """
    try:
        url = f"https://pypi.org/pypi/{package_name}/json"
        # Consider adding a specific user-agent for these requests too
        pypi_timeout = get_typed_config_value("scraper.pypi_fetch.timeout", 15, int)
        response = requests.get(url, timeout=pypi_timeout)
        response.raise_for_status()
        data = response.json()
        summary_content = data.get("info", {}).get("summary", "")
        output_filename = f"pypi_package_{package_name}.txt"
        output_filepath = os.path.join(version_data_dir, output_filename)
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, "a", encoding="utf-8") as f:
            f.write(summary_content + "\n")
        logger.info(f"Saved PyPI summary for {package_name} to {output_filepath}")

        readme_content = data.get("info", {}).get("description", "")
        readme_content_type = data.get("info", {}).get("description_content_type", "text/plain")
        readme_filename_saved = None

        if readme_content:
            readme_filename_ext = ".md" if "markdown" in readme_content_type.lower() else \
                                  ".rst" if "rst" in readme_content_type.lower() else ".txt"
            readme_filename = f"pypi_package_{package_name}_README{readme_filename_ext}"
            readme_filepath = os.path.join(version_data_dir, readme_filename)
            try:
                with open(readme_filepath, "w", encoding="utf-8") as f_readme:
                    f_readme.write(readme_content)
                logger.info(f"Saved PyPI README for {package_name} to {readme_filepath}")
                readme_filename_saved = readme_filename
            except Exception as e_readme:
                logger.error(f"Error saving PyPI README for {package_name} to {readme_filepath}: {e_readme}", exc_info=True)

        result_info = {
            "package_name": package_name,
            "version": data.get("info", {}).get("version", "N/A"),
            "summary_file_saved": output_filename,
            "summary_content_preview": summary_content[:100],
            "readme_file_saved": readme_filename_saved,
            "readme_content_preview": readme_content[:100] if readme_content else None,
        }
        return result_info
    except (requests.RequestException, RetryError) as e: # Catch RetryError from tenacity
        logger.error(f"Error fetching PyPI data for {package_name}: {e}", exc_info=True)
        return {"package_name": package_name, "summary_file_saved": None, "readme_file_saved": None, "error": str(e)}


def scrape_data(stage, sources, source_urls):
    version_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage_base_dir = os.path.join("data", stage)
    version_data_dir = os.path.join(stage_base_dir, version_timestamp)
    os.makedirs(version_data_dir, exist_ok=True)
    logger.info(
        f"Scraping data for stage '{stage}' into versioned directory: {version_data_dir}"
    )

    default_user_agent = get_config_value(
        "scraper.default_user_agent", "PythonMasterAI/1.0 (Default Scraper)"
    )
    # Scrapy LOG_LEVEL setting can be INFO, WARNING, ERROR, DEBUG, CRITICAL
    # It's better to control Scrapy's verbosity via its own settings if needed,
    # rather than just relying on the root logger level for everything.
    # For now, we set a general LOG_LEVEL for Scrapy.
    scrapy_log_level = get_config_value("logging.scrapy_log_level", "INFO")

    settings = {
        "USER_AGENT": default_user_agent,
        "TWISTED_REACTOR": "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
        "DOWNLOAD_HANDLERS": {
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        "PLAYWRIGHT_BROWSER_TYPE": "chromium",
        "PLAYWRIGHT_LAUNCH_OPTIONS": {"headless": True, "timeout": 30 * 1000},
        "LOG_LEVEL": scrapy_log_level,  # Use config for Scrapy's internal log level
        "ITEM_PIPELINES": {"scrape_data.SaveToFilePipeline": 300},
        "USER_AGENT": get_config_value(
            "scraper.default_user_agent", "PythonMasterAI_Scraper/1.0"
        ),
        # Retry Middleware Settings
        "RETRY_ENABLED": get_typed_config_value(
            "scraper.scrapy.retry_enabled", True, bool
        ),
        "RETRY_TIMES": get_typed_config_value("scraper.scrapy.retry_times", 3, int),
        "RETRY_HTTP_CODES": get_config_value(
            "scraper.scrapy.retry_http_codes", [500, 502, 503, 504, 522, 524, 408, 429]
        ),
        # AutoThrottle Extension (Rate Limiting)
        "AUTOTHROTTLE_ENABLED": get_typed_config_value(
            "scraper.scrapy.autothrottle_enabled", True, bool
        ),
        "AUTOTHROTTLE_START_DELAY": get_typed_config_value(
            "scraper.scrapy.autothrottle_start_delay", 1.0, float
        ),
        "AUTOTHROTTLE_MAX_DELAY": get_typed_config_value(
            "scraper.scrapy.autothrottle_max_delay", 10.0, float
        ),
        "AUTOTHROTTLE_TARGET_CONCURRENCY": get_typed_config_value(
            "scraper.scrapy.autothrottle_target_concurrency", 1.0, float
        ),
        "DOWNLOAD_DELAY": get_typed_config_value(
            "scraper.scrapy.download_delay", 0.5, float
        ),
        "CONCURRENT_REQUESTS_PER_DOMAIN": get_typed_config_value(
            "scraper.scrapy.concurrent_requests_per_domain", 8, int
        ),
    }
    process = CrawlerProcess(settings=settings)

    sources_needing_playwright = ["real_python", "reddit_learnpython"]
    scrapy_sources_processed = []

    for source, url in zip(sources, source_urls):
        if url:
            if source == "pypi_docs":
                continue
            use_playwright_for_source = source in sources_needing_playwright
            try:
                process.crawl(
                    PythonSpider,
                    source=source,
                    url=url,
                    stage=stage,
                    version_data_dir=version_data_dir,
                    use_playwright=use_playwright_for_source,
                )
                scrapy_sources_processed.append({"name": source, "url": url})
            except Exception as e:
                logger.error(
                    f"Error scheduling scrape for {source}: {e}", exc_info=True
                )

    if scrapy_sources_processed:
        logger.info(
            f"Starting Scrapy crawl process for stage '{stage}' into {version_data_dir}..."
        )
        crawl_start_time = time.time()
        process.start(install_signal_handlers=False)
        crawl_end_time = time.time()
        logger.info(
            f"Scrapy crawl process for stage '{stage}' finished in {crawl_end_time - crawl_start_time:.2f} seconds."
        )
    else:
        logger.info("No sources scheduled for Scrapy crawl.")

    pypi_results_for_manifest = []
    if "pypi_docs" in sources:
        packages_to_fetch = (
            ["math", "os"] if stage in ["baby", "toddler"] else ["pandas", "polars"]
        )
        logger.info(
            f"Fetching PyPI package data for: {packages_to_fetch} into {version_data_dir}"
        )
        for pkg_name in packages_to_fetch:
            pypi_data = fetch_pypi_updates(pkg_name, version_data_dir)
            pypi_results_for_manifest.append(pypi_data)

    manifest_content = {
        "version_timestamp": version_timestamp,
        "dataset_path_relative_to_stage_dir": version_timestamp,
        "creation_event_type": "scrape",
        "stage_scraped_for": stage,
        "scrapy_sources_attempted": scrapy_sources_processed,
        "pypi_packages_attempted": pypi_results_for_manifest,
    }
    manifest_filepath = os.path.join(version_data_dir, "manifest.json")
    try:
        with open(manifest_filepath, "w", encoding="utf-8") as f_manifest:
            json.dump(manifest_content, f_manifest, indent=4)
        logger.info(f"Saved manifest to {manifest_filepath}")
    except Exception as e:
        logger.error(f"Error saving manifest.json: {e}", exc_info=True)

    latest_txt_path = os.path.join(stage_base_dir, "latest.txt")
    try:
        with open(latest_txt_path, "w", encoding="utf-8") as f_latest:
            f_latest.write(version_timestamp)
        logger.info(
            f"Updated 'latest.txt' in {stage_base_dir} to version {version_timestamp}"
        )
    except Exception as e:
        logger.error(f"Error updating latest.txt: {e}", exc_info=True)


if __name__ == "__main__":
    import sys

    # Setup logging as early as possible
    # setup_logging() will read from config itself.
    setup_logging()
    # Scrapy specific loggers can be further tuned here if needed, after global setup
    # e.g., logging.getLogger('scrapy.core.engine').setLevel(logging.WARNING)

    if len(sys.argv) < 2:
        logger.error(
            "Usage: python scrape_data.py <stage> [source1 url1 source2 url2 ...]"
        )
        logger.info(
            "Example: python scrape_data.py baby github_beginner https://api.github.com/search/repositories?q=language:python+stars:>100"
        )
        sys.exit(1)

    stage_arg = sys.argv[1]
    source_url_pairs = sys.argv[2:]

    if len(source_url_pairs) % 2 != 0 and source_url_pairs:
        logger.error("Sources and URLs must be provided in pairs.")
        sys.exit(1)

    sources_arg = [source_url_pairs[i] for i in range(0, len(source_url_pairs), 2)]
    urls_arg = [source_url_pairs[i] for i in range(1, len(source_url_pairs), 2)]

    if not sources_arg or "pypi_docs" in sources_arg:
        if "pypi_docs" not in sources_arg:
            sources_arg.append("pypi_docs")
            urls_arg.append("")

    if not sources_arg:
        logger.error(
            f"No sources specified or defaulted for stage '{stage_arg}'. Exiting."
        )
        sys.exit(1)

    logger.info(
        f"Starting scrape_data script for stage: {stage_arg}, sources: {sources_arg}"
    )
    scrape_data(stage_arg, sources_arg, urls_arg)
