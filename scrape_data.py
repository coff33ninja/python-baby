# scrape_data.py
import requests
import scrapy
from scrapy.crawler import CrawlerProcess
import os
import time
import logging
from datetime import datetime # Added for versioning
import json # Added for versioning

# To reduce verbosity from repeated Scrapy setup logs for each spider,
# you can set the log level for specific Scrapy components to WARNING.
# This should be done before CrawlerProcess is instantiated.
logging.getLogger('scrapy.addons').setLevel(logging.WARNING)
logging.getLogger('scrapy.extensions.telnet').setLevel(logging.WARNING)
# The 'scrapy.middleware' logger is responsible for "Enabled extensions/middlewares/pipelines" logs.
logging.getLogger('scrapy.middleware').setLevel(logging.WARNING)

class SaveToFilePipeline:
    def process_item(self, item, spider):
        file_path = item.get('file')
        content = item.get('content')

        if not file_path:
            spider.logger.warning(f"Missing 'file' key in item from {spider.name} for source {item.get('source', 'unknown')}. Item will not be saved by this pipeline. Item: {item}")
            return item
        if content is None: # Allow empty string for content
            spider.logger.warning(f"Missing 'content' in item for {file_path}. Item: {item}")
            content = "" 

        # Ensure the directory for the file exists.
        # This will correctly create versioned subdirectories if file_path includes them.
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        try:
            with open(file_path, "a", encoding="utf-8") as f: # Append mode
                f.write(content + "\n")
            spider.logger.debug(f"Appended content to {file_path} for source {item.get('source')}")
        except Exception as e:
            spider.logger.error(f"Error writing to file {file_path}: {e}", exc_info=True)
        return item

class PythonSpider(scrapy.Spider):
    name = "python_spider"

    def __init__(self, source, url, version_data_dir, stage="baby", *args, **kwargs): # Added version_data_dir
        super().__init__(*args, **kwargs)
        self.source = source
        self.stage = stage # Retained for logging/context
        self.version_data_dir = version_data_dir # Store the versioned path
        self.start_urls = [url] if url else []
        self.use_playwright = kwargs.pop('use_playwright', False)

    async def start_requests(self):
        if not self.start_urls:
            self.logger.info(f"No start_urls defined for source {self.source}, spider will not make requests.")
            return

        for url in self.start_urls:
            meta = {}
            if self.use_playwright:
                self.logger.info(f"Using Playwright for URL: {url}")
                meta['playwright'] = True
            yield scrapy.Request(url, callback=self.parse, meta=meta)

    def parse(self, response):
        # The version_data_dir is already created by scrape_data function.
        # Files will be saved into this version_data_dir.
        output_filename = f"{self.source}.txt"
        output_filepath = os.path.join(self.version_data_dir, output_filename)

        try:
            if "github" in self.source:
                data = response.json()
                for item_data in data.get("items", [])[:5]:
                    yield {
                        "content": item_data["description"] or "",
                        "file": output_filepath,
                        "source": self.source 
                    }
            elif self.source == "study_guides":
                text = response.css("p::text").getall()
                yield {
                    "content": " ".join(text)[:1000],
                    "file": output_filepath,
                    "source": self.source
                }
            elif "stackoverflow" in self.source:
                questions = response.css(".question-summary .summary h3 a::text").getall()
                yield {
                    "content": "\n".join(questions[:5]),
                    "file": output_filepath,
                    "source": self.source
                }
            # Distinguish 'pypi' (general project pages) from 'pypi_docs' (specific package JSONs)
            elif "pypi" in self.source and self.source != "pypi_docs":
                text = response.css("div.project-description::text").getall()
                yield {
                    "content": " ".join(text)[:1000],
                    "file": output_filepath,
                    "source": self.source
                }
            elif "reddit" in self.source:
                posts = response.css(".Post__title a::text").getall()
                yield {
                    "content": "\n".join(posts[:5]),
                    "file": output_filepath,
                    "source": self.source
                }
            elif "python_docs" in self.source: # Assuming this is not the same as pypi_docs
                text = response.css("div.body p::text").getall()
                yield {
                    "content": " ".join(text)[:1000],
                    "file": output_filepath,
                    "source": self.source
                }
            else: # Default handler for other sources
                text = response.css("p::text").getall()
                yield {
                    "content": " ".join(text)[:1000],
                    "file": output_filepath,
                    "source": self.source
                }
        except Exception as e:
            self.logger.error(
                f"Error parsing {response.url} for source {self.source}. Content might be incomplete or missing. Error: {e}",
                exc_info=True
            )
            yield {"content": f"Error processing {self.source}. See logs.", "file": output_filepath, "source": self.source}


def fetch_pypi_updates(package_name: str, version_data_dir: str):
    """
    Fetches PyPI package summary and saves it to a file within version_data_dir.
    Returns a dictionary with package info and filepath for the manifest.
    """
    try:
        url = f"https://pypi.org/pypi/{package_name}/json"
        response = requests.get(url, timeout=10) # Increased timeout slightly
        response.raise_for_status()
        data = response.json()
        
        summary_content = data.get("info", {}).get("summary", "")
        # Use a more specific filename for pypi_docs if needed, or a generic one if it's always "pypi_docs.txt"
        # For individual package files:
        output_filename = f"pypi_package_{package_name}.txt" 
        output_filepath = os.path.join(version_data_dir, output_filename)
        
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True) # Ensure directory exists
        with open(output_filepath, "a", encoding="utf-8") as f: # Append mode
            f.write(summary_content + "\n")
        logging.info(f"Saved PyPI data for {package_name} to {output_filepath}")
        
        return {
            "package_name": package_name,
            "version": data.get("info", {}).get("version", "N/A"),
            "file_saved": output_filename, # Relative path for manifest
            "summary_content_preview": summary_content[:100] # For manifest brevity
        }
    except requests.RequestException as e:
        logging.error(f"Error fetching PyPI data for {package_name}: {e}")
        return {"package_name": package_name, "file_saved": None, "error": str(e)}


def scrape_data(stage, sources, source_urls):
    # --- Versioning Setup ---
    version_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage_base_dir = os.path.join("data", stage)
    version_data_dir = os.path.join(stage_base_dir, version_timestamp)
    os.makedirs(version_data_dir, exist_ok=True)
    print(f"Scraping data for stage '{stage}' into versioned directory: {version_data_dir}")

    settings = {
        "USER_AGENT": "PythonMasterAI/1.0",
        "TWISTED_REACTOR": "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
        "DOWNLOAD_HANDLERS": {
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        "PLAYWRIGHT_BROWSER_TYPE": "chromium",
        "PLAYWRIGHT_LAUNCH_OPTIONS": {"headless": True, "timeout": 30 * 1000},
        "LOG_LEVEL": "INFO",
        "ITEM_PIPELINES": {'scrape_data.SaveToFilePipeline': 300},
    }
    process = CrawlerProcess(settings=settings)

    sources_needing_playwright = ["real_python", "reddit_learnpython"]
    
    scrapy_sources_processed = [] # Keep track of sources handled by Scrapy

    for source, url in zip(sources, source_urls):
        if url:
            if source == "pypi_docs": # pypi_docs is handled by fetch_pypi_updates, not Scrapy spider
                continue
            use_playwright_for_source = source in sources_needing_playwright
            try:
                process.crawl(PythonSpider, source=source, url=url, stage=stage, 
                              version_data_dir=version_data_dir, use_playwright=use_playwright_for_source)
                scrapy_sources_processed.append({"name": source, "url": url})
            except Exception as e:
                print(f"Error scheduling scrape for {source}: {e}")
    
    if scrapy_sources_processed: # Only run Scrapy if there are sources for it
        print(f"Starting Scrapy crawl process for stage '{stage}' into {version_data_dir}...")
        crawl_start_time = time.time()
        process.start(install_signal_handlers=False)
        crawl_end_time = time.time()
        print(f"Scrapy crawl process for stage '{stage}' finished in {crawl_end_time - crawl_start_time:.2f} seconds.")
    else:
        print("No sources scheduled for Scrapy crawl.")

    pypi_results_for_manifest = []
    if "pypi_docs" in sources: # Check if 'pypi_docs' was requested
        packages_to_fetch = ["math", "os"] if stage in ["baby", "toddler"] else ["pandas", "polars"]
        print(f"Fetching PyPI package data for: {packages_to_fetch} into {version_data_dir}")
        for pkg_name in packages_to_fetch:
            pypi_data = fetch_pypi_updates(pkg_name, version_data_dir)
            pypi_results_for_manifest.append(pypi_data)

    # --- Manifest and latest.txt Creation ---
    manifest_content = {
        "version_timestamp": version_timestamp,
        "dataset_path_relative_to_stage_dir": version_timestamp, # Path of this versioned dir
        "creation_event_type": "scrape",
        "stage_scraped_for": stage,
        "scrapy_sources_attempted": scrapy_sources_processed,
        "pypi_packages_attempted": pypi_results_for_manifest
    }
    manifest_filepath = os.path.join(version_data_dir, "manifest.json")
    try:
        with open(manifest_filepath, "w", encoding="utf-8") as f_manifest:
            json.dump(manifest_content, f_manifest, indent=4)
        print(f"Saved manifest to {manifest_filepath}")
    except Exception as e:
        print(f"Error saving manifest.json: {e}")

    latest_txt_path = os.path.join(stage_base_dir, "latest.txt")
    try:
        with open(latest_txt_path, "w", encoding="utf-8") as f_latest:
            f_latest.write(version_timestamp)
        print(f"Updated 'latest.txt' in {stage_base_dir} to version {version_timestamp}")
    except Exception as e:
        print(f"Error updating latest.txt: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scrape_data.py <stage> [source1 url1 source2 url2 ...]")
        print("Example: python scrape_data.py baby github_beginner https://api.github.com/search/repositories?q=language:python+stars:>100 study_guides https://automatetheboringstuff.com/2e/chapter0/")
        sys.exit(1)
    
    stage_arg = sys.argv[1]
    source_url_pairs = sys.argv[2:]
    
    if len(source_url_pairs) % 2 != 0 and source_url_pairs: # Allow no pairs for just pypi_docs
        print("Error: Sources and URLs must be provided in pairs.")
        sys.exit(1)
        
    sources_arg = [source_url_pairs[i] for i in range(0, len(source_url_pairs), 2)]
    urls_arg = [source_url_pairs[i] for i in range(1, len(source_url_pairs), 2)]
    
    # Ensure pypi_docs is included if no specific sources are given, or if it's explicitly mentioned
    if not sources_arg or "pypi_docs" in sources_arg:
        if "pypi_docs" not in sources_arg:
             sources_arg.append("pypi_docs")
             urls_arg.append("") # pypi_docs doesn't need a URL here, handled by fetch_pypi_updates

    if not sources_arg: # Should not happen if pypi_docs is default, but as a safeguard
        print(f"No sources specified or defaulted for stage '{stage_arg}'. Exiting.")
        sys.exit(1)
    
    print(f"Starting scrape_data script for stage: {stage_arg}, sources: {sources_arg}")
    scrape_data(stage_arg, sources_arg, urls_arg)
