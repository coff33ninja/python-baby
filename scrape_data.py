# scrape_data.py
import requests
import scrapy
from scrapy.crawler import CrawlerProcess
import os
import time
import logging

# To reduce verbosity from repeated Scrapy setup logs for each spider,
# you can set the log level for specific Scrapy components to WARNING.
# This should be done before CrawlerProcess is instantiated.
logging.getLogger('scrapy.addons').setLevel(logging.WARNING)
logging.getLogger('scrapy.extensions.telnet').setLevel(logging.WARNING)
# The 'scrapy.middleware' logger is responsible for "Enabled extensions/middlewares/pipelines" logs.
logging.getLogger('scrapy.middleware').setLevel(logging.WARNING)

class PythonSpider(scrapy.Spider):
    name = "python_spider"

    def __init__(self, source, url, stage="baby", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source = source
        self.stage = stage
        self.start_urls = [url] if url else []
        self.use_playwright = kwargs.pop('use_playwright', False)

    async def start(self):
        for url in self.start_urls:
            if self.use_playwright:
                yield scrapy.Request(url, meta={'playwright': True, 'playwright_include_page': False}, callback=self.parse)
            else:
                yield scrapy.Request(url, callback=self.parse)

    def parse(self, response):
        os.makedirs(f"data/{self.stage}", exist_ok=True)
        try:
            if "github" in self.source:
                data = response.json()
                for item in data.get("items", [])[:5]:
                    yield {
                        "content": item["description"] or "",
                        "file": f"data/{self.stage}/{self.source}.txt",
                    }
            elif self.source == "study_guides":
                text = response.css("p::text").getall()
                yield {
                    "content": " ".join(text)[:1000],
                    "file": f"data/{self.stage}/{self.source}.txt",
                }
            elif "stackoverflow" in self.source:
                questions = response.css(
                    ".question-summary .summary h3 a::text"
                ).getall()
                yield {
                    "content": "\n".join(questions[:5]),
                    "file": f"data/{self.stage}/{self.source}.txt",
                }
            elif "pypi" in self.source:
                text = response.css("div.project-description::text").getall()
                yield {
                    "content": " ".join(text)[:1000],
                    "file": f"data/{self.stage}/{self.source}.txt",
                }
            elif "reddit" in self.source:
                posts = response.css(".Post__title a::text").getall()
                yield {
                    "content": "\n".join(posts[:5]),
                    "file": f"data/{self.stage}/{self.source}.txt",
                }
            elif "python_docs" in self.source:
                text = response.css("div.body p::text").getall()
                yield {
                    "content": " ".join(text)[:1000],
                    "file": f"data/{self.stage}/{self.source}.txt",
                }
            else:
                text = response.css("p::text").getall()
                yield {
                    "content": " ".join(text)[:1000],
                    "file": f"data/{self.stage}/{self.source}.txt",
                }
        except Exception as e:
            self.logger.error(
                f"Error parsing {response.url} for source {self.source}. Content might be incomplete or missing. Error: {e}",
                exc_info=True
            )
            yield {"content": f"Error processing {self.source}. See logs.", "file": f"data/{self.stage}/{self.source}.txt"}


def fetch_pypi_updates(package):
    try:
        url = f"https://pypi.org/pypi/{package}/json"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return {
            "name": package,
            "version": data["info"]["version"],
            "content": data["info"]["summary"],
        }
    except requests.RequestException:
        return {"name": package, "content": ""}


def scrape_data(stage, sources, source_urls):
    os.makedirs(f"data/{stage}", exist_ok=True)

    settings = {
        "USER_AGENT": "PythonMasterAI/1.0",
        "TWISTED_REACTOR": "twisted.internet.asyncioreactor.AsyncioSelectorReactor", # Important for Playwright
        "DOWNLOAD_HANDLERS": {
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        "PLAYWRIGHT_BROWSER_TYPE": "chromium", # Or 'firefox', 'webkit'
        "PLAYWRIGHT_LAUNCH_OPTIONS": {
            "headless": True,
            "timeout": 30 * 1000,  # 30 seconds for browser launch
        },
        "LOG_LEVEL": "INFO", # Adjust as needed, DEBUG for more verbosity
    }
    process = CrawlerProcess(settings=settings)

    # Define sources that are known to require JavaScript rendering
    sources_needing_playwright = ["real_python", "reddit_learnpython"] # Add other sources as identified

    for source, url in zip(sources, source_urls):
        if url: # Ensure URL is not None or empty before crawling
            use_playwright_for_source = source in sources_needing_playwright
            try:
                process.crawl(PythonSpider, source=source, url=url, stage=stage, use_playwright=use_playwright_for_source)
            except Exception as e:
                print(f"Error scheduling scrape for {source}: {e}")
    
    print(f"Starting Scrapy crawl process for stage '{stage}'...")
    crawl_start_time = time.time()
    process.start(install_signal_handlers=False) # Start the process after all spiders are scheduled
    crawl_end_time = time.time()
    print(f"Scrapy crawl process for stage '{stage}' finished in {crawl_end_time - crawl_start_time:.2f} seconds.")

    if "pypi_docs" in sources:
        packages = (
            ["math", "os"] if stage in ["baby", "toddler"] else ["pandas", "polars"]
        )
        for pkg in packages:
            data = fetch_pypi_updates(pkg)
            with open(f"data/{stage}/pypi_docs.txt", "a") as f:
                f.write(f"{data['content']}\n")


if __name__ == "__main__":
    scrape_data(
        "baby",
        ["github_beginner", "study_guides"],
        [
            "https://api.github.com/search/repositories?q=language:python+stars:>100",
            "https://automatetheboringstuff.com/2e/chapter0/",
        ],
    )
