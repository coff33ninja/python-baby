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
import argparse # Added for command-line argument processing
from urllib.parse import urlparse # Added for URL parsing in discovery
import re # For sanitization
import shutil # For removing directory if clone fails partway
try:
    import git # GitPython library
except ImportError:
    git = None # Allows module to load/be tested if GitPython not installed yet
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

# Define the path to the API configuration file, used by scraper's discovery functions
API_CONFIG_FILE_PATH_SCRAPER = "api_config.json"

# Scrapy's own logging is configured via settings, but we can reduce verbosity
# of some of its components if they are too noisy at the global level.
# However, setup_logging() will configure the root logger, which Scrapy also uses.
# We might need to adjust Scrapy's specific loggers if they become problematic.
# For now, let setup_logging() handle the overall configuration.
# logging.getLogger('scrapy.addons').setLevel(logging.WARNING)
# logging.getLogger('scrapy.extensions.telnet').setLevel(logging.WARNING)
# logging.getLogger('scrapy.middleware').setLevel(logging.WARNING)

# Helper function for sanitizing (can be outside class or static)
def sanitize_path_segment(segment):
    if not segment:
        return "_" # Replace empty segments
    # Remove or replace characters invalid for filenames/paths
    # Basic example: replace non-alphanumeric with underscore
    # For more robust, consider a library or more comprehensive regex
    segment = re.sub(r'[^a-zA-Z0-9._-]', '_', segment)
    return segment[:200] # Limit length

class SaveToFilePipeline:
    def process_item(self, item, spider):
        if item.get("is_archive_item"):
            # --- Archiving Logic ---
            source_url = item.get("source_url")
            raw_body = item.get("raw_response_body")
            headers = item.get("response_headers", {})
            version_dir = item.get("version_data_dir")

            if not all([source_url, raw_body is not None, version_dir]):
                spider.logger.error(f"Archive item for {source_url} missing critical data. Item keys: {item.keys()}")
                return item # Or raise DropItem

            parsed_url = urlparse(source_url)
            domain = sanitize_path_segment(parsed_url.netloc)
            
            # Handle path segments carefully
            path_from_url = parsed_url.path.strip('/')
            path_segments_raw = [seg for seg in path_from_url.split('/') if seg] # Get raw segments

            # Determine filename
            # Default filename if path ends in / or is empty
            potential_filename_from_path = os.path.basename(parsed_url.path)
            
            filename = "index.dat" # Ultimate fallback
            has_extension = '.' in potential_filename_from_path

            if potential_filename_from_path and has_extension:
                filename = sanitize_path_segment(potential_filename_from_path)
                # If filename was derived from path, remove it from path_segments_raw if it's the last one
                if path_segments_raw and path_segments_raw[-1] == potential_filename_from_path:
                    path_segments_raw = path_segments_raw[:-1]
            else: # No clear filename from path (e.g. /foo/bar/ or /foo/bar or just /)
                content_type = headers.get("Content-Type", "").split(";")[0].strip().lower()
                if content_type == "text/html":
                    filename = "index.html"
                elif content_type == "application/json":
                    filename = "data.json"
                elif content_type == "text/plain":
                    filename = "content.txt"
                # If path_segments_raw is not empty and filename is still index.dat (or similar default)
                # it means the last segment of the URL path was not treated as a file.
                # e.g. for /foo/bar, path_segments_raw = ["foo", "bar"], filename="index.html" (if text/html)
            
            # Sanitize all raw path segments
            true_path_segments = [sanitize_path_segment(seg) for seg in path_segments_raw]

            archive_base_dir = os.path.join(version_dir, "archived_sites")
            target_dir_path = os.path.join(archive_base_dir, domain, *true_path_segments)
            archive_file_path = os.path.join(target_dir_path, filename) # filename is already sanitized
            
            try:
                os.makedirs(target_dir_path, exist_ok=True)
                with open(archive_file_path, "wb") as f: # Write bytes
                    f.write(raw_body)
                spider.logger.info(f"Archived raw content from {source_url} to {archive_file_path}")
            except Exception as e:
                spider.logger.error(f"Error archiving {source_url} to {archive_file_path}: {e}", exc_info=True)
            return item
        else:
            # --- Existing Text and Meta.json Saving Logic ---
            file_path = item.get("file")
            content = item.get("content")

            if not file_path: # This check is for the text item
                spider.logger.warning(f"Missing 'file' key in standard text item from {spider.name} for source {item.get('source', 'unknown')}. Item will not be saved. Item: {item}")
                return item
            if content is None: # Check for None, empty string is fine
                spider.logger.warning(f"Missing 'content' in standard text item for {file_path}. Item: {item}")
                content = "" # Default to empty string if None

            # This os.makedirs is for the .txt file
            os.makedirs(os.path.dirname(file_path), exist_ok=True) 
            try:
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(content + "\n")
                spider.logger.debug(f"Appended content to {file_path} for source {item.get('source')}")
            except Exception as e:
                spider.logger.error(f"Error writing to file {file_path}: {e}", exc_info=True)
            
            source_url = item.get("source_url") # This is for the .meta.json file
            if source_url:
                meta_file_path = file_path + ".meta.json"
                metadata = {
                    "source_url": source_url,
                    "scraped_timestamp": datetime.now().isoformat() # Ensure datetime is imported
                }
                try:
                    # Directory for .meta.json (same as .txt file) should already exist
                    with open(meta_file_path, "w", encoding="utf-8") as meta_f:
                        json.dump(metadata, meta_f, indent=4)
                    spider.logger.debug(f"Saved metadata to {meta_file_path} for source {item.get('source')}")
                except Exception as e:
                    spider.logger.error(f"Error writing metadata to {meta_file_path}: {e}", exc_info=True)
            else:
                spider.logger.warning(f"Missing 'source_url' in standard text item for {file_path} from {spider.name}. Metadata file will not be created. Item: {item}")
            return item


class PythonSpider(scrapy.Spider):
    name = "python_spider"
    DEFAULT_CONTENT_CRITERIA = {"min_length": 20} 
    SOURCE_SPECIFIC_CONTENT_CRITERIA = {
        "study_guides": {
            "min_length": 50,
            "must_not_contain_any": ["placeholder content", "coming soon", "under construction", "page not found"],
        },
        # Example for the generic 'else' case, can be same as default or slightly different
        "default_else_criteria": { # Using a specific key for the generic 'else' if needed, or rely on default
            "min_length": 25,
             "must_not_contain_any": ["nothing to see here", "error processing request"],
        }
        # Add other source-specific criteria here as needed
    }

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

    def _check_content_meaningfulness(self, content_str, criteria, source_name_for_log): # Added source_name_for_log
        # Basic check for empty or whitespace-only content
        if not content_str or not content_str.strip():
            self.logger.debug(f"Content from source '{source_name_for_log}' failed basic check (empty or whitespace).")
            return False

        min_length = criteria.get("min_length", 0)
        if len(content_str.strip()) < min_length:
            self.logger.debug(f"Content from source '{source_name_for_log}' (length: {len(content_str.strip())}) failed min_length check of {min_length}. Criteria: {criteria}")
            return False

        must_contain_any = criteria.get("must_contain_any", [])
        if must_contain_any and not any(keyword.lower() in content_str.lower() for keyword in must_contain_any):
            self.logger.debug(f"Content from source '{source_name_for_log}' failed must_contain_any check. Missing all of: {must_contain_any}. Criteria: {criteria}")
            return False
            
        must_contain_all = criteria.get("must_contain_all", [])
        if must_contain_all and not all(keyword.lower() in content_str.lower() for keyword in must_contain_all):
            self.logger.debug(f"Content from source '{source_name_for_log}' failed must_contain_all check. Not all present from: {must_contain_all}. Criteria: {criteria}")
            return False

        must_not_contain_any = criteria.get("must_not_contain_any", [])
        if must_not_contain_any and any(keyword.lower() in content_str.lower() for keyword in must_not_contain_any):
            found_forbidden = [keyword for keyword in must_not_contain_any if keyword.lower() in content_str.lower()]
            self.logger.debug(f"Content from source '{source_name_for_log}' failed must_not_contain_any check. Found: {found_forbidden}. Criteria: {criteria}")
            return False
        
        self.logger.debug(f"Content from source '{source_name_for_log}' passed all specific criteria: {criteria}")
        return True

    def parse(self, response):
        output_filename = f"{self.source}.txt"
        output_filepath = os.path.join(self.version_data_dir, output_filename)
        extracted_content = ""
        parser_used = "Scrapy CSS Selector" # Default parser

        try:
            if "github" in self.source:
                # github logic remains the same, not using HTML parsing or _check_content_meaningfulness
                data = response.json()
                descriptions = [item["description"] for item in data.get("items", [])[:5] if item.get("description")]
                extracted_content = "\n".join(descriptions)
                parser_used = "JSON API"
                # For GitHub, we might assume content is meaningful if present, or add specific checks if needed
                # For now, it bypasses the new content meaningfulness checks for HTML.
            
            elif self.source == "study_guides":
                self.logger.debug(f"Attempting to parse {response.url} for source '{self.source}' using Scrapy CSS selectors.")
                primary_text_list = response.css("p::text").getall() # Or more specific selectors for study_guides
                extracted_content = " ".join(primary_text_list).strip()
                parser_used = "Scrapy CSS Selector"

                current_criteria = self.SOURCE_SPECIFIC_CONTENT_CRITERIA.get(self.source, self.DEFAULT_CONTENT_CRITERIA)
                
                if self._check_content_meaningfulness(extracted_content, current_criteria, self.source):
                    self.logger.info(f"Successfully parsed {response.url} (source: '{self.source}') using Scrapy CSS selectors, meeting specific criteria.")
                else:
                    self.logger.warning(f"Scrapy CSS selector parsing for '{self.source}' ({response.url}) failed source-specific content criteria. Falling back to BeautifulSoup.")
                    
                    soup = BeautifulSoup(response.text, 'lxml')
                    paragraphs_bs = soup.find_all('p') 
                    text_content_bs = [p.get_text(separator=' ', strip=True) for p in paragraphs_bs]
                    extracted_content_bs = " ".join(text_content_bs).strip()
                    
                    # Assuming BS content is preferred if primary failed and BS found something.
                    # No separate check for BS content meaningfulness here as per simplified example.
                    if extracted_content_bs:
                         self.logger.info(f"Successfully parsed {response.url} (source: '{self.source}') using BeautifulSoup after Scrapy CSS selector fallback.")
                         extracted_content = extracted_content_bs 
                         parser_used = "BeautifulSoup"
                    else:
                         self.logger.warning(f"BeautifulSoup parsing also yielded no meaningful content for {response.url} (source: '{self.source}'). Original Scrapy content (if any) will be used or content will be empty.")
                         # extracted_content remains what Scrapy gave (which failed criteria), or empty.
                         # parser_used could be updated to "BeautifulSoup (failed)" or kept as is.

            # Add other elif blocks for different sources here, applying the same fallback pattern if they parse HTML
            
            else: # Generic HTML parsing fallback for other sources
                self.logger.debug(f"Attempting to parse {response.url} for source '{self.source}' using generic Scrapy CSS selectors (p tags).")
                primary_text_list = response.css("p::text").getall()
                extracted_content = " ".join(primary_text_list).strip()
                parser_used = "Scrapy CSS Selector"

                # Use "default_else_criteria" for the generic 'else' case
                current_criteria = self.SOURCE_SPECIFIC_CONTENT_CRITERIA.get("default_else_criteria", self.DEFAULT_CONTENT_CRITERIA)

                if self._check_content_meaningfulness(extracted_content, current_criteria, self.source):
                    self.logger.info(f"Successfully parsed {response.url} (source: '{self.source}') using generic Scrapy CSS selectors, meeting specific criteria.")
                else:
                    self.logger.warning(f"Generic Scrapy CSS selector parsing for '{self.source}' ({response.url}) failed source-specific content criteria. Falling back to BeautifulSoup.")
                    
                    soup = BeautifulSoup(response.text, 'lxml')
                    paragraphs_bs = soup.find_all('p') # Basic p tag extraction
                    text_content_bs = [p.get_text(separator=' ', strip=True) for p in paragraphs_bs]
                    extracted_content_bs = " ".join(text_content_bs).strip()
                    
                    if extracted_content_bs:
                         self.logger.info(f"Successfully parsed {response.url} (source: '{self.source}') using BeautifulSoup (generic p tags) after Scrapy CSS selector fallback.")
                         extracted_content = extracted_content_bs
                         parser_used = "BeautifulSoup (generic p tags)"
                    else:
                         self.logger.warning(f"BeautifulSoup parsing (generic p tags) also yielded no meaningful content for {response.url} (source: '{self.source}'). Original Scrapy content (if any) will be used or content will be empty.")

            # Common yield statement for extracted text
            yield {
                "content": extracted_content[:1000] if extracted_content else "", # Truncate after extraction
                "file": output_filepath,
                "source": self.source,
                "parser_used": parser_used, # For logging/debugging
                "source_url": response.url
            }

            # --- New: Yield archive item if applicable ---
            # For this subtask, to simplify, let's assume 'archive_this_source' is True for demonstration
            archive_this_source = True # Placeholder for actual config check logic
            # This would typically be:
            # archive_globally = get_typed_config_value("scraper.archive_sources", False, bool)
            # archive_this_source = get_typed_config_value(f"scraper.sources.{self.source}.archive", archive_globally, bool)

            if archive_this_source:
                self.logger.info(f"Yielding archive item for {response.url} (source: {self.source})")
                yield {
                    "source_url": response.url,
                    "source_name": self.source, 
                    "version_data_dir": self.version_data_dir,
                    "is_archive_item": True,
                    "raw_response_body": response.body, # bytes
                    "response_headers": {k.decode('utf-8', 'ignore'): [v.decode('utf-8', 'ignore') for v in vs] 
                                         for k, vs in response.headers.items()}
                }

        except Exception as e:
            self.logger.error(
                f"Error parsing {response.url} for source {self.source}. Parser used at time of error: {parser_used}. Error: {e}",
                exc_info=True,
            )
            # Ensure response.url is accessed safely, though it should be available if parse was called
            url_for_error_item = response.url if response else "unknown_url_due_to_error"
            yield {
                "content": f"Error processing {self.source}. See logs.",
                "file": output_filepath,
                "source": self.source,
                "parser_used": parser_used, # Log parser even in case of error
                "source_url": url_for_error_item
            }

# --- API-based Source Discovery Functions ---

def _load_api_config_for_scraper():
    if os.path.exists(API_CONFIG_FILE_PATH_SCRAPER):
        try:
            with open(API_CONFIG_FILE_PATH_SCRAPER, "r") as f:
                config = json.load(f)
            logger.info(f"Scraper: Successfully loaded API configuration from {API_CONFIG_FILE_PATH_SCRAPER}.")
            return config
        except Exception as e:
            logger.error(f"Scraper: Error reading or parsing API config file {API_CONFIG_FILE_PATH_SCRAPER}: {e}. API discovery will be limited.")
            return {}
    else:
        logger.warning(f"Scraper: API configuration file {API_CONFIG_FILE_PATH_SCRAPER} not found. API discovery disabled.")
        return {}

def discover_urls_from_query(query_str: str, api_config: dict):
    candidates = [] # List of (source_name, url) tuples
    logger.info(f"Scraper: Discovering sources for query: '{query_str}'")

    if not api_config:
        logger.warning("Scraper: API config is empty. Cannot perform dynamic discovery.")
        return candidates

    api_keys = api_config.get("api_keys", {})
    api_endpoints = api_config.get("api_endpoints", {})
    search_params_config = api_config.get("search_parameters", {})

    # --- GitHub Repository Search (Simplified from PythonMasterAI) ---
    github_pat = api_keys.get("github_pat")
    github_repo_search_url = api_endpoints.get("github_search_repositories")
    # Use a generic query parameter or make it configurable if needed
    github_repo_query_params = search_params_config.get("github_repo_default_query_params", "sort=stars&order=desc") 

    if github_pat and github_pat != "YOUR_GITHUB_PERSONAL_ACCESS_TOKEN_HERE" and github_repo_search_url:
        headers = {"Authorization": f"token {github_pat}", "Accept": "application/vnd.github.v3+json"}
        # Form a query: use the input query_str directly, append "python"
        search_query_github_repos = f"{query_str} python" 
        full_github_repo_url = f"{github_repo_search_url}?q={search_query_github_repos}&{github_repo_query_params}"
        logger.info(f"Scraper: Querying GitHub Repos: {full_github_repo_url}")
        try:
            response = requests.get(full_github_repo_url, headers=headers, timeout=10)
            response.raise_for_status()
            gh_results = response.json()
            for i, item in enumerate(gh_results.get("items", [])[:5]): # Limit to top 5
                repo_name = item.get("full_name")
                repo_url = item.get("html_url")
                if repo_name and repo_url:
                    s_name = f"github_discovered_{repo_name.replace('/', '_')}"
                    candidates.append((s_name, repo_url))
                    logger.debug(f"  Added GitHub candidate: {s_name} - {repo_url}")
        except requests.RequestException as e:
            logger.error(f"Scraper: GitHub API repo search failed for query '{query_str}': {e}")
        except json.JSONDecodeError as e:
             logger.error(f"Scraper: Failed to parse GitHub API repo search response for query '{query_str}': {e}")

    # --- Google Custom Search (Simplified from PythonMasterAI) ---
    google_api_key = api_keys.get("google_custom_search_api_key")
    google_cx_id = api_keys.get("google_custom_search_cx_id")
    google_search_url = api_endpoints.get("google_custom_search")
    google_num_results = search_params_config.get("google_search_default_num_results", "5")

    if google_api_key and google_api_key != "YOUR_GOOGLE_CSE_API_KEY_HERE" and \
       google_cx_id and google_cx_id != "YOUR_GOOGLE_CSE_CX_ID_HERE" and google_search_url:
        
        full_google_url = f"{google_search_url}?key={google_api_key}&cx={google_cx_id}&q={query_str}&num={google_num_results}"
        logger.info(f"Scraper: Querying Google CSE: {full_google_url}")
        try:
            response = requests.get(full_google_url, timeout=10)
            response.raise_for_status()
            google_results = response.json()
            for i, item in enumerate(google_results.get("items", [])): # Process configured number of results
                title = item.get("title", f"item_{i}")
                link = item.get("link")
                if link:
                    # Sanitize title for use in source name
                    safe_title = "".join(c if c.isalnum() else "_" for c in title[:30])
                    s_name = f"web_discovered_{urlparse(link).netloc.replace('.','_')}_{safe_title}"
                    candidates.append((s_name, link))
                    logger.debug(f"  Added Google Search candidate: {s_name} - {link}")
        except requests.RequestException as e:
            logger.error(f"Scraper: Google Custom Search API failed for query '{query_str}': {e}")
        except json.JSONDecodeError as e:
             logger.error(f"Scraper: Failed to parse Google Custom Search API response for query '{query_str}': {e}")
    
    logger.info(f"Scraper: Discovered {len(candidates)} potential new sources for query '{query_str}'.")
    return candidates

def clone_repo(repo_url: str, target_base_dir: str, depth: int = 1) -> str | None:
    """
    Clones a Git repository from repo_url into a subdirectory within target_base_dir.
    The subdirectory will be named after the repository.

    Args:
        repo_url: The URL of the Git repository to clone.
        target_base_dir: The base directory where the repo's directory will be created.
        depth: The depth for a shallow clone. Default is 1. Use 0 or None for full clone.

    Returns:
        The full path to the cloned repository locally, or None if cloning failed.
    """
    if git is None:
        logger.error("GitPython library is not installed. Repository cloning is disabled.")
        return None

    clone_target_path = None # Define outside try for cleanup access
    try:
        # Derive repo name from URL to create a subdirectory
        # e.g., https://github.com/user/myrepo.git -> myrepo
        repo_name_with_ext = os.path.basename(repo_url)
        repo_name = os.path.splitext(repo_name_with_ext)[0] 
        if not repo_name: # Handle cases like http://github.com/user/myrepo (no .git extension)
            repo_name = os.path.basename(repo_url.rstrip('/'))

        if not repo_name: # Still no repo name (e.g. strange URL or just domain)
            logger.error(f"Could not derive repository name from URL: {repo_url}")
            return None
        
        # Sanitize repo_name just in case, though os.path.basename usually gives valid part
        # This also handles if repo_name was derived from a path like "user/myrepo"
        repo_name = "".join(c if c.isalnum() else "_" for c in repo_name.replace('/', '_'))


        clone_target_path = os.path.join(target_base_dir, repo_name)

        if os.path.exists(clone_target_path):
            logger.info(f"Repository directory already exists: {clone_target_path}. Assuming already cloned or handled. Skipping clone.")
            # Option: Could add logic here to pull changes if it's already a git repo,
            # or remove and re-clone if a force_clone flag is added.
            # For now, just skip if path exists.
            return clone_target_path 
        
        logger.info(f"Cloning repository {repo_url} into {clone_target_path} with depth {depth if depth else 'full'}.")
        
        clone_options = {}
        if depth and depth > 0:
            clone_options['depth'] = depth
        
        # Ensure target_base_dir exists before trying to clone into a subdir of it
        os.makedirs(target_base_dir, exist_ok=True)

        git.Repo.clone_from(repo_url, clone_target_path, **clone_options)
        
        logger.info(f"Successfully cloned {repo_url} to {clone_target_path}.")
        return clone_target_path

    except git.exc.GitCommandError as e:
        logger.error(f"GitCommandError while cloning {repo_url}: {e.stderr if hasattr(e, 'stderr') else e}", exc_info=True)
        # If clone failed, remove partially created directory if it exists
        if clone_target_path and os.path.exists(clone_target_path):
            try:
                shutil.rmtree(clone_target_path)
                logger.info(f"Removed partially cloned directory: {clone_target_path}")
            except Exception as remove_err:
                logger.error(f"Error removing partially cloned directory {clone_target_path}: {remove_err}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while cloning {repo_url}: {e}", exc_info=True)
        if clone_target_path and os.path.exists(clone_target_path): 
            try:
                shutil.rmtree(clone_target_path)
                logger.info(f"Removed directory due to unexpected error during clone: {clone_target_path}")
            except Exception as remove_err:
                logger.error(f"Error removing directory {clone_target_path} after unexpected error: {remove_err}")
        return None

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

        # Save the raw JSON response
        raw_json_filename = f"pypi_package_{package_name}_raw.json"
        raw_json_filepath = os.path.join(version_data_dir, raw_json_filename)
        try:
            os.makedirs(os.path.dirname(raw_json_filepath), exist_ok=True) # Should already exist but good practice
            with open(raw_json_filepath, "w", encoding="utf-8") as f_raw_json:
                json.dump(data, f_raw_json, ensure_ascii=False, indent=4)
            logger.info(f"Saved raw PyPI JSON for {package_name} to {raw_json_filepath}")
        except Exception as e_raw_json:
            logger.error(f"Error saving raw PyPI JSON for {package_name} to {raw_json_filepath}: {e_raw_json}", exc_info=True)
            raw_json_filename = None # Indicate failure to save

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
            "raw_json_file_saved": raw_json_filename # Add this new field
        }
        return result_info
    except (requests.RequestException, RetryError) as e: # Catch RetryError from tenacity
        logger.error(f"Error fetching PyPI data for {package_name}: {e}", exc_info=True)
        return {
            "package_name": package_name, 
            "summary_file_saved": None, 
            "readme_file_saved": None, 
            "raw_json_file_saved": None, # Add here
            "error": str(e)
        }


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
    # argparse is already imported at the top of the file

    setup_logging() # Call early

    parser = argparse.ArgumentParser(description="Scrape data for PythonMasterAI.")
    parser.add_argument("stage", type=str, help="The AI stage for which data is being scraped (e.g., baby, toddler).")
    parser.add_argument("--query", type=str, help="A query string to discover sources via APIs.")
    # Use nargs='*' to capture all remaining arguments for source-url pairs
    parser.add_argument('source_url_pairs', nargs='*', help="Pairs of source_name and source_url to scrape directly.")

    args = parser.parse_args()

    stage_arg = args.stage
    query_arg = args.query
    source_url_pairs_arg = args.source_url_pairs

    sources_to_scrape_names = []
    sources_to_scrape_urls = []

    if query_arg:
        logger.info(f"Discovery mode: Using query '{query_arg}' to find sources.")
        api_config_loaded = _load_api_config_for_scraper()
        if not api_config_loaded:
            logger.error("API config not loaded. Cannot proceed with query-based discovery. Exiting.")
            sys.exit(1)
        
        discovered_sources = discover_urls_from_query(query_arg, api_config_loaded)
        if not discovered_sources:
            logger.warning(f"No sources discovered for query '{query_arg}'. Nothing to scrape.")
            sys.exit(0)
        
        sources_to_scrape_names = [s_info[0] for s_info in discovered_sources]
        sources_to_scrape_urls = [s_info[1] for s_info in discovered_sources]
        logger.info(f"Discovered {len(sources_to_scrape_names)} sources to scrape: {sources_to_scrape_names}")

    elif source_url_pairs_arg:
        if len(source_url_pairs_arg) % 2 != 0:
            logger.error("Sources and URLs for direct scraping must be provided in pairs.")
            sys.exit(1)
        sources_to_scrape_names = [source_url_pairs_arg[i] for i in range(0, len(source_url_pairs_arg), 2)]
        sources_to_scrape_urls = [source_url_pairs_arg[i] for i in range(1, len(source_url_pairs_arg), 2)]
        logger.info(f"Direct mode: Scraping specified {len(sources_to_scrape_names)} sources.")
    
    # Default to PyPI docs if no other sources are specified by query or direct input
    # (This maintains part of the original logic for ensuring some data is always available for certain stages)
    if not sources_to_scrape_names and "pypi_docs" not in sources_to_scrape_names :
         logger.info("No specific sources from query or arguments, adding 'pypi_docs' by default.")
         sources_to_scrape_names.append("pypi_docs")
         sources_to_scrape_urls.append("") # PyPI docs handling is special within scrape_data

    if not sources_to_scrape_names:
        logger.error(f"No sources specified or discovered for stage '{stage_arg}'. Exiting.")
        sys.exit(1)

    logger.info(f"Starting scrape_data script for stage: {stage_arg}, sources: {sources_to_scrape_names}")
    scrape_data(stage_arg, sources_to_scrape_names, sources_to_scrape_urls)
