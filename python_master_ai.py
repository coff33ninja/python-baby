# python_master_ai.py
import torch
import torch.nn as nn
import requests
import hashlib
import json
from collections import defaultdict
from transformers import AutoTokenizer
import ast
import re
import time
from urllib.parse import urlparse
from datetime import datetime
import os

class PythonMasterAI(nn.Module):
    MASTER_KEY = "8f9b7f8f6e6c9b9d7e7f9b8f6e6c9b9d7e7f9b8f6e6c9b9d7e7f9b8f6e6c9b9d"

    def __init__(self, vocab_size=16000, n_layers=2, n_heads=4, hidden_size=256,
                 dropout=0.1, dim_feedforward=None, activation="relu"):
        super().__init__()
        self.vocab_size = vocab_size # Store for reference
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_layers = n_layers # Initial number of layers
        self.dropout = dropout
        self.activation = activation
        self.dim_feedforward = dim_feedforward if dim_feedforward is not None else hidden_size * 4

        # Utilize hashlib to create a configuration ID
        config_params_str = f"v{vocab_size}_l{n_layers}_h{n_heads}_hs{hidden_size}_d{dropout}_df{self.dim_feedforward}_a{activation}"
        self.configuration_id = hashlib.sha1(config_params_str.encode()).hexdigest()[
            :12
        ]
        print(f"Configuration ID: {self.configuration_id}")

        # Initialize the embedding layer
        self.embed = nn.Embedding(vocab_size, hidden_size)
        # The nn.Transformer module will create TransformerEncoderLayers internally.
        # We configure it to use batch_first=True and pass other relevant hyperparameters.
        self.transformer = nn.Transformer(
            d_model=self.hidden_size,
            nhead=self.n_heads,
            num_encoder_layers=self.n_layers,
            num_decoder_layers=0,  # Explicitly setting to 0 for an encoder-only architecture
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True  # Key change: ensures internal layers are batch_first
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

        self.performance_log = []
        self.research_log = []
        self.source_log = []
        self.stage = "baby"
        self.growth_tasks = self.define_growth_tasks()
        self.task_progress = defaultdict(int)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Replace with custom
        self.knowledge_gaps = []
        self.known_sources = self.load_known_sources()

        # Utilize torch for device management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x, src_key_padding_mask=None):
        # Input x shape: (batch_size, seq_len)
        # Embedding output shape: (batch_size, seq_len, hidden_size) due to batch_first=True
        embedded_src = self.embed(x)

        # Pass through the Transformer's encoder part
        # Input shape: (batch_size, seq_len, hidden_size)
        # Output shape: (batch_size, seq_len, hidden_size)
        if self.n_layers > 0: # Only use transformer if layers exist
            transformer_output = self.transformer.encoder(embedded_src, src_key_padding_mask=src_key_padding_mask)
        else: # If no layers (e.g. n_layers=0 initially), bypass transformer encoder
            transformer_output = embedded_src

        # Output layer
        # Input shape: (batch_size, seq_len, hidden_size)
        # Output shape: (batch_size, seq_len, vocab_size)
        output = self.fc(transformer_output)
        return output

    def log_performance(self, metric, value):
        self.performance_log.append((metric, value))
        with open("performance_log.json", "a") as f:
            json.dump({"metric": metric, "value": value}, f)

    def log_research(self, topic, sources, success, note=""):
        log_entry = {"topic": topic, "sources": sources, "success": success}
        if note:
            log_entry["note"] = note
        self.research_log.append(log_entry)
        with open("research_log.json", "a") as f:
            json.dump(log_entry, f)

    def log_source(self, source, url, score, added):
        self.source_log.append({"source": source, "url": url, "score": score, "added": added})
        with open("source_log.json", "a") as f:
            json.dump({"source": source, "url": url, "score": score, "added": added}, f)

    def load_known_sources(self):
        return {
            "github_beginner": ["https://api.github.com/search/repositories?q=language:python+stars:>100"],
            "study_guides": ["https://automatetheboringstuff.com/2e/chapter0/"],
            "stackoverflow_basic": ["https://stackoverflow.com/questions/tagged/python?sort=newest"],
            "github_intermediate": ["https://api.github.com/search/repositories?q=language:python+stars:>500"],
            "pypi_docs": ["https://pypi.org/project/math/", "https://pypi.org/project/os/"],
            "real_python": ["https://realpython.com/"],
            "github_advanced": ["https://api.github.com/search/repositories?q=language:python+stars:>1000"],
            "peps": ["https://peps.python.org/"],
            "reddit_learnpython": ["https://www.reddit.com/r/learnpython/"],
            "python_docs": ["https://docs.python.org/3/"]
        }

    def assess_performance(self):
        avg_loss = sum(v for k, v in self.performance_log if k == "loss") / max(1, len(self.performance_log))
        task_completion = {task: self.task_progress[task] for task in self.growth_tasks[self.stage]}
        accuracy = self.task_progress.get("unit_test_accuracy", 0)
        research_needed = len(self.knowledge_gaps) > 0 or accuracy < self.growth_tasks[self.stage]["unit_test_accuracy"]
        ready_to_grow = avg_loss < 0.5 and accuracy >= self.growth_tasks[self.stage]["unit_test_accuracy"] and \
                        all(self.task_progress[task] >= count for task, count in self.growth_tasks[self.stage].items() if isinstance(count, int))
        return {"loss": avg_loss, "needs_growth": ready_to_grow, "needs_research": research_needed, "tasks": task_completion, "accuracy": accuracy}

    def update_stage(self):
        params = sum(p.numel() for p in self.parameters())
        old_stage = self.stage
        if params >= 10_000_000_000:
            self.stage = "adult"
        elif params >= 1_000_000_000:
            self.stage = "teenager"
        elif params >= 10_000_000:
            self.stage = "toddler"
        if old_stage != self.stage:
            print(f"Promoted to {self.stage.capitalize()} with {params:,} parameters!")
            self.task_progress = defaultdict(int)
            self.knowledge_gaps = []
            with open("task_progress.json", "w") as f:
                json.dump({}, f)

    def define_growth_tasks(self):
        return {
            "baby": {"write_functions": 10, "explain_variables": 5, "research_basics": 3, "find_sources": 2, "unit_test_accuracy": 0.8},
            "toddler": {"write_classes": 20, "use_libraries": 10, "explain_classes": 5, "research_intermediate": 5, "find_sources": 3, "unit_test_accuracy": 0.85},
            "teenager": {"write_complex_scripts": 50, "debug_errors": 20, "handle_peps": 5, "research_advanced": 10, "find_sources": 5, "unit_test_accuracy": 0.9},
            "adult": {"optimize_libraries": 10, "propose_peps": 3, "research_trends": 5, "find_sources": 5, "unit_test_accuracy": 0.95}
        }

    def log_task_progress(self, task, success=True):
        if task in self.growth_tasks[self.stage]:
            self.task_progress[task] += 1 if success else 0
            with open("task_progress.json", "w") as f:
                json.dump(dict(self.task_progress), f)

    def validate_code(self, code):
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def identify_knowledge_gaps(self, input_text, task, success):
        if not success:
            topic = re.search(r"(write|explain|debug)\s+(\w+)", input_text.lower())
            if topic:
                gap = f"{topic.group(2)} {task.split('_')[-1]}"
                if gap not in self.knowledge_gaps:
                    self.knowledge_gaps.append(gap)
                    print(f"Gap identified: {gap}")

    def formulate_research_queries(self):
        queries = []
        for gap in self.knowledge_gaps:
            if self.stage == "baby":
                queries.append(f"{gap} beginner tutorial")
            elif self.stage == "toddler":
                queries.append(f"{gap} python documentation")
            elif self.stage == "teenager":
                queries.append(f"{gap} advanced example")
            else:
                queries.append(f"{gap} python ecosystem trends")
        return queries or ["python " + self.stage + " tutorial"]

    def _fetch_github_repo_details(self, repo_url_or_api_url):
        """
        Fetches repository details from GitHub API.
        Handles both web URLs (e.g., https://github.com/owner/repo)
        and direct API URLs (e.g., https://api.github.com/repos/owner/repo).
        """
        parsed_url = urlparse(repo_url_or_api_url)
        api_url_to_use = None

        if parsed_url.netloc == "api.github.com" and parsed_url.path.startswith("/repos/"):
            api_url_to_use = repo_url_or_api_url
        elif parsed_url.netloc == "github.com":
            path_parts = parsed_url.path.strip('/').split('/')
            if len(path_parts) >= 2:
                owner, repo_name_segment = path_parts[0], path_parts[1]
                # Ensure repo name doesn't contain further slashes like branches/tags
                repo_name = repo_name_segment.split('/')[0]
                api_url_to_use = f"https://api.github.com/repos/{owner}/{repo_name}"

        if not api_url_to_use:
            # print(f"Debug: Could not determine GitHub API URL from: {repo_url_or_api_url}")
            return None

        try:
            # Consider adding authentication (e.g., a GitHub token) for higher rate limits
            # headers = {"Authorization": "token YOUR_GITHUB_TOKEN"}
            response = requests.get(api_url_to_use, timeout=5) # Add headers=headers if using token
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Warning: Error fetching GitHub repo details for {api_url_to_use}: {e}")
            return None

    def _fetch_pypi_package_info(self, package_name_or_url):
        """
        Fetches package information from PyPI JSON API.
        Handles PyPI project URLs (e.g., https://pypi.org/project/requests)
        or just package names (e.g., "requests").
        """
        package_name = ""
        parsed_uri = urlparse(package_name_or_url)

        if parsed_uri.netloc == "pypi.org":
            path_segments = parsed_uri.path.strip('/').split('/')
            if len(path_segments) > 1:
                if path_segments[0] == "project":
                    package_name = path_segments[1]
                elif path_segments[0] == "pypi": # e.g. /pypi/<package_name>/json
                    package_name = path_segments[1]
        elif not parsed_uri.scheme and not parsed_uri.netloc: # Likely just a package name
            package_name = package_name_or_url.strip()

        if not package_name:
            # print(f"Debug: Could not determine PyPI package name from: {package_name_or_url}")
            return None

        api_url = f"https://pypi.org/pypi/{package_name}/json"
        try:
            response = requests.get(api_url, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Warning: Error fetching PyPI package info for {package_name}: {e}")
            return None

    def select_research_sources(self, query):
        sources = list(self.known_sources.keys())
        if self.stage == "baby":
            return [s for s in sources if "beginner" in s or "study_guides" in s or "stackoverflow_basic" in s]
        elif self.stage == "toddler":
            return [s for s in sources if "intermediate" in s or "pypi_docs" in s or "real_python" in s]
        elif self.stage == "teenager":
            return [s for s in sources if "advanced" in s or "peps" in s or "reddit_learnpython" in s]
        return [s for s in sources if "trending" in s or "python_docs" in s]

    def validate_research_data(self, content, source):
        if "github" in source:
            return len(content) > 10 and "python" in content.lower()
        if "stackoverflow" in source:
            return len(content.split("\n")) > 1
        if "pypi" in source or "python_docs" in source:
            return len(content) > 50
        return True

    def get_research_scrape_targets(self):
        """
        Determines sources and URLs to scrape based on knowledge gaps.
        Returns a list of (source_name, url) tuples.
        """
        queries = self.formulate_research_queries()
        scrape_targets_dict = {} # Use a dict {source_name: url} to avoid duplicates

        for query in queries:
            selected_sources_for_query = self.select_research_sources(query)
            for src_name in selected_sources_for_query:
                if src_name in self.known_sources and self.known_sources[src_name]:
                    # Assuming the first URL for the source is the target
                    scrape_targets_dict[src_name] = self.known_sources[src_name][0]
                else:
                    print(f"Warning: Source '{src_name}' for query '{query}' not in known_sources or has no URL for research.")
        return list(scrape_targets_dict.items())

    def process_scraped_research_data(self, stage):
        """
        Processes scraped data files to validate research, log, and update knowledge gaps.
        This should be called AFTER scrape_data has completed.
        """
        queries = self.formulate_research_queries()
        if not queries:
            print("No active research queries to process post-scraping.")
            return

        query_resolution_map = {query: False for query in queries}

        all_relevant_sources_for_queries = set()
        for query in queries:
            all_relevant_sources_for_queries.update(self.select_research_sources(query))

        for source_name in all_relevant_sources_for_queries:
            file_path = f"data/{stage}/{source_name}.txt"
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                is_content_valid = self.validate_research_data(content, source_name)

                for query in queries:
                    if source_name in self.select_research_sources(query): # Check if this source was relevant for this query
                        if is_content_valid:
                            if not query_resolution_map[query]: # Log success only once per query
                                self.log_research(query, [source_name], success=True)
                                query_resolution_map[query] = True
                                research_task_key = next((k for k in self.growth_tasks[self.stage] if k.startswith("research_")), None)
                                if research_task_key:
                                    self.log_task_progress(research_task_key)
                        # else: # If content invalid, and query not yet resolved, it remains unresolved.
                        # if not query_resolution_map[query]:
                        # self.log_research(query, [source_name], success=False) # Optionally log this failure
            else: # File does not exist
                for query in queries: # Mark query as failed if a relevant source file is missing and query not yet resolved
                    if source_name in self.select_research_sources(query) and not query_resolution_map[query]:
                        self.log_research(query, [source_name], success=False, note="Scraped file not found")
                        print(f"Warning: Scraped file {file_path} not found for source {source_name} relevant to query '{query}'.")

        # Update knowledge gaps based on resolved queries
        resolved_gaps_this_cycle = set()
        for query, resolved in query_resolution_map.items():
            if resolved:
                for gap_text in self.knowledge_gaps:
                    # More robust check if the query was derived from this gap
                    # This assumes formulate_research_queries creates queries directly from gap text
                    if gap_text.lower() in query.lower(): # A simple check, might need refinement
                        resolved_gaps_this_cycle.add(gap_text)

        if resolved_gaps_this_cycle:
            print(f"Knowledge gaps resolved this cycle: {resolved_gaps_this_cycle}")
            self.knowledge_gaps = [gap for gap in self.knowledge_gaps if gap not in resolved_gaps_this_cycle]

        if queries and not self.knowledge_gaps:
            print("All knowledge gaps from this research cycle appear to be resolved.")
        elif queries:
            print(f"Remaining knowledge gaps after research: {self.knowledge_gaps}")

    def conduct_research(self): # This method is now primarily for GUI or other direct calls
        """
        Orchestrates research: gets targets, triggers scraping, and processes results.
        """
        print("Conduct_research called. Identifying targets...")

        research_targets = self.get_research_scrape_targets()

        if research_targets:
            print(f"Conduct_research initiating focused scraping for targets: {research_targets}")
            # Import locally to avoid circular dependencies or loading it if not needed,
            # and because it's specific to this block of logic.
            from scrape_data import scrape_data 

            sources_to_scrape, urls_to_scrape = zip(*research_targets)

            # Convert tuples from zip to lists, as expected by scrape_data
            scrape_data(self.stage, list(sources_to_scrape), list(urls_to_scrape))
        else:
            print("No specific research targets identified by conduct_research. Relying on general scraping or previously scraped data.")

        # Crucially, after attempting to scrape (if targets were present), OR if no targets were present,
        # it MUST still call self.process_scraped_research_data(self.stage)
        # to process any data that might be available (either from its own scrape or a previous one).
        self.process_scraped_research_data(self.stage)

    # Old conduct_research logic that called scrape_data internally per query:
    # def conduct_research_old_per_query_scrape(self):
    #     queries = self.formulate_research_queries()
    #     for query in queries:
    # ... (rest of the old method that calls scrape_data in a loop) ...

    def discover_new_sources(self):
        queries = self.formulate_research_queries()
        new_sources = []
        for query in queries:
            # Placeholder: Use Google API or GitHub trends (simulated here)
            candidate_sources = self.search_for_sources(query)
            for source, url in candidate_sources:
                score = self.evaluate_source(source, url, query)
                if score > 0.7:  # Threshold for quality
                    self.known_sources[source] = [url]
                    self.log_source(source, url, score, added=True)
                    new_sources.append(source)
                    self.log_task_progress("find_sources")
                else:
                    self.log_source(source, url, score, added=False)
        print(f"Discovered sources: {new_sources}")
        return new_sources

    def search_for_sources(self, query):
        # Simulated: Replace with Google API or GitHub API
        candidates = []
        if self.stage == "baby":
            candidates.append(("python_blog", "https://python-blog.example.com"))
        elif self.stage == "toddler":
            candidates.append(("python_tutorials", "https://tutorials.python.org"))
        elif self.stage == "teenager":
            candidates.append(("python_forum", "https://forum.python.org"))
        else:
            candidates.append(("python_conference", "https://pycon.org"))
        return candidates

    def evaluate_source(self, source_name, url, query):
        """
        Evaluates a potential new source based on relevance, authority, and freshness.
        Returns a score between 0.0 and 1.0.
        """
        relevance_score = 0.0
        authority_score = 0.0
        freshness_score = 0.0

        # 1. Relevance Score (max 0.4)
        query_lower = query.lower()
        query_terms = query_lower.split()
        relevance_points = 0
        if any(term in source_name.lower() for term in query_terms):
            relevance_points += 1
        if any(term in url.lower() for term in query_terms):
            relevance_points += 1
        if "python" in source_name.lower() or "python" in url.lower():
            relevance_points += 0.5 # Bonus for general Python keyword

        relevance_score = min(relevance_points / 2.5, 1.0) * 0.4 if relevance_points > 0 else 0.05

        # --- Fetch details once if applicable ---
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        github_details = None
        pypi_details = None

        if "github.com" in domain or "api.github.com" in domain:
            github_details = self._fetch_github_repo_details(url)
        elif "pypi.org" in domain:
            pypi_details = self._fetch_pypi_package_info(url)

        # 2. Authority Score (max 0.3)
        if github_details and 'stargazers_count' in github_details:
            stars = github_details['stargazers_count']
            if stars >= 1000:
                authority_score = 0.3
            elif stars >= 100:
                authority_score = 0.2
            else:
                authority_score = 0.1
        elif pypi_details and 'info' in pypi_details and 'version' in pypi_details['info']:
            authority_score = 0.25 # PyPI packages are generally curated and versioned
        elif "stackoverflow.com" in domain:
            authority_score = 0.25
        elif "docs.python.org" in domain or "peps.python.org" in domain:
            authority_score = 0.3
        elif any(known_good_domain in domain for known_good_domain in ["realpython.com", "djangoproject.com", "flask.palletsprojects.com"]):
            authority_score = 0.25
        else: # Default for unknown or less structured sources
            authority_score = 0.1

        # 3. Freshness Score (max 0.3)
        current_time_ts = time.time()
        # Default freshness, can be overridden by specific source logic
        freshness_score = 0.05 # Default to not very fresh if no specific info

        if github_details and 'pushed_at' in github_details:
            last_push_date_str = github_details['pushed_at']
            try:
                last_push_dt = datetime.fromisoformat(last_push_date_str.replace('Z', '+00:00'))
                last_push_ts = last_push_dt.timestamp()
                age_days = (current_time_ts - last_push_ts) / (60 * 60 * 24)

                if age_days <= 30:
                    freshness_score = 0.3    # Updated in last month
                elif age_days <= 180:
                    freshness_score = 0.2  # Updated in last 6 months
                elif age_days <= 365:
                    freshness_score = 0.15 # Updated in last year
                else:
                    freshness_score = 0.05                 # Older than a year
            except ValueError:
                print(f"Warning: Could not parse GitHub date: {last_push_date_str}")
        elif pypi_details and 'releases' in pypi_details and pypi_details['releases']:
            latest_version_str = pypi_details.get('info', {}).get('version')
            if latest_version_str and latest_version_str in pypi_details['releases']:
                release_dates = [
                    item['upload_time_iso_8601']
                    for item in pypi_details['releases'][latest_version_str]
                    if 'upload_time_iso_8601' in item
                ]
                if release_dates:
                    # Take the most recent upload time for the latest version
                    latest_upload_time_str = max(release_dates)
                    try:
                        upload_dt = datetime.fromisoformat(latest_upload_time_str.replace('Z', '+00:00'))
                        upload_ts = upload_dt.timestamp()
                        age_days = (current_time_ts - upload_ts) / (60 * 60 * 24)

                        if age_days <= 90:
                            freshness_score = 0.3   # Released in last 3 months
                        elif age_days <= 365:
                            freshness_score = 0.2 # Released in last year
                        else:
                            freshness_score = 0.1
                    except ValueError:
                        print(f"Warning: Could not parse PyPI date: {latest_upload_time_str}")
        # For general websites, freshness is hard without scraping content for dates.
        # Could try 'Last-Modified' HTTP header if fetching the page.

        total_score = relevance_score + authority_score + freshness_score
        # print(f"Debug: Evaluated '{source_name}' ({url}) for query '{query}': R={relevance_score:.2f}, A={authority_score:.2f}, F={freshness_score:.2f} -> Total={total_score:.2f}")
        return min(total_score, 1.0) # Ensure score is capped at 1.0


    def prioritize_scraping(self):
        """Selects a list of source names to scrape based on the current stage."""
        all_sources_for_stage = []
        if self.stage == "baby":
            all_sources_for_stage = [s for s in self.known_sources.keys() if "beginner" in s or "study_guides" in s or "stackoverflow_basic" in s]
        elif self.stage == "toddler":
            all_sources_for_stage = [s for s in self.known_sources.keys() if "intermediate" in s or "pypi_docs" in s or "real_python" in s]
        elif self.stage == "teenager":
            all_sources_for_stage = [s for s in self.known_sources.keys() if "advanced" in s or "peps" in s or "reddit_learnpython" in s]
        else: # adult
            all_sources_for_stage = [s for s in self.known_sources.keys() if "trending" in s or "python_docs" in s or "peps" in s]

        # Ensure pypi_docs is included if relevant for the stage, as scrape_data has special handling for it.
        if "pypi_docs" not in all_sources_for_stage and "pypi_docs" in self.known_sources:
            if self.stage in ["baby", "toddler"]: # Based on scrape_data.py logic for pypi_docs packages
                all_sources_for_stage.append("pypi_docs")

        valid_sources = [s for s in all_sources_for_stage if s in self.known_sources and self.known_sources[s]]
        return valid_sources

    def generate_response(self, input_text):
        if self.assess_performance()["needs_research"]:
            self.conduct_research()
            self.discover_new_sources()
        if "write" in input_text.lower():
            task = "write_functions" if self.stage == "baby" else "write_classes" if self.stage == "toddler" else "write_complex_scripts"
            code = self.generate_code(input_text)
            success = self.validate_code(code)
            self.log_task_progress(task, success)
            self.identify_knowledge_gaps(input_text, task, success)
            return f"{self.stage.capitalize()} AI: Generated code:\n```python\n{code}\n```"
        elif "explain" in input_text.lower():
            task = "explain_variables" if self.stage == "baby" else "explain_classes"
            explanation = self.generate_explanation(input_text)
            self.log_task_progress(task)
            self.identify_knowledge_gaps(input_text, task, True)
            return f"{self.stage.capitalize()} AI: {explanation}"
        elif "debug" in input_text.lower():
            task = "debug_errors" if self.stage == "teenager" else None
            if task:
                debug = self.debug_code(input_text)
                self.log_task_progress(task)
                self.identify_knowledge_gaps(input_text, task, True)
                return f"{self.stage.capitalize()} AI: Debug: {debug}"
        return f"{self.stage.capitalize()} AI: Processing {input_text}..."

    def generate_code(self, input_text):
        if self.stage == "baby":
            return "def add(a, b):\n    return a + b"
        elif self.stage == "toddler":
            return "class Counter:\n    def __init__(self):\n        self.count = 0\n    def increment(self):\n        self.count += 1"
        return "# Complex script (to be implemented)"

    def generate_explanation(self, input_text):
        if self.stage == "baby":
            return "A variable is a name that stores data, like `x = 5` (per 'Automate the Boring Stuff')."
        return "A class is a blueprint for objects, defining properties and methods (per Python docs)."

    def debug_code(self, input_text):
        return "Check for missing colons or incorrect indentation (placeholder)."

    def process_input(self, input_text, user_key):
        response = requests.post("http://localhost:8000/master/auth", json={"key": user_key, "command": input_text})
        if response.status_code == 200:
            if response.json().get("action") == "pause":
                self.reset_to_checkpoint()
                return f"{self.stage.capitalize()} paused by Master’s stop code"
            if "MASTER:" in input_text:
                return f"Serving Master: {self.generate_response(input_text.replace('MASTER:', ''))}"
        return self.generate_response(input_text)

    def reset_to_checkpoint(self):
        print("Reverting to checkpoint")
        self.task_progress = defaultdict(int)
        self.knowledge_gaps = []

    def get_status(self):
        params = sum(p.numel() for p in self.parameters())
        status = {
            "stage": self.stage,
            "parameters": f"{params:,}",
            "configuration_id": self.configuration_id,
            "device": str(self.device),
            "tasks": self.task_progress,
            "gaps": self.knowledge_gaps,
            "sources": list(self.known_sources.keys()),
        }
        return json.dumps(status, indent=2)
