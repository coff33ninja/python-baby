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
import glob # Added for checkpoint loading

class PythonMasterAI(nn.Module):
    MASTER_KEY = "8f9b7f8f6e6c9b9d7e7f9b8f6e6c9b9d7e7f9b8f6e6c9b9d7e7f9b8f6e6c9b9d"

    def __init__(self, vocab_size=16000, n_layers=2, n_heads=4, hidden_size=256,
                 dropout=0.1, dim_feedforward=None, activation="relu", 
                 previous_model_state_dict=None, previous_model_config=None): # Added previous_model_config
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward if dim_feedforward is not None else hidden_size * 4
        self.activation = activation

        # Configuration ID must be generated after all architectural params are set.
        self.recalculate_configuration_id() 
        print(f"Initialized Model Configuration ID: {self.configuration_id}")

        # Initialize the embedding layer
        self.embed = nn.Embedding(self.vocab_size, self.hidden_size)
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
        self.current_dataset_version = None # Added for dataset version tracking

        # Utilize torch for device management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Model must be moved to device BEFORE attempting to load previous_model_state_dict,
        # as previous_model_state_dict might be on CPU or a different device.
        # Or, ensure previous_model_state_dict is mapped to self.device before loading.
        # For simplicity, we build the model on CPU, then load weights, then move to device.

        current_new_model_state_dict_on_cpu = self.state_dict() # Get it while model is on CPU

        if previous_model_state_dict:
            print("Attempting to load weights from previous_model_state_dict (matching layers)...")
            loaded_count = 0
            mismatched_count = 0
            skipped_non_exist = 0
            for name, param_prev in previous_model_state_dict.items():
                if name in current_new_model_state_dict_on_cpu:
                    param_new = current_new_model_state_dict_on_cpu[name]
                    if param_prev.shape == param_new.shape:
                        param_new.copy_(param_prev.clone()) # Use clone to avoid issues if prev dict is from live model
                        print(f"  Loaded weights for matching layer: {name} (Shape: {param_prev.shape})")
                        loaded_count += 1
                    else:
                        print(f"  Shape mismatch for layer: {name}. Previous: {param_prev.shape}, New: {param_new.shape}. Skipped.")
                        mismatched_count += 1
                else:
                    # This case is fine, e.g. layers from a much larger model not present in the new one.
                    skipped_non_exist +=1
            print(f"Weight loading from previous model (matching layers) complete. Loaded: {loaded_count}, Shape Mismatched: {mismatched_count}, Not in New Model: {skipped_non_exist}")

            # --- Seeding New Layers from Existing Layers ---
            if previous_model_config and self.n_layers > previous_model_config.get('n_layers', 0):
                old_n_layers = previous_model_config['n_layers']
                print(f"Seeding new layers ({old_n_layers} to {self.n_layers-1}) from previous model's last layer (layer {old_n_layers-1}).")
                scaling_factor = 0.5 # As specified, can be tuned
                
                # Define parameter names within a standard nn.TransformerEncoderLayer
                # This list needs to be accurate and match the PyTorch implementation.
                # Common parameters:
                # self_attn.in_proj_weight, self_attn.in_proj_bias
                # self_attn.out_proj.weight, self_attn.out_proj.bias
                # linear1.weight, linear1.bias
                # linear2.weight, linear2.bias
                # norm1.weight, norm1.bias
                # norm2.weight, norm2.bias
                param_suffixes = [
                    "self_attn.in_proj_weight", "self_attn.in_proj_bias",
                    "self_attn.out_proj.weight", "self_attn.out_proj.bias",
                    "linear1.weight", "linear1.bias",
                    "linear2.weight", "linear2.bias",
                    "norm1.weight", "norm1.bias",
                    "norm2.weight", "norm2.bias"
                ]

                if old_n_layers > 0: # Can only seed if there was at least one old layer
                    old_last_layer_idx = old_n_layers - 1
                    
                    for new_layer_idx in range(old_n_layers, self.n_layers):
                        print(f"  Initializing new layer {new_layer_idx}:")
                        new_layer_prefix = f"transformer.encoder.layers.{new_layer_idx}."
                        old_last_layer_prefix = f"transformer.encoder.layers.{old_last_layer_idx}."

                        for param_suffix in param_suffixes:
                            old_param_key = old_last_layer_prefix + param_suffix
                            new_param_key = new_layer_prefix + param_suffix

                            if old_param_key in previous_model_state_dict and new_param_key in current_new_model_state_dict_on_cpu:
                                old_param_data = previous_model_state_dict[old_param_key]
                                new_param_data_target = current_new_model_state_dict_on_cpu[new_param_key]
                                
                                if old_param_data.shape == new_param_data_target.shape:
                                    new_param_data_target.data.copy_(old_param_data.data.clone()) # Copy
                                    new_param_data_target.data.mul_(scaling_factor) # Scale
                                    print(f"    Seeded {new_param_key} from {old_param_key} with scaling {scaling_factor}")
                                else:
                                    print(f"    Warning: Shape mismatch for seeding {new_param_key} from {old_param_key}. Old: {old_param_data.shape}, New: {new_param_data_target.shape}. Using default initialization for this param.")
                            else:
                                missing_keys_info = []
                                if not (old_param_key in previous_model_state_dict): missing_keys_info.append(f"old key '{old_param_key}' missing")
                                if not (new_param_key in current_new_model_state_dict_on_cpu): missing_keys_info.append(f"new key '{new_param_key}' missing")
                                print(f"    Warning: Could not seed parameter for suffix '{param_suffix}' ({', '.join(missing_keys_info)}). Using default initialization for this param in new layer {new_layer_idx}.")
                else:
                    print("  Skipping seeding new layers as previous model had no layers (old_n_layers == 0). New layers will use default initialization.")
            elif previous_model_config and self.n_layers <= previous_model_config.get('n_layers', 0) :
                 print("Info: New model does not have more layers than previous. No layer seeding needed.")

        self.to(self.device) # Move the entire model to the target device after potential weight loading

        # Attempt to load the latest checkpoint for this model's NEW configuration.
        # This is generally for resuming training if this specific grown configuration was trained before.
        # Note: If previous_model_state_dict was just used, this usually means it's the first time
        # this grown configuration is created, so it likely won't find a checkpoint unless
        # this exact grown configuration was created and checkpointed before.
        self._try_load_latest_checkpoint()

    def recalculate_configuration_id(self):
        """Recalculates and updates the model's configuration_id."""
        config_params_str = (
            f"v{self.vocab_size}_l{self.n_layers}_h{self.n_heads}_"
            f"hs{self.hidden_size}_d{self.dropout}_df{self.dim_feedforward}_"
            f"a{self.activation}"
        )
        self.configuration_id = hashlib.sha1(config_params_str.encode()).hexdigest()[:12]
        print(f"Recalculated Configuration ID: {self.configuration_id}")


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
            f.write("\n") # Ensure each JSON object is on a new line

    def log_research(self, topic, sources, success, note=""):
        log_entry = {"topic": topic, "sources": sources, "success": success}
        if note:
            log_entry["note"] = note
        self.research_log.append(log_entry)
        with open("research_log.json", "a") as f:
            json.dump(log_entry, f)
            f.write("\n") # Ensure each JSON object is on a new line

    def log_source(self, source, url, score, added):
        self.source_log.append({"source": source, "url": url, "score": score, "added": added})
        with open("source_log.json", "a") as f:
            json.dump({"source": source, "url": url, "score": score, "added": added}, f)
            f.write("\n") # Ensure each JSON object is on a new line

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
        print(f"Debug: Evaluated '{source_name}' ({url}) for query '{query}': R={relevance_score:.2f}, A={authority_score:.2f}, F={freshness_score:.2f} -> Total={total_score:.2f}")
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

    def get_state_for_checkpoint(self):
        """
        Gathers the AI's state for checkpointing.
        For logs, consider saving only recent entries if they become too large.
        """
        # For logs, saving the full log for now.
        # If these grow excessively, consider subsetting, e.g., self.performance_log[-1000:].
        return {
            "stage": self.stage,
            "task_progress": dict(self.task_progress),  # Convert defaultdict to dict for saving
            "knowledge_gaps": list(self.knowledge_gaps), # Convert to list
            "known_sources": self.known_sources,
            "configuration_id": self.configuration_id,
            "vocab_size": self.vocab_size,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "dim_feedforward": self.dim_feedforward,
            "activation": self.activation,
            "performance_log": self.performance_log,
            "research_log": self.research_log,
            "source_log": self.source_log,
            "current_dataset_version": self.current_dataset_version, # Added for checkpointing
        }

    def load_checkpoint(self, filepath, optimizer=None):
        """
        Loads a checkpoint from the given filepath.
        Performs configuration checks before loading state.
        """
        print(f"Attempting to load checkpoint from: {filepath}")
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
        except FileNotFoundError:
            print(f"Checkpoint file not found: {filepath}")
            return False
        except Exception as e:
            print(f"Error loading checkpoint file {filepath}: {e}")
            return False

        # --- Model Configuration Check ---
        ckpt_ai_state = checkpoint.get('ai_state')
        if not ckpt_ai_state:
            print("Error: Checkpoint is missing 'ai_state'. Cannot verify configuration or load.")
            return False

        ckpt_config_id = ckpt_ai_state.get('configuration_id')
        if self.configuration_id != ckpt_config_id:
            print(f"ERROR: Configuration ID mismatch! Model: '{self.configuration_id}', Checkpoint: '{ckpt_config_id}'. Aborting load.")
            return False

        # Parameter names to check
        params_to_check = ['vocab_size', 'n_layers', 'n_heads', 'hidden_size', 'dropout', 'dim_feedforward', 'activation']
        config_mismatch = False
        for param in params_to_check:
            model_param_val = getattr(self, param)
            ckpt_param_val = ckpt_ai_state.get(param)
            if model_param_val != ckpt_param_val:
                print(f"ERROR: Parameter mismatch for '{param}'. Model: {model_param_val}, Checkpoint: {ckpt_param_val}.")
                config_mismatch = True
        
        if config_mismatch:
            print("Aborting checkpoint loading due to model parameter mismatch.")
            return False
        
        print("Checkpoint configuration matches model configuration.")

        # Load model state
        try:
            self.load_state_dict(checkpoint['model_state_dict'])
            print("Model state_dict loaded successfully.")
        except Exception as e:
            print(f"Error loading model state_dict: {e}")
            return False # Critical error

        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state_dict loaded successfully.")
            except Exception as e:
                print(f"Error loading optimizer state_dict: {e}")
                # Non-critical, model state might still be useful

        # Load AI state attributes
        self.stage = ckpt_ai_state.get('stage', self.stage) # Default to current if not in ckpt
        
        # Ensure task_progress is a defaultdict
        loaded_task_progress = ckpt_ai_state.get('task_progress', {})
        self.task_progress = defaultdict(int, loaded_task_progress)
        
        self.knowledge_gaps = ckpt_ai_state.get('knowledge_gaps', [])
        self.known_sources = ckpt_ai_state.get('known_sources', self.load_known_sources()) # Default to fresh load if not in ckpt
        
        # Use .get for logs for backward compatibility
        self.performance_log = ckpt_ai_state.get('performance_log', [])
        self.research_log = ckpt_ai_state.get('research_log', [])
        self.source_log = ckpt_ai_state.get('source_log', [])
        self.current_dataset_version = ckpt_ai_state.get('current_dataset_version', None) # Load dataset version
        
        loaded_epoch = checkpoint.get('epoch', 'N/A')
        print(f"Checkpoint loaded successfully. Resuming at stage '{self.stage}', epoch {loaded_epoch}, dataset version '{self.current_dataset_version}'.")
        return True

    def _try_load_latest_checkpoint(self):
        """
        Tries to load the latest checkpoint for the current model's stage and configuration.
        """
        Tries to load the latest checkpoint for the current model's stage and configuration.
        Returns a status message.
        """
        CHECKPOINT_DIR = "checkpoints" # Should match train.py
        status_message = ""

        if not os.path.exists(CHECKPOINT_DIR):
            status_message = f"Checkpoint directory '{CHECKPOINT_DIR}' not found. Model will start fresh."
            print(status_message)
            return status_message

        expected_filename = f"model_stage_{self.stage}_config_{self.configuration_id}_latest.pt"
        latest_checkpoint_filepath = os.path.join(CHECKPOINT_DIR, expected_filename)

        if os.path.exists(latest_checkpoint_filepath):
            print(f"Latest checkpoint found for current configuration: {latest_checkpoint_filepath}. Attempting to load.")
            # The load_checkpoint method already prints detailed success/failure messages.
            if self.load_checkpoint(latest_checkpoint_filepath):
                # Try to get epoch from the loaded checkpoint.
                # The 'epoch' is stored at the top level of the checkpoint, not in 'ai_state'.
                try:
                    # Temporarily load just to get the epoch without full state change if self.load_checkpoint didn't make it available
                    # However, self.load_checkpoint already loaded the state.
                    # We need a way to access the epoch from the checkpoint that was loaded.
                    # For simplicity, we'll rely on the print from load_checkpoint and just confirm success.
                    # A more robust way would be for load_checkpoint to return the epoch or store it.
                    # Let's assume 'epoch' was part of ai_state for this example, or we enhance load_checkpoint.
                    # For now, the message from load_checkpoint itself is the primary feedback.
                    status_message = f"Successfully loaded checkpoint: {latest_checkpoint_filepath}. Stage: {self.stage}."
                    # To get epoch, we would need load_checkpoint to store it on self, e.g. self.last_loaded_epoch
                    # Or retrieve it from self.task_progress if we decide to store it there.
                    # For now, keeping it simple as load_checkpoint already prints the epoch.
                print(status_message) # Also print to console
                return status_message
            else:
                status_message = f"Found checkpoint {latest_checkpoint_filepath}, but failed to load (e.g., config mismatch or corrupt file). Check console for details."
                print(status_message) # Also print to console
                return status_message
        else:
            status_message = f"No existing checkpoint found for stage '{self.stage}' and config_id '{self.configuration_id}' at '{latest_checkpoint_filepath}'. Model remains in its current state or starts fresh."
            print(status_message) # Also print to console
            return status_message

    def get_config_dict(self):
        """Returns a dictionary of the model's key architectural parameters."""
        return {
            'vocab_size': self.vocab_size,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'hidden_size': self.hidden_size,
            'dropout': self.dropout,
            'dim_feedforward': self.dim_feedforward,
            'activation': self.activation,
            'configuration_id': self.configuration_id # Include for reference
        }

    def generate_for_evaluation(self, prompt_text: str, task_type: str, context_code: str = None) -> str:
        """
        Generates a response from the model for a given evaluation task.
        Uses placeholder logic for now.
        """
        self.eval() # Ensure model is in evaluation mode

        # Placeholder generation logic
        if task_type == "code_generation":
            func_name = "example_func"
            try:
                # Basic attempts to extract a function name from the prompt
                if "function called" in prompt_text:
                    # Handles "Create a Python function called 'my_func' that..."
                    split_parts = prompt_text.split("function called '")
                    if len(split_parts) > 1:
                        func_name = split_parts[1].split("'")[0]
                elif "function named" in prompt_text:
                    # Handles "Create a Python function named 'my_func' that..."
                    split_parts = prompt_text.split("function named '")
                    if len(split_parts) > 1:
                        func_name = split_parts[1].split("'")[0]
                elif "function" in prompt_text and ("takes" in prompt_text or "returns" in prompt_text or "calculates" in prompt_text or "computes" in prompt_text):
                    # Handles "Create a Python function my_func that takes..." or "... function my_func returns..."
                    # This is more heuristic and might need refinement.
                    match = re.search(r"function\s+([\w_]+)", prompt_text)
                    if match:
                        func_name = match.group(1)
            except Exception as e:
                print(f"Note: Crude function name extraction failed for prompt '{prompt_text}': {e}")
            
            # Basic parameter extraction placeholder - very naive
            params_match = re.search(r"takes parameters? ([\w\s,and]+)(?:\s+and|\s+which|\s+that|\.)", prompt_text, re.IGNORECASE)
            params_str = ""
            if params_match:
                raw_params = params_match.group(1).replace(" and ", ", ")
                params_list = [p.strip() for p in raw_params.split(',') if p.strip()]
                params_str = ", ".join(params_list)

            return f"def {func_name}({params_str}):\n  # TODO: Implement based on: {prompt_text}\n  pass"
        
        elif task_type == "code_explanation":
            if context_code:
                return f"This code snippet:\n```python\n{context_code}\n```\nExplanation: The code does something interesting related to: {prompt_text}"
            else:
                return f"Explanation for prompt (no code provided): {prompt_text}"
        
        elif task_type == "docstring_generation":
            # Assume context_code is the function for which to generate a docstring
            if context_code:
                # Attempt to find the function definition to place the docstring correctly
                def_line_match = re.search(r"def\s+[\w_]+\(.*\):", context_code)
                if def_line_match:
                    def_line = def_line_match.group(0)
                    # Insert docstring after the def line
                    docstring = f'  """\n  TODO: Generate docstring based on: {prompt_text}\n  This function might use: {context_code[:100]}...\n  """'
                    return f"{def_line}\n{docstring}\n{context_code[len(def_line):].strip()}"
                else: # Fallback if no clear def line
                    return f'"""\nTODO: Generate docstring based on: {prompt_text}\nThis function might use: {context_code[:100]}...\n"""\n{context_code}'
            else:
                 return f'"""\nTODO: Generate docstring based on: {prompt_text}\n"""'

        elif task_type == "concept_explanation":
            return f"Regarding {prompt_text}: This is an important concept in Python. It involves..."
        
        else:
            # Fallback for unknown task types or if full generation is not ready
            # For a real model, this would involve:
            # combined_input = f"{context_code}\n{prompt_text}" if context_code else prompt_text
            # inputs = self.tokenizer(combined_input, return_tensors="pt", padding=True, truncation=True).to(self.device)
            # generated_ids = self.generate(inputs.input_ids, max_length=150) # Assuming self.generate exists
            # generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            # return generated_text
            return f"Generated placeholder response for task_type '{task_type}' with prompt: {prompt_text}"

    def get_latest_dataset_path(self, stage: str) -> str | None:
        """
        Reads the version timestamp from data/{stage}/latest.txt
        and constructs the path to the latest versioned dataset directory.
        Returns the path string or None if latest.txt or directory is not found.
        """
        stage_data_dir = os.path.join("data", stage)
        latest_txt_path = os.path.join(stage_data_dir, "latest.txt")
        
        version_timestamp = None
        try:
            with open(latest_txt_path, "r") as f:
                version_timestamp = f.read().strip()
        except FileNotFoundError:
            print(f"Info: 'latest.txt' not found in {stage_data_dir}. No dataset version specified.")
            return None
        except Exception as e:
            print(f"Error reading 'latest.txt' in {stage_data_dir}: {e}")
            return None

        if not version_timestamp:
            print(f"Info: 'latest.txt' in {stage_data_dir} is empty. No dataset version specified.")
            return None
            
        dataset_path = os.path.join(stage_data_dir, version_timestamp)
        
        if not os.path.isdir(dataset_path):
            print(f"Error: Dataset directory '{dataset_path}' (specified in latest.txt) does not exist.")
            return None
            
        return dataset_path
