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
from utils import get_config_value # Added for config management

class PythonMasterAI(nn.Module):
    MASTER_KEY = "8f9b7f8f6e6c9b9d7e7f9b8f6e6c9b9d7e7f9b8f6e6c9b9d7e7f9b8f6e6c9b9d"

    def __init__(self, vocab_size=None, n_layers=None, n_heads=None, hidden_size=None,
                 dropout=None, dim_feedforward=None, activation=None,
                 previous_model_state_dict=None, previous_model_config=None):
        super().__init__()

        # Load defaults from config, allowing overrides from constructor arguments
        default_vocab_size = get_config_value('model_defaults.vocab_size', 16000)
        default_n_layers = get_config_value('model_defaults.n_layers', 2)
        default_n_heads = get_config_value('model_defaults.n_heads', 4)
        default_hidden_size = get_config_value('model_defaults.hidden_size', 256)
        default_dropout = get_config_value('model_defaults.dropout', 0.1)
        default_activation = get_config_value('model_defaults.activation', "relu")
        default_ff_factor = get_config_value('model_defaults.dim_feedforward_factor', 4)

        self.vocab_size = vocab_size if vocab_size is not None else default_vocab_size
        self.n_layers = n_layers if n_layers is not None else default_n_layers
        self.n_heads = n_heads if n_heads is not None else default_n_heads
        self.hidden_size = hidden_size if hidden_size is not None else default_hidden_size
        self.dropout = dropout if dropout is not None else default_dropout
        self.activation = activation if activation is not None else default_activation

        if dim_feedforward is not None:
            self.dim_feedforward = dim_feedforward
        else:
            # Use the hidden_size that was set (either from arg or default)
            self.dim_feedforward = self.hidden_size * default_ff_factor

        self.recalculate_configuration_id()
        print(f"Initialized Model Configuration ID: {self.configuration_id}")

        self.embed = nn.Embedding(self.vocab_size, self.hidden_size)
        self.transformer = nn.Transformer(
            d_model=self.hidden_size,
            nhead=self.n_heads,
            num_encoder_layers=self.n_layers,
            num_decoder_layers=0,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

        self.performance_log = []
        self.research_log = []
        self.source_log = []
        self.stage = "baby"
        self.growth_tasks = self.define_growth_tasks()
        self.task_progress = defaultdict(int)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Tokenizer pad_token set to eos_token.")

        self.knowledge_gaps = []
        self.known_sources = self.load_known_sources()
        self.current_dataset_version = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        current_new_model_state_dict_on_cpu = self.state_dict()

        if previous_model_state_dict:
            print("Attempting to load weights from previous_model_state_dict (matching layers)...")
            loaded_count = 0
            mismatched_count = 0
            skipped_non_exist = 0
            for name, param_prev in previous_model_state_dict.items():
                if name in current_new_model_state_dict_on_cpu:
                    param_new = current_new_model_state_dict_on_cpu[name]
                    if param_prev.shape == param_new.shape:
                        param_new.copy_(param_prev.clone())
                        print(f"  Loaded weights for matching layer: {name} (Shape: {param_prev.shape})")
                        loaded_count += 1
                    else:
                        print(f"  Shape mismatch for layer: {name}. Previous: {param_prev.shape}, New: {param_new.shape}. Skipped.")
                        mismatched_count += 1
                else:
                    skipped_non_exist +=1
            print(f"Weight loading from previous model (matching layers) complete. Loaded: {loaded_count}, Shape Mismatched: {mismatched_count}, Not in New Model: {skipped_non_exist}")

            if previous_model_config and self.n_layers > previous_model_config.get('n_layers', 0):
                old_n_layers = previous_model_config['n_layers']
                print(f"Seeding new layers ({old_n_layers} to {self.n_layers-1}) from previous model's last layer (layer {old_n_layers-1}).")
                scaling_factor = 0.5
                param_suffixes = [
                    "self_attn.in_proj_weight", "self_attn.in_proj_bias",
                    "self_attn.out_proj.weight", "self_attn.out_proj.bias",
                    "linear1.weight", "linear1.bias",
                    "linear2.weight", "linear2.bias",
                    "norm1.weight", "norm1.bias",
                    "norm2.weight", "norm2.bias"
                ]
                if old_n_layers > 0:
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
                                    new_param_data_target.data.copy_(old_param_data.data.clone())
                                    new_param_data_target.data.mul_(scaling_factor)
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

        self.to(self.device)
        self._try_load_latest_checkpoint()

    def recalculate_configuration_id(self):
        config_params_str = (
            f"v{self.vocab_size}_l{self.n_layers}_h{self.n_heads}_"
            f"hs{self.hidden_size}_d{self.dropout}_df{self.dim_feedforward}_"
            f"a{self.activation}"
        )
        self.configuration_id = hashlib.sha1(config_params_str.encode()).hexdigest()[:12]
        print(f"Recalculated Configuration ID: {self.configuration_id}")

    def forward(self, x, src_key_padding_mask=None):
        embedded_src = self.embed(x)
        if self.n_layers > 0:
            transformer_output = self.transformer.encoder(embedded_src, src_key_padding_mask=src_key_padding_mask)
        else:
            transformer_output = embedded_src
        output = self.fc(transformer_output)
        return output

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 max_new_tokens: int = 100, eos_token_id: int = None,
                 temperature: float = 0.7, top_k: int = 50) -> torch.Tensor:
        self.eval()
        if eos_token_id is None and self.tokenizer.eos_token_id is not None:
            eos_token_id = self.tokenizer.eos_token_id
        generated_ids = input_ids
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.forward(generated_ids, src_key_padding_mask=attention_mask)
                next_token_logits = outputs[:, -1, :]
                if temperature > 0.001:
                    next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    eff_top_k = min(top_k, next_token_logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, eff_top_k, dim=-1)
                    top_k_probs = torch.softmax(top_k_logits, dim=-1)
                    next_token_index_in_top_k = torch.multinomial(top_k_probs, num_samples=1)
                    next_token_id = torch.gather(top_k_indices, -1, next_token_index_in_top_k)
                else:
                    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                if attention_mask is not None:
                    new_attention_mask_token = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=self.device)
                    attention_mask = torch.cat([attention_mask, new_attention_mask_token], dim=1)
                if eos_token_id is not None and (next_token_id == eos_token_id).all():
                    break
        return generated_ids

    def generate_for_evaluation(self, prompt_text: str, task_type: str,
                                context_code: str = None, max_gen_length: int = 150) -> str:
        self.eval()
        full_prompt = prompt_text
        if task_type == "code_explanation":
            if context_code:
                full_prompt = f"You are PythonMasterAI. Explain the following Python code based on the request.\n\nCode:\n```python\n{context_code}\n```\n\nRequest: {prompt_text}\n\nExplanation:"
            else:
                full_prompt = f"You are PythonMasterAI. Provide an explanation for the following concept/request: {prompt_text}\n\nExplanation:"
        elif task_type == "docstring_generation":
            if context_code:
                full_prompt = f"You are PythonMasterAI. Generate a Python docstring for the following function. {prompt_text}\n\nFunction:\n```python\n{context_code}\n```\n\nDocstring:"
            else:
                full_prompt = f"You are PythonMasterAI. Generate a Python docstring based on the following description: {prompt_text}\n\nDocstring:"
        elif task_type == "code_generation":
            full_prompt = f"You are PythonMasterAI. Generate Python code for the following request: {prompt_text}\n\nCode:"
        elif task_type == "concept_explanation":
            full_prompt = f"You are PythonMasterAI. Explain the following Python concept: {prompt_text}\n\nExplanation:"
        else:
            full_prompt = f"You are PythonMasterAI. Respond to the following: {prompt_text}"
        tokenizer_max_input_len = 512
        inputs = self.tokenizer(full_prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=tokenizer_max_input_len)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        eos_token_id_to_use = self.tokenizer.eos_token_id
        if eos_token_id_to_use is None:
            print("Warning: Tokenizer does not have a default eos_token_id. Generation might run to max_gen_length.")
        output_ids = self.generate(
            input_ids,
            attention_mask,
            max_new_tokens=max_gen_length,
            eos_token_id=eos_token_id_to_use,
            temperature=0.7,
            top_k=50
        )
        generated_token_ids = output_ids[0, input_ids.size(1):]
        generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        return generated_text.strip()

    def log_performance(self, metric, value):
        self.performance_log.append((metric, value))
        with open("performance_log.json", "a") as f:
            json.dump({"metric": metric, "value": value}, f)
            f.write("\n")

    def log_research(self, topic, sources, success, note=""):
        log_entry = {"topic": topic, "sources": sources, "success": success}
        if note:
            log_entry["note"] = note
        self.research_log.append(log_entry)
        with open("research_log.json", "a") as f:
            json.dump(log_entry, f)
            f.write("\n")

    def log_source(self, source, url, score, added):
        self.source_log.append({"source": source, "url": url, "score": score, "added": added})
        with open("source_log.json", "a") as f:
            json.dump({"source": source, "url": url, "score": score, "added": added}, f)
            f.write("\n")

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
        parsed_url = urlparse(repo_url_or_api_url)
        api_url_to_use = None
        if parsed_url.netloc == "api.github.com" and parsed_url.path.startswith("/repos/"):
            api_url_to_use = repo_url_or_api_url
        elif parsed_url.netloc == "github.com":
            path_parts = parsed_url.path.strip('/').split('/')
            if len(path_parts) >= 2:
                owner, repo_name_segment = path_parts[0], path_parts[1]
                repo_name = repo_name_segment.split('/')[0]
                api_url_to_use = f"https://api.github.com/repos/{owner}/{repo_name}"
        if not api_url_to_use: return None
        try:
            response = requests.get(api_url_to_use, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Warning: Error fetching GitHub repo details for {api_url_to_use}: {e}")
            return None

    def _fetch_pypi_package_info(self, package_name_or_url):
        package_name = ""
        parsed_uri = urlparse(package_name_or_url)
        if parsed_uri.netloc == "pypi.org":
            path_segments = parsed_uri.path.strip('/').split('/')
            if len(path_segments) > 1:
                if path_segments[0] == "project": package_name = path_segments[1]
                elif path_segments[0] == "pypi": package_name = path_segments[1]
        elif not parsed_uri.scheme and not parsed_uri.netloc:
            package_name = package_name_or_url.strip()
        if not package_name: return None
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
        if "github" in source: return len(content) > 10 and "python" in content.lower()
        if "stackoverflow" in source: return len(content.split("\n")) > 1
        if "pypi" in source or "python_docs" in source: return len(content) > 50
        return True

    def get_research_scrape_targets(self):
        queries = self.formulate_research_queries()
        scrape_targets_dict = {}
        for query in queries:
            selected_sources_for_query = self.select_research_sources(query)
            for src_name in selected_sources_for_query:
                if src_name in self.known_sources and self.known_sources[src_name]:
                    scrape_targets_dict[src_name] = self.known_sources[src_name][0]
                else:
                    print(f"Warning: Source '{src_name}' for query '{query}' not in known_sources or has no URL for research.")
        return list(scrape_targets_dict.items())

    def process_scraped_research_data(self, stage):
        queries = self.formulate_research_queries()
        if not queries:
            print("No active research queries to process post-scraping.")
            return
        query_resolution_map = {query: False for query in queries}
        all_relevant_sources_for_queries = set()
        for query in queries:
            all_relevant_sources_for_queries.update(self.select_research_sources(query))
        for source_name in all_relevant_sources_for_queries:
            latest_dataset_dir = self.get_latest_dataset_path(stage)
            if not latest_dataset_dir:
                print(f"Warning: Cannot process research data for stage '{stage}' as no latest dataset directory found.")
                return
            file_path = os.path.join(latest_dataset_dir, f"{source_name}.txt")

            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f: content = f.read()
                is_content_valid = self.validate_research_data(content, source_name)
                for query in queries:
                    if source_name in self.select_research_sources(query):
                        if is_content_valid:
                            if not query_resolution_map[query]:
                                self.log_research(query, [source_name], success=True)
                                query_resolution_map[query] = True
                                research_task_key = next((k for k in self.growth_tasks[self.stage] if k.startswith("research_")), None)
                                if research_task_key: self.log_task_progress(research_task_key)
            else:
                for query in queries:
                    if source_name in self.select_research_sources(query) and not query_resolution_map[query]:
                        self.log_research(query, [source_name], success=False, note="Scraped file not found")
                        print(f"Warning: Scraped file {file_path} not found for source {source_name} relevant to query '{query}'.")
        resolved_gaps_this_cycle = set()
        for query, resolved in query_resolution_map.items():
            if resolved:
                for gap_text in self.knowledge_gaps:
                    if gap_text.lower() in query.lower():
                        resolved_gaps_this_cycle.add(gap_text)
        if resolved_gaps_this_cycle:
            print(f"Knowledge gaps resolved this cycle: {resolved_gaps_this_cycle}")
            self.knowledge_gaps = [gap for gap in self.knowledge_gaps if gap not in resolved_gaps_this_cycle]
        if queries and not self.knowledge_gaps: print("All knowledge gaps from this research cycle appear to be resolved.")
        elif queries: print(f"Remaining knowledge gaps after research: {self.knowledge_gaps}")

    def conduct_research(self):
        print("Conduct_research called. Identifying targets...")
        research_targets = self.get_research_scrape_targets()
        if research_targets:
            print(f"Conduct_research initiating focused scraping for targets: {research_targets}")
            from scrape_data import scrape_data
            sources_to_scrape, urls_to_scrape = zip(*research_targets)
            scrape_data(self.stage, list(sources_to_scrape), list(urls_to_scrape))
        else:
            print("No specific research targets identified by conduct_research. Relying on general scraping or previously scraped data.")
        self.process_scraped_research_data(self.stage)

    def discover_new_sources(self):
        queries = self.formulate_research_queries()
        new_sources = []
        for query in queries:
            candidate_sources = self.search_for_sources(query)
            for source, url in candidate_sources:
                score = self.evaluate_source(source, url, query)
                if score > 0.7:
                    self.known_sources[source] = [url]
                    self.log_source(source, url, score, added=True)
                    new_sources.append(source)
                    self.log_task_progress("find_sources")
                else:
                    self.log_source(source, url, score, added=False)
        print(f"Discovered sources: {new_sources}")
        return new_sources

    def search_for_sources(self, query):
        candidates = []
        if self.stage == "baby": candidates.append(("python_blog", "https://python-blog.example.com"))
        elif self.stage == "toddler": candidates.append(("python_tutorials", "https://tutorials.python.org"))
        elif self.stage == "teenager": candidates.append(("python_forum", "https://forum.python.org"))
        else: candidates.append(("python_conference", "https://pycon.org"))
        return candidates

    def evaluate_source(self, source_name, url, query):
        relevance_score = 0.0; authority_score = 0.0; freshness_score = 0.0
        query_lower = query.lower(); query_terms = query_lower.split(); relevance_points = 0
        if any(term in source_name.lower() for term in query_terms): relevance_points += 1
        if any(term in url.lower() for term in query_terms): relevance_points += 1
        if "python" in source_name.lower() or "python" in url.lower(): relevance_points += 0.5
        relevance_score = min(relevance_points / 2.5, 1.0) * 0.4 if relevance_points > 0 else 0.05
        parsed_url = urlparse(url); domain = parsed_url.netloc.lower()
        github_details = None; pypi_details = None
        if "github.com" in domain or "api.github.com" in domain: github_details = self._fetch_github_repo_details(url)
        elif "pypi.org" in domain: pypi_details = self._fetch_pypi_package_info(url)
        if github_details and 'stargazers_count' in github_details:
            stars = github_details['stargazers_count']
            if stars >= 1000: authority_score = 0.3
            elif stars >= 100: authority_score = 0.2
            else: authority_score = 0.1
        elif pypi_details and 'info' in pypi_details and 'version' in pypi_details['info']: authority_score = 0.25
        elif "stackoverflow.com" in domain: authority_score = 0.25
        elif "docs.python.org" in domain or "peps.python.org" in domain: authority_score = 0.3
        elif any(known_good_domain in domain for known_good_domain in ["realpython.com", "djangoproject.com", "flask.palletsprojects.com"]): authority_score = 0.25
        else: authority_score = 0.1
        current_time_ts = time.time(); freshness_score = 0.05
        if github_details and 'pushed_at' in github_details:
            last_push_date_str = github_details['pushed_at']
            try:
                last_push_dt = datetime.fromisoformat(last_push_date_str.replace('Z', '+00:00')); last_push_ts = last_push_dt.timestamp()
                age_days = (current_time_ts - last_push_ts) / (60 * 60 * 24)
                if age_days <= 30: freshness_score = 0.3
                elif age_days <= 180: freshness_score = 0.2
                elif age_days <= 365: freshness_score = 0.15
                else: freshness_score = 0.05
            except ValueError: print(f"Warning: Could not parse GitHub date: {last_push_date_str}")
        elif pypi_details and 'releases' in pypi_details and pypi_details['releases']:
            latest_version_str = pypi_details.get('info', {}).get('version')
            if latest_version_str and latest_version_str in pypi_details['releases']:
                release_dates = [item['upload_time_iso_8601'] for item in pypi_details['releases'][latest_version_str] if 'upload_time_iso_8601' in item]
                if release_dates:
                    latest_upload_time_str = max(release_dates)
                    try:
                        upload_dt = datetime.fromisoformat(latest_upload_time_str.replace('Z', '+00:00')); upload_ts = upload_dt.timestamp()
                        age_days = (current_time_ts - upload_ts) / (60 * 60 * 24)
                        if age_days <= 90: freshness_score = 0.3
                        elif age_days <= 365: freshness_score = 0.2
                        else: freshness_score = 0.1
                    except ValueError: print(f"Warning: Could not parse PyPI date: {latest_upload_time_str}")
        total_score = relevance_score + authority_score + freshness_score
        print(f"Debug: Evaluated '{source_name}' ({url}) for query '{query}': R={relevance_score:.2f}, A={authority_score:.2f}, F={freshness_score:.2f} -> Total={total_score:.2f}")
        return min(total_score, 1.0)

    def prioritize_scraping(self):
        all_sources_for_stage = []
        if self.stage == "baby": all_sources_for_stage = [s for s in self.known_sources.keys() if "beginner" in s or "study_guides" in s or "stackoverflow_basic" in s]
        elif self.stage == "toddler": all_sources_for_stage = [s for s in self.known_sources.keys() if "intermediate" in s or "pypi_docs" in s or "real_python" in s]
        elif self.stage == "teenager": all_sources_for_stage = [s for s in self.known_sources.keys() if "advanced" in s or "peps" in s or "reddit_learnpython" in s]
        else: all_sources_for_stage = [s for s in self.known_sources.keys() if "trending" in s or "python_docs" in s or "peps" in s]
        if "pypi_docs" not in all_sources_for_stage and "pypi_docs" in self.known_sources:
            if self.stage in ["baby", "toddler"]: all_sources_for_stage.append("pypi_docs")
        valid_sources = [s for s in all_sources_for_stage if s in self.known_sources and self.known_sources[s]]
        return valid_sources

    def generate_response(self, input_text):
        if self.assess_performance()["needs_research"]:
            self.conduct_research()
            self.discover_new_sources()
        if "write" in input_text.lower():
            task = "write_functions" if self.stage == "baby" else "write_classes" if self.stage == "toddler" else "write_complex_scripts"
            code = self.generate_for_evaluation(prompt_text=input_text, task_type="code_generation", max_gen_length=200)
            success = self.validate_code(code)
            self.log_task_progress(task, success)
            self.identify_knowledge_gaps(input_text, task, success)
            return f"{self.stage.capitalize()} AI: Generated code:\n```python\n{code}\n```"
        elif "explain" in input_text.lower():
            task = "explain_variables" if self.stage == "baby" else "explain_classes"
            explanation = self.generate_for_evaluation(prompt_text=input_text, task_type="concept_explanation", max_gen_length=150)
            self.log_task_progress(task)
            self.identify_knowledge_gaps(input_text, task, True)
            return f"{self.stage.capitalize()} AI: {explanation}"
        general_response = self.generate_for_evaluation(prompt_text=input_text, task_type="general", max_gen_length=100)
        return f"{self.stage.capitalize()} AI: {general_response}"

    def generate_code(self, input_text):
        print("Warning: generate_code() is deprecated. Use generate_for_evaluation() with task_type='code_generation'.")
        return self.generate_for_evaluation(input_text, "code_generation", max_gen_length=200)

    def generate_explanation(self, input_text):
        print("Warning: generate_explanation() is deprecated. Use generate_for_evaluation() with task_type='concept_explanation'.")
        return self.generate_for_evaluation(input_text, "concept_explanation", max_gen_length=150)

    def debug_code(self, input_text):
        print("Warning: debug_code() is deprecated and not fully implemented in generate_for_evaluation.")
        return self.generate_for_evaluation(input_text, "debug_code_placeholder", max_gen_length=100)

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
        status = {"stage": self.stage, "parameters": f"{params:,}", "configuration_id": self.configuration_id, "device": str(self.device), "tasks": self.task_progress, "gaps": self.knowledge_gaps, "sources": list(self.known_sources.keys())}
        return json.dumps(status, indent=2)

    def get_state_for_checkpoint(self):
        return {"stage": self.stage, "task_progress": dict(self.task_progress), "knowledge_gaps": list(self.knowledge_gaps), "known_sources": self.known_sources, "configuration_id": self.configuration_id, "vocab_size": self.vocab_size, "n_layers": self.n_layers, "n_heads": self.n_heads, "hidden_size": self.hidden_size, "dropout": self.dropout, "dim_feedforward": self.dim_feedforward, "activation": self.activation, "performance_log": self.performance_log, "research_log": self.research_log, "source_log": self.source_log, "current_dataset_version": self.current_dataset_version}

    def load_checkpoint(self, filepath, optimizer=None):
        print(f"Attempting to load checkpoint from: {filepath}")
        try: checkpoint = torch.load(filepath, map_location=self.device)
        except FileNotFoundError: print(f"Checkpoint file not found: {filepath}"); return False
        except Exception as e: print(f"Error loading checkpoint file {filepath}: {e}"); return False
        ckpt_ai_state = checkpoint.get('ai_state')
        if not ckpt_ai_state: print("Error: Checkpoint is missing 'ai_state'. Cannot verify configuration or load."); return False
        ckpt_config_id = ckpt_ai_state.get('configuration_id')
        if self.configuration_id != ckpt_config_id: print(f"ERROR: Configuration ID mismatch! Model: '{self.configuration_id}', Checkpoint: '{ckpt_config_id}'. Aborting load."); return False
        params_to_check = ['vocab_size', 'n_layers', 'n_heads', 'hidden_size', 'dropout', 'dim_feedforward', 'activation']
        config_mismatch = False
        for param in params_to_check:
            model_param_val = getattr(self, param); ckpt_param_val = ckpt_ai_state.get(param)
            if model_param_val != ckpt_param_val: print(f"ERROR: Parameter mismatch for '{param}'. Model: {model_param_val}, Checkpoint: {ckpt_param_val}."); config_mismatch = True
        if config_mismatch: print("Aborting checkpoint loading due to model parameter mismatch."); return False
        print("Checkpoint configuration matches model configuration.")
        try: self.load_state_dict(checkpoint['model_state_dict']); print("Model state_dict loaded successfully.")
        except Exception as e: print(f"Error loading model state_dict: {e}"); return False
        if optimizer and 'optimizer_state_dict' in checkpoint:
            try: optimizer.load_state_dict(checkpoint['optimizer_state_dict']); print("Optimizer state_dict loaded successfully.")
            except Exception as e: print(f"Error loading optimizer state_dict: {e}")
        self.stage = ckpt_ai_state.get('stage', self.stage)
        self.task_progress = defaultdict(int, ckpt_ai_state.get('task_progress', {}))
        self.knowledge_gaps = ckpt_ai_state.get('knowledge_gaps', [])
        self.known_sources = ckpt_ai_state.get('known_sources', self.load_known_sources())
        self.performance_log = ckpt_ai_state.get('performance_log', [])
        self.research_log = ckpt_ai_state.get('research_log', [])
        self.source_log = ckpt_ai_state.get('source_log', [])
        self.current_dataset_version = ckpt_ai_state.get('current_dataset_version', None)
        loaded_epoch = checkpoint.get('epoch', 'N/A')
        print(f"Checkpoint loaded successfully. Resuming at stage '{self.stage}', epoch {loaded_epoch}, dataset version '{self.current_dataset_version}'.")
        return True

    def _try_load_latest_checkpoint(self):
        # Corrected: Use get_config_value for checkpoint_dir
        checkpoint_dir_from_config = get_config_value('checkpointing.checkpoint_dir', "checkpoints")
        status_message = ""
        if not os.path.exists(checkpoint_dir_from_config):
            status_message = f"Checkpoint directory '{checkpoint_dir_from_config}' not found. Model will start fresh."
            print(status_message); return status_message

        expected_filename = f"model_stage_{self.stage}_config_{self.configuration_id}_latest.pt"
        latest_checkpoint_filepath = os.path.join(checkpoint_dir_from_config, expected_filename)

        if os.path.exists(latest_checkpoint_filepath):
            print(f"Latest checkpoint found for current configuration: {latest_checkpoint_filepath}. Attempting to load.")
            if self.load_checkpoint(latest_checkpoint_filepath):
                try: status_message = f"Successfully loaded checkpoint: {latest_checkpoint_filepath}. Stage: {self.stage}."
                except Exception: pass
                print(status_message); return status_message
            else: status_message = f"Found checkpoint {latest_checkpoint_filepath}, but failed to load. Check console for details."
            print(status_message); return status_message
        else:
            status_message = f"No existing checkpoint for stage '{self.stage}' and config '{self.configuration_id}' at '{latest_checkpoint_filepath}'. Model starts fresh."
            print(status_message); return status_message

    def get_config_dict(self):
        return {'vocab_size': self.vocab_size, 'n_layers': self.n_layers, 'n_heads': self.n_heads, 'hidden_size': self.hidden_size, 'dropout': self.dropout, 'dim_feedforward': self.dim_feedforward, 'activation': self.activation, 'configuration_id': self.configuration_id}

    def get_latest_dataset_path(self, stage: str) -> str | None:
        stage_data_dir = os.path.join("data", stage); latest_txt_path = os.path.join(stage_data_dir, "latest.txt"); version_timestamp = None
        try:
            with open(latest_txt_path, "r") as f: version_timestamp = f.read().strip()
        except FileNotFoundError: print(f"Info: 'latest.txt' not found in {stage_data_dir}. No dataset version specified."); return None
        except Exception as e: print(f"Error reading 'latest.txt' in {stage_data_dir}: {e}"); return None
        if not version_timestamp: print(f"Info: 'latest.txt' in {stage_data_dir} is empty. No dataset version specified."); return None
        dataset_path = os.path.join(stage_data_dir, version_timestamp)
        if not os.path.isdir(dataset_path): print(f"Error: Dataset directory '{dataset_path}' (specified in latest.txt) does not exist."); return None
        return dataset_path
