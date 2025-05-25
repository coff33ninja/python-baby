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
from datetime import datetime, timezone
import os # type: ignore
import glob # Added for checkpoint loading
from utils import get_config_value, get_typed_config_value # Updated import
from typing import Dict, List, Optional, TypeVar, Type, cast # Added for Optional type hint and helper
import logging

# --- Initialize logger for this module ---
logger = logging.getLogger(__name__)

API_CONFIG_FILE_PATH = "api_config.json"
class PythonMasterAI(nn.Module):
    MASTER_KEY = "8f9b7f8f6e6c9b9d7e7f9b8f6e6c9b9d7e7f9b8f6e6c9b9d7e7f9b8f6e6c9b9d"

    def __init__(self,
                 vocab_size: Optional[int] = None,
                 n_layers: Optional[int] = None,
                 n_heads: Optional[int] = None,
                 hidden_size: Optional[int] = None,
                 dropout: Optional[float] = None,
                 dim_feedforward: Optional[int] = None,
                 activation: Optional[str] = None,
                 previous_model_state_dict=None, previous_model_config=None):
        super().__init__()

        # Load defaults from config, allowing overrides from constructor arguments
        config_vocab_size = get_typed_config_value('model_defaults.vocab_size', 16000, int)
        config_n_layers = get_typed_config_value('model_defaults.n_layers', 2, int)
        config_n_heads = get_typed_config_value('model_defaults.n_heads', 4, int)
        config_hidden_size = get_typed_config_value('model_defaults.hidden_size', 256, int)
        config_dropout = get_typed_config_value('model_defaults.dropout', 0.1, float)
        config_activation = get_typed_config_value('model_defaults.activation', "relu", str)
        config_ff_factor = get_typed_config_value('model_defaults.dim_feedforward_factor', 4, int)
        self.max_log_entries = get_typed_config_value('logging.max_in_memory_log_entries', 1000, int)

        # Assign attributes, ensuring correct types
        self.vocab_size: int = int(vocab_size) if vocab_size is not None else config_vocab_size
        self.n_layers: int = int(n_layers) if n_layers is not None else config_n_layers
        self.n_heads: int = int(n_heads) if n_heads is not None else config_n_heads
        self.hidden_size: int = int(hidden_size) if hidden_size is not None else config_hidden_size
        self.dropout: float = float(dropout) if dropout is not None else config_dropout
        self.activation: str = str(activation) if activation is not None else config_activation

        # Ensure ff_factor is int/float before multiplication
        _ff_factor = config_ff_factor

        if dim_feedforward is not None:
            self.dim_feedforward: int = int(dim_feedforward)
        else:
            # Use the hidden_size that was set (either from arg or default)
            self.dim_feedforward: int = self.hidden_size * _ff_factor
        self.recalculate_configuration_id()
        logger.info(f"Initialized Model Configuration ID: {self.configuration_id}")

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
        self.growth_tasks = self.define_growth_tasks() # Defines unit_test_accuracy as float
        self.task_progress = defaultdict(float) # Changed from int to float
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Tokenizer pad_token set to eos_token.")

        self.knowledge_gaps: list = []  # Stores identified knowledge gaps
        self.known_sources: Dict[str, Dict] = (
            {}
        )  # Stores evaluated sources: {url: {details}}
        self.initial_source_categories: Dict[str, List[str]] = (
            self._load_initial_source_categories()
        )  # Predefined categories and seed URLs/queries
        self.current_dataset_version = None
        self.api_config = None # For external API configurations
        self._load_api_config() # Load external API configurations

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        current_new_model_state_dict_on_cpu = self.state_dict()
        if previous_model_state_dict:
            logger.info("Attempting to load weights from previous_model_state_dict (matching layers)...")
            # Create a dictionary of weights to load, filtering from previous_model_state_dict
            # for keys that exist in the current model and have matching shapes.
            state_dict_to_load_for_matching_layers = {}
            loaded_count = 0
            mismatched_count = 0
            skipped_non_exist = 0

            for name, param_prev in previous_model_state_dict.items():
                if name in current_new_model_state_dict_on_cpu: # Check against the new model's structure
                    if current_new_model_state_dict_on_cpu[name].shape == param_prev.shape:
                        state_dict_to_load_for_matching_layers[name] = param_prev.clone() # Clone to avoid issues
                        logger.debug(f"  Loaded weights for matching layer: {name} (Shape: {param_prev.shape})")
                        loaded_count += 1
                    else:
                        logger.warning(f"  Shape mismatch for layer: {name}. Previous: {param_prev.shape}, New: {current_new_model_state_dict_on_cpu[name].shape}. Skipped for direct load.")
                        mismatched_count += 1
                else:
                    skipped_non_exist += 1 # Corrected: Increment if key from old_dict is not in new_dict

            if state_dict_to_load_for_matching_layers:
                self.load_state_dict(state_dict_to_load_for_matching_layers, strict=False)
                logger.info(f"Direct weight loading from previous model (matching layers) complete. Loaded: {loaded_count}, Shape Mismatched (skipped direct): {mismatched_count}, Not in New Model (skipped direct): {skipped_non_exist}")

            prev_n_layers_from_config = 0 # Default if not found or invalid
            if previous_model_config:
                val = previous_model_config.get('n_layers')
                if isinstance(val, int):
                    prev_n_layers_from_config = val
                elif val is not None: # It exists but is not int
                    try:
                        prev_n_layers_from_config = int(val)
                        logger.warning(f"'n_layers' in previous_model_config was '{val}', converted to int: {prev_n_layers_from_config}.")
                    except (ValueError, TypeError):
                        logger.warning(f"'n_layers' in previous_model_config ('{val}') is not a valid integer. Using 0 for comparison.")
                        prev_n_layers_from_config = 0

            if previous_model_config and self.n_layers > prev_n_layers_from_config:
                old_n_layers = prev_n_layers_from_config # This is now an int
                logger.info(f"Seeding new layers ({old_n_layers} to {self.n_layers-1}) from previous model's last layer (layer {old_n_layers-1}).")
                scaling_factor = 0.5
                param_suffixes = [
                    "self_attn.in_proj_weight", "self_attn.in_proj_bias",
                    "self_attn.out_proj.weight", "self_attn.out_proj.bias",
                    "linear1.weight", "linear1.bias", "norm1.weight", "norm1.bias",
                    "linear2.weight", "linear2.bias", "norm2.weight", "norm2.bias",
                ]
                if old_n_layers > 0:
                    old_last_layer_idx = old_n_layers - 1
                    for new_layer_idx in range(old_n_layers, self.n_layers):
                        logger.debug(f"  Initializing new layer {new_layer_idx}:")
                        new_layer_module = self.transformer.encoder.layers[new_layer_idx]
                        old_last_layer_module = self.transformer.encoder.layers[old_last_layer_idx] # Source of truth for weights

                        for param_suffix in param_suffixes:
                            try:
                                # Get the actual parameter tensor from the old layer module
                                # For nested attributes like self_attn.in_proj_weight:
                                parts = param_suffix.split('.')
                                old_attr = old_last_layer_module
                                new_attr = new_layer_module
                                for part in parts[:-1]:
                                    old_attr = getattr(old_attr, part)
                                for part in parts[:-1]:
                                    new_attr = getattr(new_attr, part)
                                old_param_data = getattr(old_attr, parts[-1]).data
                                new_param_to_update = getattr(new_attr, parts[-1])

                                if old_param_data.shape == new_param_to_update.data.shape:
                                    new_param_to_update.data.copy_(old_param_data.clone())
                                    new_param_to_update.data.mul_(scaling_factor)
                                    logger.debug(f"    Seeded {param_suffix} in new layer {new_layer_idx} from old layer {old_last_layer_idx} with scaling {scaling_factor}")
                                else:
                                    logger.warning(f"    Shape mismatch for seeding {param_suffix} in new layer {new_layer_idx}. Old: {old_param_data.shape}, New: {new_param_to_update.data.shape}. Using default init.")
                            except AttributeError:
                                logger.warning(f"    Attribute error while trying to seed {param_suffix} for new layer {new_layer_idx}. Using default init for this param.")
                else:
                    logger.info("  Skipping seeding new layers as previous model had no layers (old_n_layers == 0). New layers will use default initialization.")
            elif previous_model_config and self.n_layers <= previous_model_config.get('n_layers', 0) :
                logger.info("New model does not have more layers than previous. No layer seeding needed.")

        self.to(self.device)
        self._try_load_latest_checkpoint()

    def _load_api_config(self):
        """Loads API configurations from api_config.json."""
        if os.path.exists(API_CONFIG_FILE_PATH):
            try:
                with open(API_CONFIG_FILE_PATH, "r") as f:
                    self.api_config = json.load(f)
                logger.info(f"Successfully loaded API configuration from {API_CONFIG_FILE_PATH}.")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {API_CONFIG_FILE_PATH}: {e}. API features might be limited.")
                self.api_config = {}
            except Exception as e:
                logger.error(f"Error reading API config file {API_CONFIG_FILE_PATH}: {e}. API features might be limited.")
                self.api_config = {}
        else:
            logger.warning(
                f"API configuration file {API_CONFIG_FILE_PATH} not found. Real-time source discovery will be limited to hardcoded sources."
            )
            self.api_config = {}
    def recalculate_configuration_id(self):
        config_params_str = (
            f"v{self.vocab_size}_l{self.n_layers}_h{self.n_heads}_"
            f"hs{self.hidden_size}_d{self.dropout}_df{self.dim_feedforward}_"
            f"a{self.activation}"
        )
        self.configuration_id = hashlib.sha1(config_params_str.encode(), usedforsecurity=False).hexdigest()[:12]
        logger.info(f"Recalculated Configuration ID: {self.configuration_id}")

    def forward(self, x, src_key_padding_mask=None):
        embedded_src = self.embed(x)
        if self.n_layers > 0:
            transformer_output = self.transformer.encoder(embedded_src, src_key_padding_mask=src_key_padding_mask)
        else:
            transformer_output = embedded_src
        output = self.fc(transformer_output)
        return output

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 max_new_tokens: int = 100, eos_token_id: Optional[int] = None,
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
                                context_code: Optional[str] = None, max_gen_length: int = 150) -> str: # Allow context_code to be Optional
        self.eval()
        tokenizer_max_input_len = get_typed_config_value(
            "evaluation.tokenizer_max_input_length", 512, int
        )

        # Ensure prompt_text used in f-strings is a string, defaulting to empty if None was passed
        # (though current type hint is str, this makes it robust if None slips through or if hint changes)
        _prompt_text_str = prompt_text if prompt_text is not None else ""

        # context_code is already Optional[str], f-strings will convert None to "None" string

        if task_type == "code_explanation":
            if context_code:
                full_prompt = f"You are PythonMasterAI. Explain the following Python code based on the request.\n\nCode:\n```python\n{context_code}\n```\n\nRequest: {_prompt_text_str}\n\nExplanation:"
            else:
                full_prompt = f"You are PythonMasterAI. Provide an explanation for the following concept/request: {_prompt_text_str}\n\nExplanation:"
        elif task_type == "docstring_generation":
            if context_code:
                full_prompt = f"You are PythonMasterAI. Generate a Python docstring for the following function. {_prompt_text_str}\n\nFunction:\n```python\n{context_code}\n```\n\nDocstring:"
            else:
                full_prompt = f"You are PythonMasterAI. Generate a Python docstring based on the following description: {_prompt_text_str}\n\nDocstring:"
        elif task_type == "code_generation":
            full_prompt = f"You are PythonMasterAI. Generate Python code for the following request: {_prompt_text_str}\n\nCode:"
        elif task_type == "concept_explanation":
            full_prompt = f"You are PythonMasterAI. Explain the following Python concept: {_prompt_text_str}\n\nExplanation:"
        else:
            full_prompt = f"You are PythonMasterAI. Respond to the following: {_prompt_text_str}"

        inputs = self.tokenizer(full_prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=tokenizer_max_input_len)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        eos_token_id_to_use = self.tokenizer.eos_token_id
        if eos_token_id_to_use is None:
            logger.warning("Tokenizer does not have a default eos_token_id. Generation might run to max_gen_length.")
        output_ids = self.generate(
            input_ids,
            attention_mask,
            max_new_tokens=max_gen_length,
            eos_token_id=eos_token_id_to_use,
            temperature=0.7,
            top_k=50
        )
        generated_token_ids = output_ids[0, input_ids.size(1):]
        generated_text = self.tokenizer.decode(cast(torch.Tensor, generated_token_ids), skip_special_tokens=True) # type: ignore
        return generated_text.strip()

    def log_performance(self, metric, value):
        self.performance_log.append((metric, value))
        # Note: The performance_log.json file is appended to and can grow large.
        # Consider external log rotation or management for disk usage if it becomes an issue.
        with open("performance_log.json", "a") as f:
            json.dump({"metric": metric, "value": value}, f)
            f.write("\n")
        if len(self.performance_log) > self.max_log_entries:
            self.performance_log = self.performance_log[-self.max_log_entries:]

    def log_research(self, topic, sources, success, note=""):
        log_entry = {"topic": topic, "sources": sources, "success": success}
        if note:
            log_entry["note"] = note
        self.research_log.append(log_entry)
        # Note: The research_log.json file is appended to and can grow large.
        # Consider external log rotation or management for disk usage if it becomes an issue.
        with open("research_log.json", "a") as f:
            json.dump(log_entry, f)
            f.write("\n")
        if len(self.research_log) > self.max_log_entries:
            self.research_log = self.research_log[-self.max_log_entries:]

    def log_source(self, source_name: str, url: str, score: float, added: bool, query: str, stage: str, reason_if_not_added: Optional[str] = None):
        timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "timestamp": timestamp,
            "source_name": source_name,
            "url": url,
            "score": score,
            "added": added,
            "discovery_query": query,
            "ai_stage": stage,
        }
        if not added and reason_if_not_added:
            log_entry["reason_if_not_added"] = reason_if_not_added

        self.source_log.append(log_entry)
        # Note: The source_log.json file is appended to and can grow large.
        # Consider external log rotation or management for disk usage if it becomes an issue.
        with open("source_log.json", "a") as f:
            json.dump(log_entry, f)
            f.write("\n")
        if len(self.source_log) > self.max_log_entries:
            self.source_log = self.source_log[-self.max_log_entries:]

    def _load_initial_source_categories(self) -> Dict[str, List[str]]:
        """
        Loads predefined categories of sources and their initial seed URLs or API query strings.
        This is used to guide the initial discovery process.
        These could eventually be loaded from a configuration file.
        """
        return {
            "github_beginner_python": [
                "search_query:python beginner tutorial stars:>50"
            ],
            "python_official_docs": [
                "https://docs.python.org/3/tutorial/",
                "https://docs.python.org/3/library/",
            ],
            "pypi_core_packages": [
                "requests",
                "numpy",
                "pandas",
                "matplotlib",
                "scikit-learn",
            ],  # Example: package names for PyPI lookup
            "study_guides_general_python": [
                "https://realpython.com/",
                "https://www.programiz.com/python-programming",
                "https://automatetheboringstuff.com/",
            ],
            "stackoverflow_python_common": [
                "search_query:python list comprehension example",
                "search_query:python dictionary iteration",
            ],
            "authoritative_style_guide": [
                "https://peps.python.org/pep-0008/",
                "https://peps.python.org/pep-0020/",
            ],
            "advanced_python_concepts": [
                "search_query:python generators advanced",
                "search_query:python decorators metaclasses",
            ],
            "python_news_blogs": ["search_query:python programming blog latest"],
        }

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
        loss_values = [v for k, v in self.performance_log if k == "loss"]
        avg_loss = sum(loss_values) / max(1, len(loss_values))
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
            self.task_progress = defaultdict(float) # Changed from int to float
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
            if task == "unit_test_accuracy": # Special handling if it's a direct value not a counter
                # This assumes unit_test_accuracy is set directly, not incremented.
                # If it were to be averaged or set, that logic would be elsewhere.
                pass # For now, let's assume it's set elsewhere or not incremented here.
            else:
                self.task_progress[task] += 1.0 if success else 0.0 # Ensure float arithmetic
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
                    logger.info(f"Gap identified: {gap}")

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
        if not api_url_to_use:
            return None
        try:
            response = requests.get(api_url_to_use, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.warning(f"Error fetching GitHub repo details for {api_url_to_use}: {e}")
            return None

    def _fetch_pypi_package_info(self, package_name_or_url):
        package_name = ""
        parsed_uri = urlparse(package_name_or_url)
        if parsed_uri.netloc == "pypi.org":
            path_segments = parsed_uri.path.strip('/').split('/')
            if len(path_segments) > 1:
                if path_segments[0] == "project":
                    package_name = path_segments[1]
                elif path_segments[0] == "pypi":
                    package_name = path_segments[1]
        elif not parsed_uri.scheme and not parsed_uri.netloc:
            package_name = package_name_or_url.strip()
        if not package_name:
            return None
        api_url = f"https://pypi.org/pypi/{package_name}/json"
        try:
            response = requests.get(api_url, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.warning(f"Error fetching PyPI package info for {package_name}: {e}")
            return None

    def select_research_sources(self, query: str) -> List[Dict]:
        """
        Selects relevant KNOWN sources (from self.known_sources) based on the query and AI's stage.
        Returns a list of source detail dictionaries.
        """
        relevant_sources_details = []
        for url, details in self.known_sources.items():
            if self._is_source_relevant_for_query(details, query):
                # Further filter by stage appropriateness if needed.
                # This example includes a basic stage check.
                category_hint = details.get("category_hint", "").lower()
                source_type = details.get("type", "").lower()
                source_url_lower = details.get("url", "").lower()  # Should always exist
                score = details.get("score", 0.0)

                is_stage_appropriate = False
                if self.stage == "baby" and (
                    "beginner" in category_hint
                    or "study_guides" in category_hint
                    or score > 0.5
                ):
                    is_stage_appropriate = True
                elif self.stage == "toddler" and (
                    "intermediate" in category_hint
                    or "pypi" in source_type
                    or "realpython" in source_url_lower
                    or score > 0.6
                ):
                    is_stage_appropriate = True
                elif self.stage == "teenager" and (
                    "advanced" in category_hint
                    or "peps" in category_hint
                    or score > 0.7
                ):
                    is_stage_appropriate = True
                elif self.stage == "adult" and (
                    "trends" in category_hint or "peps" in category_hint or score > 0.8
                ):  # Example for adult
                    is_stage_appropriate = True

                if is_stage_appropriate:
                    relevant_sources_details.append(details)  # Appending the dictionary
        logger.debug(
            f"Selected {len(relevant_sources_details)} sources for query '{query}' and stage '{self.stage}'."
        )
        return relevant_sources_details

    def validate_research_data(self, content, source):
        if "github" in source:
            return len(content) > 10 and "python" in content.lower()
        if "stackoverflow" in source:
            return len(content.split("\n")) > 1
        if "pypi" in source or "python_docs" in source:
            return len(content) > 50
        return True

    def get_research_scrape_targets(self):
        queries = self.formulate_research_queries()
        scrape_targets_dict = {}
        unique_urls_to_scrape = set()

        # Initialize variables for the source discovery logic block if it's intended to be here.
        # Note: This duplicates discovery logic from discover_new_sources().
        newly_added_sources_info = []
        new_sources_added_count = 0
        evaluated_urls_this_cycle = set()

        # Add sources based on dynamic queries
        if queries:
            for query in queries:
                selected_source_details_list = self.select_research_sources(query) # Returns list of dicts
                for source_details in selected_source_details_list:
                    unique_urls_to_scrape.add(source_details['url'])

        # Add sources based on general prioritization (e.g., high-value initial categories for current stage)
        # This ensures some scraping even if no specific knowledge gaps/queries exist.
        priority_urls = self.prioritize_scraping() # Returns list of URLs
        for url in priority_urls:
            unique_urls_to_scrape.add(url)

        return [(self.known_sources[url]['name'], url) for url in unique_urls_to_scrape if url in self.known_sources]

    def process_scraped_research_data(self, stage):
        queries = self.formulate_research_queries()

        # Initialize variables for the source discovery logic block within this method.
        newly_added_sources_info: List[Dict] = []
        new_sources_added_count: int = 0
        evaluated_urls_this_cycle: set[str] = set()

        # Phase 1: Process dynamically formulated research queries from knowledge gaps
        if queries:
            logger.info(
                f"Processing {len(queries)} dynamically formulated research queries from knowledge gaps."
            )
            for query in queries:
                candidate_sources_from_query = self.search_for_sources(
                    query
                )  # Returns (name, url) tuples
                for source_name, url in candidate_sources_from_query:
                    if url in evaluated_urls_this_cycle:
                        logger.debug(
                            f"Skipping already evaluated URL in this cycle: {url} for query '{query}'"
                        )
                        continue
                    evaluated_urls_this_cycle.add(url)
                    # Pass the original query as the hint/context
                    source_eval_info = self.evaluate_source(source_name, url, query)
                    if source_eval_info and source_eval_info.get("added"):
                        newly_added_sources_info.append(source_eval_info)
                        new_sources_added_count += 1
                        self.log_task_progress(
                            "find_sources"
                        )  # Generic task for dynamic discovery
        else:
            logger.info("No dynamic research queries to process from knowledge gaps.")

        # Phase 2: Process initial source categories (seed URLs/queries)
        # This phase runs regardless of dynamic queries, to ensure foundational sources are considered.
        logger.info(
            f"Processing {len(self.initial_source_categories)} initial source categories."
        )
        for category_name, seed_items in self.initial_source_categories.items():
            for seed_item in seed_items:
                # Determine if seed_item is a direct URL or a search query string
                if seed_item.startswith("http://") or seed_item.startswith("https://"):
                    derived_source_name = (
                        f"{category_name}_{urlparse(seed_item).netloc}"
                    )
                    if seed_item in evaluated_urls_this_cycle:
                        logger.debug(
                            f"Skipping already evaluated seed URL in this cycle: {seed_item} for category '{category_name}'"
                        )
                        continue
                    evaluated_urls_this_cycle.add(seed_item)
                    source_eval_info = self.evaluate_source(
                        derived_source_name, seed_item, category_name
                    )
                    if source_eval_info and source_eval_info.get("added"):
                        newly_added_sources_info.append(source_eval_info)
                        new_sources_added_count += 1
                        self.log_task_progress("find_sources_initial_category")
                elif seed_item.startswith("search_query:"):
                    actual_query = seed_item.replace("search_query:", "").strip()
                    logger.info(
                        f"Processing seed search query '{actual_query}' for category '{category_name}'"
                    )
                    candidate_sources_from_seed_query = self.search_for_sources(
                        actual_query
                    )
                    for source_name, url in candidate_sources_from_seed_query:
                        if url in evaluated_urls_this_cycle:
                            logger.debug(
                                f"Skipping already evaluated URL in this cycle: {url} for seed query '{actual_query}' (category '{category_name}')"
                            )
                            continue
                        evaluated_urls_this_cycle.add(url)
                        source_eval_info = self.evaluate_source(
                            source_name, url, category_name
                        )  # Use category as hint
                        if source_eval_info and source_eval_info.get("added"):
                            newly_added_sources_info.append(source_eval_info)
                            new_sources_added_count += 1
                            self.log_task_progress("find_sources_initial_category")
                elif category_name.startswith(
                    "pypi_"
                ):  # Heuristic for PyPI package names
                    pypi_url = f"https://pypi.org/project/{seed_item}/"  # seed_item is package name
                    derived_source_name = f"pypi_{seed_item}"
                    if pypi_url in evaluated_urls_this_cycle:
                        logger.debug(
                            f"Skipping already evaluated PyPI URL in this cycle: {pypi_url} for category '{category_name}'"
                        )
                        continue
                    evaluated_urls_this_cycle.add(pypi_url)
                    source_eval_info = self.evaluate_source(
                        derived_source_name, pypi_url, category_name
                    )
                    if source_eval_info and source_eval_info.get("added"):
                        newly_added_sources_info.append(source_eval_info)
                        new_sources_added_count += 1
                        self.log_task_progress("find_sources_initial_category")
                else:
                    logger.warning(
                        f"Seed item '{seed_item}' for category '{category_name}' is not a recognized URL, search_query, or PyPI package. Skipping."
                    )

        if (
            not newly_added_sources_info
            and not queries
            and not self.initial_source_categories
        ):
            logger.info("No active research queries to process post-scraping.")
            return
        query_resolution_map = {query: False for query in queries}
        # Iterate through all known (evaluated and added) sources
        for url, source_details in self.known_sources.items():
            source_name = source_details['name']
            latest_dataset_dir = self.get_latest_dataset_path(stage)
            if not latest_dataset_dir:
                logger.warning(f"Cannot process research data for stage '{stage}' as no latest dataset directory found.")
                return
            file_path = os.path.join(latest_dataset_dir, f"{source_name}.txt")

            # Check if this source is relevant to any of the current queries
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                is_content_valid = self.validate_research_data(content, source_name)
                for query in queries:
                    # Check if this source (by its category_hint or name) is relevant to the query
                    # This requires select_research_sources to be able to check if a specific source_details matches a query
                    if self._is_source_relevant_for_query(source_details, query):
                        if is_content_valid:
                            if not query_resolution_map[query]:
                                self.log_research(query, [source_name], success=True)
                                query_resolution_map[query] = True
                                research_task_key = next((k for k in self.growth_tasks[self.stage] if k.startswith("research_")), None)
                                if research_task_key:
                                    self.log_task_progress(research_task_key)
            # else: # If file doesn't exist, it means it wasn't scraped or failed.
            # This case is implicitly handled as the query won't be marked resolved by this source.
            # Logging of failed scrapes should happen in scrape_data.py or when get_research_scrape_targets is called.
        resolved_gaps_this_cycle = set()
        for query, resolved in query_resolution_map.items():
            if resolved:
                for gap_text in self.knowledge_gaps:
                    if gap_text.lower() in query.lower():
                        resolved_gaps_this_cycle.add(gap_text)
        if resolved_gaps_this_cycle:
            logger.info(f"Knowledge gaps resolved this cycle: {resolved_gaps_this_cycle}")
            self.knowledge_gaps = [gap for gap in self.knowledge_gaps if gap not in resolved_gaps_this_cycle]
        if queries and not self.knowledge_gaps:
            logger.info("All knowledge gaps from this research cycle appear to be resolved.")
        elif queries:
            logger.info(f"Remaining knowledge gaps after research: {self.knowledge_gaps}")

    def _is_source_relevant_for_query(self, source_details: Dict, query: str) -> bool:
        """
        Helper to determine if a given evaluated source is relevant to a specific query.
        This can be based on the source's category_hint, name, or type.
        """
        category_hint = source_details.get("category_hint", "").lower()
        source_name_lower = source_details.get("name", "").lower()
        query_lower = query.lower()

        # Simple keyword matching for now. This could be more sophisticated.
        if any(kw in category_hint for kw in query_lower.split()) or any(
            kw in source_name_lower for kw in query_lower.split()
        ):
            return True
        return False

    def conduct_research(self):
        logger.info("Conduct_research called. Identifying targets...")
        research_targets = self.get_research_scrape_targets()
        if research_targets:
            logger.info(f"Conduct_research initiating focused scraping for targets: {research_targets}")
            from scrape_data import scrape_data
            sources_to_scrape, urls_to_scrape = zip(*research_targets)
            scrape_data(self.stage, list(sources_to_scrape), list(urls_to_scrape))
        else:
            print("No specific research targets identified by conduct_research. Relying on general scraping or previously scraped data.")
        self.process_scraped_research_data(self.stage)

    def discover_new_sources(self):
        queries = self.formulate_research_queries()
        newly_added_sources_info = []
        new_sources_added_count = 0
        evaluated_urls_this_cycle = set() # To avoid re-evaluating the same URL in this cycle

        if not queries:
            logger.info("No research queries formulated, skipping source discovery.")
        logger.info(f"Source discovery cycle complete. Added {new_sources_added_count} new sources overall.")
        return newly_added_sources_info, new_sources_added_count

    def search_for_sources(self, query):
        """
        Searches for new information sources based on a query using configured APIs.
        """
        candidates = []
        logger.info(f"Searching for sources with query: '{query}'")

        if not self.api_config or not isinstance(self.api_config, dict):
            logger.warning("API config not loaded or invalid. Falling back to basic hardcoded source suggestions.")
            # Fallback to very basic suggestions if API config is missing
            if "python" in query.lower():
                candidates.append(("python_docs_official", "https://docs.python.org/3/"))
                candidates.append(("pypi_org", "https://pypi.org/"))
            return candidates

        api_keys = self.api_config.get("api_keys", {})
        api_endpoints = self.api_config.get("api_endpoints", {})
        search_params_config = self.api_config.get("search_parameters", {})

        # --- GitHub Repository Search ---
        github_pat = api_keys.get("github_pat")
        github_repo_search_url = api_endpoints.get("github_search_repositories")
        github_repo_query_params = search_params_config.get("github_repo_default_query_params", "")

        if github_pat and github_pat != "YOUR_GITHUB_PERSONAL_ACCESS_TOKEN_HERE" and github_repo_search_url:
            headers = {"Authorization": f"token {github_pat}"}

            query_lower = query.lower()
            # Heuristically extract primary keywords from the AI's formulated query.
            # Assumes query structure like "{topic} {task_type_suffix} {stage_keyword}"
            # e.g., "classes basics python documentation" or "decorators advanced example"
            core_query_parts = query.split()[:2] # e.g., ["classes", "basics"] or ["decorators", "advanced"]

            repo_search_terms = list(core_query_parts) # Start with core topic/task

            if "documentation" in query_lower:
                repo_search_terms.append("documentation")
            elif "tutorial" in query_lower:
                repo_search_terms.append("tutorial")
            elif "example" in query_lower and "advanced" not in query_lower : # "advanced example" might be better for code search
                repo_search_terms.append("example")

            # Ensure "python" is in search terms for relevance
            if "python" not in [term.lower() for term in repo_search_terms]:
                repo_search_terms.insert(0, "python")

            search_query_github_repos = " ".join(repo_search_terms)

            full_github_repo_url = f"{github_repo_search_url}?q={search_query_github_repos} language:python&{github_repo_query_params}"
            logger.info(f"Attempting GitHub repository search with query: '{search_query_github_repos} language:python'")
            try:
                response = requests.get(full_github_repo_url, headers=headers, timeout=10)
                response.raise_for_status()
                gh_results = response.json()
                for item in gh_results.get("items", []):
                    repo_name = item.get("full_name")
                    repo_url = item.get("html_url")
                    if repo_name and repo_url:
                        candidates.append((f"github_repo_{repo_name.replace('/', '_')}", repo_url))
                        logger.debug(f"  Added GitHub candidate: {repo_name} - {repo_url}")
            except requests.RequestException as e:
                logger.error(f"GitHub API search failed for query '{query}': {e}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse GitHub API response for query '{query}': {e}")
        else:
            logger.warning("GitHub PAT or search URL not configured. Skipping GitHub search.")

        # --- GitHub Code Search ---
        github_code_search_url = api_endpoints.get("github_search_code")
        github_code_query_params = search_params_config.get("github_code_default_query_params", "")

        if github_pat and github_pat != "YOUR_GITHUB_PERSONAL_ACCESS_TOKEN_HERE" and github_code_search_url:
            headers = {"Authorization": f"token {github_pat}"}
            query_lower_for_code = query.lower()
            code_search_terms = []

            # Determine if code search is appropriate and form terms
            if "example" in query_lower_for_code or "implement" in query_lower_for_code or \
               any(kw in query_lower_for_code for kw in ["function", "class", "method", "algorithm"]):

                core_code_topic = query.split()[:2] # e.g., ["decorators", "advanced"] or ["add", "functions"]
                code_search_terms.extend(core_code_topic)
                if "example" not in [t.lower() for t in code_search_terms]: # Add example if not implied
                    code_search_terms.append("example")

            if code_search_terms:
                if "python" not in [term.lower() for term in code_search_terms]:
                    code_search_terms.insert(0, "python")

                search_query_github_code = " ".join(code_search_terms)
                # `in:file` searches within file content. `language:python` is crucial.
                full_github_code_url = f"{github_code_search_url}?q={search_query_github_code} language:python in:file&{github_code_query_params}"
                logger.info(f"Attempting GitHub code search with query: '{search_query_github_code} language:python in:file'")
                try:
                    response = requests.get(full_github_code_url, headers=headers, timeout=10)
                    response.raise_for_status()
                    gh_code_results = response.json()
                    for item in gh_code_results.get("items", []):
                        repo_info = item.get("repository", {})
                        repo_name = repo_info.get("full_name")
                        file_path_in_repo = item.get("path")
                        file_html_url = item.get("html_url")
                        if repo_name and file_path_in_repo and file_html_url:
                            # Sanitize file_path_in_repo for use in source_name
                            safe_file_path = file_path_in_repo.replace('/', '_').replace('.', '_')
                            source_name = f"github_code_{repo_name.replace('/', '_')}_{safe_file_path}"
                            candidates.append((source_name, file_html_url))
                            logger.debug(f"  Added GitHub Code candidate: {source_name} - {file_html_url}")
                except requests.RequestException as e:
                    logger.error(f"GitHub code search failed for query '{query}': {e}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse GitHub code search response for query '{query}': {e}")
            else:
                logger.info(f"Skipping GitHub code search for query '{query}' as it doesn't strongly imply code examples.")
        else:
            logger.warning("GitHub PAT or code search URL not configured. Skipping GitHub code search.")

        # --- Google Custom Search ---
        google_api_key = api_keys.get("google_custom_search_api_key")
        google_cx_id = api_keys.get("google_custom_search_cx_id")
        google_search_url = api_endpoints.get("google_custom_search")
        google_num_results = search_params_config.get("google_search_default_num_results", "5")

        if google_api_key and google_api_key != "YOUR_GOOGLE_CSE_API_KEY_HERE" and \
           google_cx_id and google_cx_id != "YOUR_GOOGLE_CSE_CX_ID_HERE" and google_search_url:
            # Example: Google Custom Search for the query
            full_google_url = f"{google_search_url}?key={google_api_key}&cx={google_cx_id}&q={query}&num={google_num_results}"
            logger.info(f"Attempting Google Custom Search: {full_google_url}")
            try:
                response = requests.get(full_google_url, timeout=10)
                response.raise_for_status()
                google_results = response.json()
                for item in google_results.get("items", []):
                    title = item.get("title")
                    link = item.get("link")
                    if title and link:
                        candidates.append((f"web_{urlparse(link).netloc}_{title[:20].replace(' ', '_')}", link))
                        logger.debug(f"  Added Google Search candidate: {title} - {link}")
            except requests.RequestException as e:
                logger.error(f"Google Custom Search API failed for query '{query}': {e}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Google Custom Search API response for query '{query}': {e}")
        else:
            logger.warning("Google Custom Search API Key, CX ID, or URL not configured. Skipping Google search.")

        logger.info(f"Found {len(candidates)} potential new sources for query '{query}'.")
        return candidates

    def evaluate_source(self, source_name: str, url: str, query_or_category_hint: str) -> dict:
        """
        Evaluates a potential source based on its content, relevance, authority, and freshness.
        Logs the evaluation attempt and adds to self.known_sources if criteria are met.
        Returns a dictionary with evaluation details.
        """
        relevance_score = 0.0
        authority_score = 0.0
        freshness_score = 0.0
        if url in self.known_sources:
            existing_score = self.known_sources[url].get("score", 0.0)
            self.log_source(
                source_name,
                url,
                existing_score,
                False,
                query_or_category_hint,
                self.stage,
                reason_if_not_added="Already known",
            )
            logger.info(
                f"Source {url} already known with score {existing_score}. Not re-adding."
            )
            return {
                "name": source_name,
                "url": url,
                "score": existing_score,
                "added": False,
                "reason": "Already known",
                "type": self.known_sources[url].get("type", "unknown"),
            }

        query_keywords = set(query_or_category_hint.lower().split())
        relevance_points = 0
        if any(kw in url.lower() for kw in query_keywords):
            relevance_points += 1
        if any(kw in source_name.lower() for kw in query_keywords):
            relevance_points += 1
        if any(kw in url.lower() for kw in ["python", "programming", "tutorial", "example"]):
            relevance_points += 1
        relevance_score = min(relevance_points / 2.5, 1.0) * 0.4 if relevance_points > 0 else 0.05
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        github_details = None # type: ignore
        pypi_details = None # type: ignore
        source_type = "unknown"

        if "github.com" in domain or "api.github.com" in domain:
            github_details = self._fetch_github_repo_details(url)
            source_type = "github"
        elif "pypi.org" in domain:
            pypi_details = self._fetch_pypi_package_info(url)
            source_type = "pypi"
        if github_details and 'stargazers_count' in github_details:
            stars = github_details['stargazers_count']
            if stars >= 1000:
                authority_score = 0.3
            elif stars >= 100:
                authority_score = 0.2
            else:
                authority_score = 0.1
        elif pypi_details and 'info' in pypi_details and 'version' in pypi_details['info']:
            authority_score = 0.25
        elif "stackoverflow.com" in domain:
            authority_score = 0.25
        elif "docs.python.org" in domain or "peps.python.org" in domain:
            authority_score = 0.3
        elif any(known_good_domain in domain for known_good_domain in ["realpython.com", "djangoproject.com", "flask.palletsprojects.com"]):
            authority_score = 0.25
        else:
            authority_score = 0.1
        current_time_ts = time.time()
        freshness_score = 0.05
        if github_details and 'pushed_at' in github_details:
            last_push_date_str = github_details['pushed_at']
            try:
                last_push_dt = datetime.fromisoformat(last_push_date_str.replace('Z', '+00:00')).replace(tzinfo=timezone.utc)
                last_push_ts = last_push_dt.timestamp()
                age_days = (current_time_ts - last_push_ts) / (60 * 60 * 24)
                if age_days <= 30:
                    freshness_score = 0.3
                elif age_days <= 180:
                    freshness_score = 0.2
                elif age_days <= 365:
                    freshness_score = 0.15
                else:
                    freshness_score = 0.05
            except ValueError:
                logger.warning(f"Could not parse GitHub date: {last_push_date_str}")
        elif pypi_details and 'releases' in pypi_details and pypi_details['releases']:
            latest_version_str = pypi_details.get('info', {}).get('version')
            if latest_version_str and latest_version_str in pypi_details['releases']:
                release_dates = [item['upload_time_iso_8601'] for item in pypi_details['releases'][latest_version_str] if isinstance(item, dict) and 'upload_time_iso_8601' in item]
                if release_dates:
                    latest_upload_time_str = max(release_dates)
                    try:
                        upload_dt = datetime.fromisoformat(latest_upload_time_str.replace('Z', '+00:00')).replace(tzinfo=timezone.utc)
                        upload_ts = upload_dt.timestamp()
                        age_days = (current_time_ts - upload_ts) / (60 * 60 * 24)
                        if age_days <= 90:
                            freshness_score = 0.3
                        elif age_days <= 365:
                            freshness_score = 0.2
                        else:
                            freshness_score = 0.1
                    except ValueError:
                        logger.warning(f"Could not parse PyPI date: {latest_upload_time_str}")
        total_score = relevance_score + authority_score + freshness_score
        final_score = round(min(total_score, 1.0), 2)  # type: ignore
        logger.debug(
            f"Evaluated '{source_name}' ({url}) for query/category '{query_or_category_hint}': R={relevance_score:.2f}, A={authority_score:.2f}, F={freshness_score:.2f} -> Total={final_score:.2f}"
        )

        add_threshold = 0.6 # This could be configurable
        if final_score > add_threshold: # No need to check if url in self.known_sources again, already did at the start
            self.known_sources[url] = {
                "name": source_name,
                "url": url,
                "score": final_score,
                "type": source_type,
                "category_hint": query_or_category_hint,
                "added_timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.log_source(
                source_name, url, final_score, True, query_or_category_hint, self.stage
            )
            logger.info(f"Added new source: {source_name} ({url}) with score {final_score:.2f}")
            return {"name": source_name, "url": url, "score": final_score, "added": True, "type": source_type}
        else:
            self.log_source(
                source_name,
                url,
                final_score,
                False,
                query_or_category_hint,
                self.stage,
                reason_if_not_added=f"Score {final_score:.2f} <= threshold {add_threshold}",
            )
            logger.info(f"Source {source_name} ({url}) not added. Score {final_score:.2f} below threshold {add_threshold}.")
            return {"name": source_name, "url": url, "score": final_score, "added": False, "reason": f"Score {final_score:.2f} <= threshold {add_threshold}", "type": source_type}

    def prioritize_scraping(self):
        all_sources_for_stage = []
        if self.stage == "baby":
            all_sources_for_stage = [
                details["url"]
                for url, details in self.known_sources.items()
                if "beginner" in details.get("category_hint", "").lower()
                or "study_guides" in details.get("category_hint", "").lower()
            ]
        elif self.stage == "toddler":
            all_sources_for_stage = [
                details["url"]
                for url, details in self.known_sources.items()
                if "intermediate" in details.get("category_hint", "").lower()
                or "pypi" in details.get("type", "").lower()
                or "realpython" in details.get("url", "").lower()
            ]
        elif self.stage == "teenager":
            all_sources_for_stage = [
                details["url"]
                for url, details in self.known_sources.items()
                if "advanced" in details.get("category_hint", "").lower()
                or "peps" in details.get("category_hint", "").lower()
            ]
        else: # adult and beyond
            all_sources_for_stage = [
                details["url"]
                for url, details in self.known_sources.items()
                if "trends" in details.get("category_hint", "").lower()
                or "peps" in details.get("category_hint", "").lower()
                or details.get("score", 0) > 0.8
            ]  # High score sources
        # Ensure we return URLs that are actually in known_sources (should be by definition above)
        # and potentially filter by score or other criteria.
        # For now, the list comprehension above already filters by known_sources.
        # We return URLs, the caller (train.py) will need to get names if needed.
        logger.info(
            f"Prioritized {len(all_sources_for_stage)} URLs for scraping for stage '{self.stage}'."
        )
        return list(set(all_sources_for_stage))  # Return unique URLs

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
        logger.warning("generate_code() is deprecated. Use generate_for_evaluation() with task_type='code_generation'.")
        return self.generate_for_evaluation(input_text, "code_generation", max_gen_length=200)

    def generate_explanation(self, input_text):
        logger.warning("generate_explanation() is deprecated. Use generate_for_evaluation() with task_type='concept_explanation'.")
        return self.generate_for_evaluation(input_text, "concept_explanation", max_gen_length=150)

    def debug_code(self, input_text):
        logger.warning("debug_code() is deprecated and not fully implemented in generate_for_evaluation.")
        return self.generate_for_evaluation(input_text, "debug_code_placeholder", max_gen_length=100)

    def process_input(self, input_text, user_key):
        master_auth_url = get_typed_config_value("master_service.auth_url", "http://localhost:8000/master/auth", str)
        try:
            response = requests.post(master_auth_url, json={"key": user_key, "command": input_text}, timeout=5)
            if response.status_code == 200:
                if response.json().get("action") == "pause":
                    self.reset_to_checkpoint()
                    return f"{self.stage.capitalize()} paused by Master’s stop code"
                if "MASTER:" in input_text:
                    return f"Serving Master: {self.generate_response(input_text.replace('MASTER:', ''))}"
            return self.generate_response(input_text) # Default action if not a master command or auth fails non-critically
        except requests.RequestException as e:
            logger.error(f"Could not contact master authentication service at {master_auth_url}: {e}")
            return "Error: Could not verify master key with authentication service. Proceeding with standard response." + self.generate_response(input_text)

    def reset_to_checkpoint(self):
        logger.info("Reverting to checkpoint")
        self.task_progress = defaultdict(float) # Changed to float for consistency
        self.knowledge_gaps = []

    def get_status(self):
        params = sum(p.numel() for p in self.parameters())
        status = {"stage": self.stage, "parameters": f"{params:,}", "configuration_id": self.configuration_id, "device": str(self.device), "tasks": self.task_progress, "gaps": self.knowledge_gaps, "sources": list(self.known_sources.keys())}
        return json.dumps(status, indent=2)

    def get_state_for_checkpoint(self):
        return {"stage": self.stage,
                "task_progress": dict(self.task_progress), "knowledge_gaps": list(self.knowledge_gaps),             "known_sources": dict(self.known_sources),
            "initial_source_categories": dict(self.initial_source_categories), "configuration_id": self.configuration_id, "vocab_size": self.vocab_size, "n_layers": self.n_layers, "n_heads": self.n_heads, "hidden_size": self.hidden_size, "dropout": self.dropout, "dim_feedforward": self.dim_feedforward, "activation": self.activation, "performance_log": self.performance_log, "research_log": self.research_log, "source_log": self.source_log, "current_dataset_version": self.current_dataset_version}

    def load_checkpoint(self, filepath, optimizer=None):
        logger.info(f"Attempting to load checkpoint from: {filepath}")
        try:
            # Bandit B614: Ensure checkpoints are loaded only from trusted sources.
            checkpoint = torch.load(filepath, map_location=self.device)
        except FileNotFoundError:
            logger.warning(f"Checkpoint file not found: {filepath}")
            return False
        except Exception as e:
            logger.error(f"Error loading checkpoint file {filepath}: {e}", exc_info=True)
            return False
        ckpt_ai_state = checkpoint.get('ai_state')
        if not ckpt_ai_state:
            logger.error("Checkpoint is missing 'ai_state'. Cannot verify configuration or load.")
            return False
        # Add type check for ckpt_ai_state to help Pylance and for runtime safety
        if not isinstance(ckpt_ai_state, dict):
            logger.error(f"Checkpoint 'ai_state' is not a dictionary (type: {type(ckpt_ai_state)}). Cannot load.")
            return False
        ckpt_config_id = ckpt_ai_state.get('configuration_id')
        if self.configuration_id != ckpt_config_id:
            logger.error(f"Configuration ID mismatch! Model: '{self.configuration_id}', Checkpoint: '{ckpt_config_id}'. Aborting load.")
            return False
        params_to_check = ['vocab_size', 'n_layers', 'n_heads', 'hidden_size', 'dropout', 'dim_feedforward', 'activation']
        config_mismatch = False
        for param in params_to_check:
            model_param_val = getattr(self, param)
            ckpt_param_val = ckpt_ai_state.get(param)
            if model_param_val != ckpt_param_val:
                logger.error(f"Parameter mismatch for '{param}'. Model: {model_param_val}, Checkpoint: {ckpt_param_val}.")
                config_mismatch = True
        if config_mismatch:
            logger.error("Aborting checkpoint loading due to model parameter mismatch.")
            return False # Indent this return to be part of the if block

        # This block executes if config_mismatch is False
        logger.info("Checkpoint configuration matches model configuration.")
        try: # De-indent try block
            self.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model state_dict loaded successfully.")
        except Exception as e: # De-indent except block
            logger.error(f"Error loading model state_dict: {e}", exc_info=True)
            return False
        if optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Optimizer state_dict loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading optimizer state_dict: {e}", exc_info=True)
        self.stage = ckpt_ai_state.get('stage', self.stage)
        self.task_progress = defaultdict(float, ckpt_ai_state.get('task_progress', {})) # Changed from int to float
        self.knowledge_gaps = ckpt_ai_state.get('knowledge_gaps', [])
        self.known_sources = ckpt_ai_state.get('known_sources', {}) # Default to empty dict
        self.performance_log = ckpt_ai_state.get('performance_log', [])
        self.research_log = ckpt_ai_state.get('research_log', [])
        self.source_log = ckpt_ai_state.get('source_log', [])
        self.current_dataset_version = ckpt_ai_state.get('current_dataset_version', None)
        loaded_epoch = checkpoint.get('epoch', 'N/A')
        logger.info(f"Checkpoint loaded successfully. Resuming at stage '{self.stage}', epoch {loaded_epoch}, dataset version '{self.current_dataset_version}'.")
        return True

    def _try_load_latest_checkpoint(self):
        # Corrected: Use get_config_value for checkpoint_dir
        checkpoint_dir_from_config = str(get_config_value('checkpointing.checkpoint_dir', "checkpoints"))
        status_message = ""
        if not os.path.exists(checkpoint_dir_from_config):
            status_message = f"Checkpoint directory '{checkpoint_dir_from_config}' not found. Model will start fresh."
            logger.info(status_message)
            return status_message

        # --- Attempt 1: Load the specific '_latest.pt' file ---
        expected_latest_filename = f"model_stage_{self.stage}_config_{self.configuration_id}_latest.pt"
        latest_checkpoint_filepath = os.path.join(checkpoint_dir_from_config, expected_latest_filename)

        tried_primary = False
        primary_failed_to_load = False

        expected_filename = f"model_stage_{self.stage}_config_{self.configuration_id}_latest.pt"
        latest_checkpoint_filepath = os.path.join(checkpoint_dir_from_config, expected_filename)

        if os.path.exists(latest_checkpoint_filepath):
            tried_primary = True
            logger.info(f"Primary checkpoint target found: {latest_checkpoint_filepath}. Attempting to load.")
            if self.load_checkpoint(latest_checkpoint_filepath):
                status_message = f"Successfully loaded primary checkpoint: {latest_checkpoint_filepath} (Stage: {self.stage}, Config: {self.configuration_id})."
                logger.info(status_message)
                return status_message
            else: # Add an explicit else block
                primary_failed_to_load = True
                logger.warning(f"Found primary checkpoint {latest_checkpoint_filepath}, but failed to load. Will check for alternatives.")
        else:
            logger.info(f"Primary checkpoint '{latest_checkpoint_filepath}' not found. Searching for alternatives.")
        # --- Attempt 2: If '_latest.pt' not found or failed to load, use glob to find other epoch checkpoints ---
        glob_pattern = os.path.join(checkpoint_dir_from_config, f"model_stage_{self.stage}_config_{self.configuration_id}_epoch_*.pt")
        logger.info(f"Searching for alternative epoch-based checkpoints with pattern: {glob_pattern}")

        epoch_checkpoints = []
        for f_path in glob.glob(glob_pattern): # glob is imported at the top
            filename = os.path.basename(f_path)
            match = re.search(r"_epoch_(\d+)\.pt$", filename) # re is imported at the top
            if match:
                epoch_num = int(match.group(1))
                epoch_checkpoints.append((epoch_num, f_path))

        if epoch_checkpoints:
            epoch_checkpoints.sort(key=lambda x: x[0], reverse=True)
            best_alternative_filepath = epoch_checkpoints[0][1]
            logger.info(f"Found alternative epoch-based checkpoints. Highest epoch version is: {best_alternative_filepath}")

            if self.load_checkpoint(best_alternative_filepath):
                if tried_primary and primary_failed_to_load:
                    status_message = f"Primary checkpoint {latest_checkpoint_filepath} failed. Successfully loaded alternative: {best_alternative_filepath} (Stage: {self.stage}, Config: {self.configuration_id})."
                else: # Primary was not found
                    status_message = f"Primary checkpoint not found. Successfully loaded alternative: {best_alternative_filepath} (Stage: {self.stage}, Config: {self.configuration_id})."
                logger.info(status_message)
                return status_message
            else: # Alternative also failed
                if tried_primary and primary_failed_to_load:
                    status_message = f"Primary checkpoint {latest_checkpoint_filepath} failed. Alternative {best_alternative_filepath} also failed. Model starts fresh."
                else: # Primary not found, and alternative failed
                    status_message = f"Primary checkpoint not found. Alternative {best_alternative_filepath} also failed. Model starts fresh."
                logger.warning(status_message)
                return status_message
        else: # No alternative epoch checkpoints found by glob
            if tried_primary and primary_failed_to_load:
                status_message = f"Primary checkpoint {latest_checkpoint_filepath} failed. No other alternatives found. Model starts fresh."
            else: # Primary not found, and no alternatives by glob
                status_message = f"No suitable checkpoints found for stage '{self.stage}' and config '{self.configuration_id}'. Model starts fresh."
            logger.info(status_message)
            return status_message

    def get_config_dict(self):
        return {'vocab_size': self.vocab_size, 'n_layers': self.n_layers, 'n_heads': self.n_heads, 'hidden_size': self.hidden_size, 'dropout': self.dropout, 'dim_feedforward': self.dim_feedforward, 'activation': self.activation, 'configuration_id': self.configuration_id}
    def get_latest_dataset_path(self, stage: str) -> str | None:
        stage_data_dir = os.path.join("data", stage)
        latest_txt_path = os.path.join(stage_data_dir, "latest.txt")
        version_timestamp = None

        try:
            with open(latest_txt_path, "r") as f:
                version_timestamp = f.read().strip()
        except FileNotFoundError:
            logger.info(f"'latest.txt' not found in {stage_data_dir}. No dataset version specified.")
            return None
        except Exception as e:
            logger.error(f"Error reading 'latest.txt' in {stage_data_dir}: {e}", exc_info=True)
            return None # Return None if there was an error reading the file

        if not version_timestamp:
            logger.info(f"'latest.txt' in {stage_data_dir} is empty. No dataset version specified.")
            return None

        dataset_path = os.path.join(stage_data_dir, version_timestamp)
        if not os.path.isdir(dataset_path):
            logger.error(f"Dataset directory '{dataset_path}' (specified in latest.txt) does not exist.")
            return None
        return dataset_path
