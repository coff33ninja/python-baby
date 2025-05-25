import argparse
import json
import logging  # Added for logging
import multiprocessing
import sys # Added for sys.platform
import os
import time
import queue # For queue.Empty
import traceback
from io import StringIO

# Call freeze_support() when the module is imported on Windows,
# before any multiprocessing objects might be created.
if sys.platform == "win32":
    multiprocessing.freeze_support()

import torch
import evaluate as hf_evaluate # Use the 'evaluate' library for metrics
from RestrictedPython import compile_restricted, safe_globals # type: ignore[import-untyped]
from RestrictedPython.PrintCollector import PrintCollector

from python_master_ai import PythonMasterAI
from utils import get_config_value, setup_logging  # Added for config and logging
from typing import Optional, Type, TypeVar # Added for helper and casting

# --- Initialize logger for this module ---
logger = logging.getLogger(__name__)

# --- Helper function for typed config values (local to this module) ---
_T_HELPER = TypeVar('_T_HELPER', float, int, str, bool)

def _get_typed_config_value(key: str, default_value: _T_HELPER, target_type: Type[_T_HELPER]) -> _T_HELPER:
    val = get_config_value(key, default_value) # get_config_value from utils returns Any
    # If val is already the exact target type (and not a bool masquerading as int/float if target is int/float)
    if isinstance(val, target_type) and not (target_type in (int, float) and isinstance(val, bool)):
        return val
    # If val is a type that can be directly converted (int, float, str, bool)
    if isinstance(val, (int, float, str, bool)):
        try:
            return target_type(val) # Attempt conversion
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Could not convert configured value '{str(val)[:100]}' for key '{key}' to {target_type.__name__}: {e}. "
                f"Using default value: {default_value}"
            )
            return default_value
    else: # val is some other unexpected type (e.g., dict, list)
        logger.warning(
            f"Configuration value for '{key}' is of unexpected type: {type(val)} (value: '{str(val)[:100]}'). "
            f"Using default value: {default_value}"
        )
        return default_value

# --- Secure Code Execution Target Function (for multiprocessing) ---
def _execute_restricted_code_target(
    code_string: str, tests_string: str, result_queue: multiprocessing.Queue
): # pragma: no cover
    restricted_globals = dict(safe_globals)
    _print_collector_instance = PrintCollector()
    restricted_globals["_print_"] = _print_collector_instance # type: ignore[assignment]
    restricted_globals["_getattr_"] = getattr # type: ignore[assignment]
    restricted_globals["_getitem_"] = lambda obj, index: obj[index] # type: ignore[assignment]
    restricted_globals["_write_"] = lambda x: x # type: ignore[assignment]
    restricted_globals["__builtins__"]["AssertionError"] = AssertionError # type: ignore[assignment]
    restricted_globals["__builtins__"]["Exception"] = Exception
    restricted_globals["__builtins__"]["True"] = True
    restricted_globals["__builtins__"]["False"] = False
    restricted_globals["__builtins__"]["None"] = None
    restricted_globals["__builtins__"]["len"] = len
    restricted_globals["__builtins__"]["list"] = list
    restricted_globals["__builtins__"]["dict"] = dict
    restricted_globals["__builtins__"]["str"] = str
    restricted_globals["__builtins__"]["int"] = int
    restricted_globals["__builtins__"]["float"] = float
    restricted_globals["__builtins__"]["range"] = range

    results = {"passed": False, "log": "", "stdout": "", "stderr": ""}
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_stdout_exec = StringIO()
    redirected_stderr_exec = StringIO()

    try:
        sys.stdout = redirected_stdout_exec
        sys.stderr = redirected_stderr_exec
        byte_code = compile_restricted(
            code_string, filename="<ai_generated_code>", mode="exec"
        )
        test_byte_code = compile_restricted(
            tests_string, filename="<test_code>", mode="exec"
        )
        local_scope = {}
        exec(byte_code, restricted_globals, local_scope)  # nosec B102
        exec(test_byte_code, restricted_globals, local_scope)  # nosec B102
        results["passed"] = True
        results["log"] = (
            "Execution completed. All assertions passed (if any within tests_string)."
        )
    except AssertionError as e:
        results["passed"] = False
        results["log"] = f"AssertionError: {e}"
        results["stderr"] = traceback.format_exc()
    except Exception as e:
        results["passed"] = False
        results["log"] = f"Execution error: {type(e).__name__}: {e}"
        results["stderr"] = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        # Use the public printed() method to get text and clear the collector
        collected_text_from_restricted_print = _print_collector_instance.printed()
        results["stdout"] = collected_text_from_restricted_print + redirected_stdout_exec.getvalue()

        err_val = redirected_stderr_exec.getvalue()
        if err_val:
            results["stderr"] = (
                results.get("stderr", "") + "\n--- Raw Stderr Capture ---\n" + err_val
            )
        redirected_stdout_exec.close()
        redirected_stderr_exec.close()
        result_queue.put(results)


# --- Secure Executor Class ---
class SecureExecutor:
    def __init__(self, timeout_seconds=None):
        default_timeout = _get_typed_config_value("evaluation.secure_exec_timeout", 10, int)
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else default_timeout
        )
        logger.info(
            f"SecureExecutor initialized with timeout: {self.timeout_seconds} seconds."
        )

    def execute(self, code_string, tests_string):
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_execute_restricted_code_target,
            args=(code_string, tests_string, result_queue),
        )
        process.start()
        process.join(timeout=self.timeout_seconds)
        if process.is_alive():
            logger.warning(
                f"Process {process.pid} timed out after {self.timeout_seconds}s. Terminating."
            )
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                logger.error(
                    f"Process {process.pid} did not terminate gracefully. Killing."
                )
                process.kill()
                process.join()
            return (
                False,
                "Execution timed out.",
                "",
                "Timeout Error: Process terminated.",
            )
        try:
            results = result_queue.get(timeout=2)
            return (
                results["passed"],
                results["log"],
                results["stdout"],
                results.get("stderr", ""),
            )
        except queue.Empty: # Changed from multiprocessing.queues.Empty
            exit_code = process.exitcode
            log_message = f"Execution process {process.pid} ended unexpectedly (exit code: {exit_code}) without returning results via queue."
            logger.error(log_message)
            return False, log_message, "", f"Process Error (exit code {exit_code})"
        except Exception as e:
            log_message = f"Failed to retrieve results from subprocess queue: {type(e).__name__}: {e}"
            logger.error(log_message, exc_info=True)
            return False, log_message, "", "Queue/Process Communication Error"
        finally:
            result_queue.close()
            result_queue.join_thread()


# --- Text Metric Calculation ---
def calculate_text_metrics(predictions: list[str], references: list[list[str]]):
    results = {}
    try:
        sacrebleu_metric = hf_evaluate.load("sacrebleu") # type: ignore[attr-defined]
        rouge_metric = hf_evaluate.load("rouge") # type: ignore[attr-defined]
        str_predictions = [str(p) for p in predictions]
        sacrebleu_score = sacrebleu_metric.compute(
            predictions=str_predictions, references=references
        )
        results["sacrebleu"] = sacrebleu_score["score"]
        rouge_references_flat = [ref[0] if ref else "" for ref in references]
        rouge_score = rouge_metric.compute(
            predictions=str_predictions, references=rouge_references_flat
        )
        results["rouge"] = {
            "rouge1": rouge_score["rouge1"].mid.fmeasure,
            "rouge2": rouge_score["rouge2"].mid.fmeasure,
            "rougeL": rouge_score["rougeL"].mid.fmeasure,
        }
    except Exception as e:
        logger.error(f"Error calculating text metrics: {e}", exc_info=True)
        results["sacrebleu"] = 0.0
        results["rouge"] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    return results


def main():
    default_output_dir = _get_typed_config_value("evaluation.results_dir", "eval_results", str)
    default_dataset_path = _get_typed_config_value(
        "evaluation.default_eval_dataset", "sample_evaluation_dataset.jsonl"
    , str)
    default_timeout_main = _get_typed_config_value("evaluation.secure_exec_timeout", 10, int)

    parser = argparse.ArgumentParser(description="Evaluate PythonMasterAI model.")
    parser.add_argument(
        "--model_checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--eval_dataset_path",
        type=str,
        default=default_dataset_path,
        help=f"Path to the evaluation dataset (.jsonl file). Default: {default_dataset_path}",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output_dir,
        help=f"Directory to save evaluation results. Default: {default_output_dir}",
    )
    parser.add_argument(
        "--timeout_seconds",
        type=int,
        default=default_timeout_main,
        help=f"Timeout for code execution in seconds. Default: {default_timeout_main}",
    )

    args = parser.parse_args()
    start_eval_time = time.time()
    logger.info(f"Starting evaluation with arguments: {args}")
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading model from checkpoint: {args.model_checkpoint_path}")
    try:
        # Bandit B614: Ensure checkpoints are loaded only from trusted sources.
        checkpoint = torch.load(
            args.model_checkpoint_path, map_location=torch.device("cpu")
        )
        ai_state = checkpoint.get("ai_state")
        if not ai_state:
            raise ValueError(
                "Checkpoint does not contain 'ai_state'. Cannot determine model configuration."
            )
        model = PythonMasterAI()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logger.info(f"Model loaded successfully. Device: {model.device}")
        model_config_id = model.configuration_id
        model_stage_loaded = model.stage
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        sys.exit(1)

    secure_executor = SecureExecutor(timeout_seconds=args.timeout_seconds)

    all_results = []
    code_gen_results_summary = {"total": 0, "passed": 0}
    text_gen_metrics_agg = {"sacrebleu": [], "rouge1": [], "rouge2": [], "rougeL": []}

    logger.info(f"Processing evaluation dataset: {args.eval_dataset_path}")
    try:
        with open(args.eval_dataset_path, "r", encoding="utf-8") as f_eval:
            for line_num, line in enumerate(f_eval):
                try:
                    task_data = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Skipping malformed JSON line {line_num+1} in {args.eval_dataset_path}: {e}"
                    )
                    continue
                task_id = task_data.get("task_id", f"task_{line_num+1}")
                task_type = task_data.get("task_type")
                prompt = task_data.get("prompt")

                raw_val_context_code = task_data.get("reference_code")
                eval_context_code: Optional[str] = None
                if task_type in ["code_explanation", "docstring_generation"]:
                    if isinstance(raw_val_context_code, str):
                        eval_context_code = raw_val_context_code
                    elif raw_val_context_code is not None:
                        logger.info(f"Task {task_id}: Converting non-string reference_code to string.")
                        try:
                            eval_context_code = str(raw_val_context_code)
                        except Exception:
                            logger.warning(f"Could not convert reference_code to string for task {task_id}. Treating as None.")
                            eval_context_code = None
                # If raw_val_context_code is None, eval_context_code remains None

                logger.info(f"Processing Task ID: {task_id}, Type: {task_type}")
                if not task_type or not prompt:
                    logger.warning(
                        f"  Skipping task {task_id} due to missing 'task_type' or 'prompt'."
                    )
                    all_results.append(
                        {
                            "task_id": task_id,
                            "status": "skipped",
                            "reason": "Missing task_type or prompt",
                        }
                    )
                    continue
                generated_output = model.generate_for_evaluation(
                    prompt_text=prompt,
                    task_type=task_type,
                    context_code=eval_context_code, # type: ignore[arg-type]
                    max_gen_length=512
                )
                task_result_detail = {
                    "task_id": task_id,
                    "task_type": task_type,
                    "prompt": prompt,
                    "generated_output": generated_output,
                }
                if task_type == "code_generation":
                    unit_tests_str = task_data.get("unit_tests", "")
                    if not unit_tests_str:
                        logger.warning(
                            f"  No unit tests provided for code_generation task {task_id}."
                        )
                        task_result_detail.update(
                            {
                                "status": "no_tests",
                                "passed": False,
                                "execution_log": "No unit tests provided.",
                            }
                        )
                    else:
                        passed, log, stdout_capture, stderr_capture = (
                            secure_executor.execute(generated_output, unit_tests_str)
                        )
                        task_result_detail.update(
                            {
                                "status": "executed",
                                "passed": passed,
                                "execution_log": log,
                                "execution_stdout": stdout_capture,
                                "execution_stderr": stderr_capture,
                            }
                        )
                        code_gen_results_summary["total"] += 1
                        if passed:
                            code_gen_results_summary["passed"] += 1
                elif task_type in [
                    "code_explanation",
                    "docstring_generation",
                    "concept_explanation",
                ]:
                    reference_text_list = task_data.get("reference_text", [])
                    if not isinstance(reference_text_list, list):
                        reference_text_list = (
                            [str(reference_text_list)] if reference_text_list else []
                        )
                    sacrebleu_references = [[ref] for ref in reference_text_list]
                    if not reference_text_list:
                        logger.warning(
                            f"  No reference text provided for task {task_id}. Metrics will be 0."
                        )
                        task_result_detail.update(
                            {
                                "sacrebleu": 0.0,
                                "rouge": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0},
                            }
                        )
                    else:
                        metrics = calculate_text_metrics(
                            predictions=[generated_output],
                            references=sacrebleu_references,
                        )
                        task_result_detail.update(metrics)
                        text_gen_metrics_agg["sacrebleu"].append(metrics["sacrebleu"])
                        text_gen_metrics_agg["rouge1"].append(
                            metrics["rouge"]["rouge1"]
                        )
                        text_gen_metrics_agg["rouge2"].append(
                            metrics["rouge"]["rouge2"]
                        )
                        text_gen_metrics_agg["rougeL"].append(
                            metrics["rouge"]["rougeL"]
                        )
                else:
                    logger.warning(
                        f"  Unknown task_type '{task_type}' for task {task_id}. No specific evaluation performed."
                    )
                    task_result_detail["status"] = "unknown_task_type"
                all_results.append(task_result_detail)
    except FileNotFoundError:
        logger.error(
            f"Evaluation dataset not found at {args.eval_dataset_path}", exc_info=True
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error processing evaluation dataset: {e}", exc_info=True)
        sys.exit(1)

    summary = {
        "evaluation_start_time_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_eval_time)
        ),
        "model_checkpoint_path": args.model_checkpoint_path,
        "model_configuration_id": model_config_id,
        "model_stage_loaded": model_stage_loaded,
        "evaluation_dataset": args.eval_dataset_path,
        "total_tasks_processed": len(all_results),
    }
    if code_gen_results_summary["total"] > 0:
        pass_rate = (
            code_gen_results_summary["passed"] / code_gen_results_summary["total"]
        ) * 100
        summary["code_generation"] = {
            "total": code_gen_results_summary["total"],
            "passed": code_gen_results_summary["passed"],
            "pass_rate_percentage": round(pass_rate, 2),
        }
    if text_gen_metrics_agg["sacrebleu"]:
        summary["text_generation_avg_metrics"] = {
            "avg_sacrebleu": (
                round(
                    sum(text_gen_metrics_agg["sacrebleu"])
                    / len(text_gen_metrics_agg["sacrebleu"]),
                    4,
                )
                if text_gen_metrics_agg["sacrebleu"]
                else 0.0
            ),
            "avg_rouge1": (
                round(
                    sum(text_gen_metrics_agg["rouge1"])
                    / len(text_gen_metrics_agg["rouge1"]),
                    4,
                )
                if text_gen_metrics_agg["rouge1"]
                else 0.0
            ),
            "avg_rouge2": (
                round(
                    sum(text_gen_metrics_agg["rouge2"])
                    / len(text_gen_metrics_agg["rouge2"]),
                    4,
                )
                if text_gen_metrics_agg["rouge2"]
                else 0.0
            ),
            "avg_rougeL": (
                round(
                    sum(text_gen_metrics_agg["rougeL"])
                    / len(text_gen_metrics_agg["rougeL"]),
                    4,
                )
                if text_gen_metrics_agg["rougeL"]
                else 0.0
            ),
        }

    end_eval_time = time.time()
    total_eval_duration = end_eval_time - start_eval_time
    summary["total_evaluation_duration_seconds"] = round(total_eval_duration, 2)

    results_filename_base = f"eval_results_{model_config_id}_{model_stage_loaded}"
    detailed_results_path = os.path.join(
        args.output_dir, f"{results_filename_base}.jsonl"
    )
    summary_results_path = os.path.join(
        args.output_dir, f"eval_summary_{model_config_id}_{model_stage_loaded}.json"
    )

    try:
        with open(detailed_results_path, "w", encoding="utf-8") as f_detailed:
            for result_item in all_results:
                f_detailed.write(json.dumps(result_item) + "\n")
        logger.info(f"Detailed evaluation results saved to: {detailed_results_path}")
    except Exception as e:
        logger.error(f"Error saving detailed results: {e}", exc_info=True)
    try:
        with open(summary_results_path, "w", encoding="utf-8") as f_summary:
            json.dump(summary, f_summary, indent=4)
        logger.info(f"Summary evaluation results saved to: {summary_results_path}")
    except Exception as e:
        logger.error(f"Error saving summary results: {e}", exc_info=True)

    logger.info("\n--- Evaluation Summary ---")
    # Log the summary as well, including the new duration
    logger.info(json.dumps(summary, indent=4))
    logger.info(f"Total evaluation duration: {total_eval_duration:.2f} seconds.")
    print(
        "Evaluation finished. Check logs for details."
    )  # Keep a simple print for final console output


if __name__ == "__main__":
    # Setup logging using values from config file
    # setup_logging() will read from config itself.
    setup_logging()
    main()
