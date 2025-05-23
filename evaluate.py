import argparse
import json
import os
import torch
from python_master_ai import PythonMasterAI # Assuming it's in the same directory orPYTHONPATH
from datasets import load_metric
import traceback
import sys
from io import StringIO
import multiprocessing
from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.PrintCollector import PrintCollector
# from RestrictedPython.Guards import full_write_guard # Could be used for stricter write control
import time

# --- Secure Code Execution Target Function (for multiprocessing) ---
def _execute_restricted_code_target(code_string: str, tests_string: str, result_queue: multiprocessing.Queue):
    restricted_globals = dict(safe_globals)
    restricted_globals['_print_'] = PrintCollector # Capture print statements
    restricted_globals['_getattr_'] = getattr # Standard getattr, can be replaced with safer_getattr if needed
    restricted_globals['_getitem_'] = lambda obj, index: obj[index] # Basic item access guard
    restricted_globals['_write_'] = lambda x: x # Basic write guard, allows writing to attributes of objects created within restricted code
    # For more security, one might implement or use more sophisticated guards like full_write_guard,
    # but that can also break legitimate code that modifies objects.

    # Minimal set of builtins often needed
    restricted_globals['__builtins__']['AssertionError'] = AssertionError
    restricted_globals['__builtins__']['Exception'] = Exception
    restricted_globals['__builtins__']['True'] = True
    restricted_globals['__builtins__']['False'] = False
    restricted_globals['__builtins__']['None'] = None
    restricted_globals['__builtins__']['len'] = len
    restricted_globals['__builtins__']['list'] = list
    restricted_globals['__builtins__']['dict'] = dict
    restricted_globals['__builtins__']['str'] = str
    restricted_globals['__builtins__']['int'] = int
    restricted_globals['__builtins__']['float'] = float
    restricted_globals['__builtins__']['range'] = range
    # Add other safe builtins as necessary for common operations

    results = {"passed": False, "log": "", "stdout": "", "stderr": ""}
    
    # Capture stdout/stderr within the subprocess for exec, though RestrictedPython's _print_ handles prints
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_stdout_exec = StringIO()
    redirected_stderr_exec = StringIO()
    # Note: RestrictedPython's PrintCollector is the primary way to get 'print' output.
    # Redirecting sys.stdout/stderr here is more for completeness or if some part of the
    # executed code bypasses PrintCollector (less likely with compiled_restricted code).

    try:
        sys.stdout = redirected_stdout_exec
        sys.stderr = redirected_stderr_exec

        byte_code = compile_restricted(code_string, filename='<ai_generated_code>', mode='exec')
        test_byte_code = compile_restricted(tests_string, filename='<test_code>', mode='exec')
        
        local_scope = {} # Execution scope for the generated code and tests
        exec(byte_code, restricted_globals, local_scope)
        exec(test_byte_code, restricted_globals, local_scope)

        results["passed"] = True
        results["log"] = "Execution completed. All assertions passed (if any within tests_string)."

    except AssertionError as e:
        results["passed"] = False
        results["log"] = f"AssertionError: {e}"
        results["stderr"] = traceback.format_exc() # Store traceback in stderr field
    except Exception as e:
        results["passed"] = False
        results["log"] = f"Execution error: {type(e).__name__}: {e}"
        results["stderr"] = traceback.format_exc() # Store traceback in stderr field
    finally:
        sys.stdout = old_stdout # Restore original stdout/stderr
        sys.stderr = old_stderr
        
        results["stdout"] = restricted_globals['_print_'].collected_output() + redirected_stdout_exec.getvalue()
        # If stderr has content from redirection (e.g. from C extensions, or unhandled Python errors not caught by PrintCollector)
        err_val = redirected_stderr_exec.getvalue()
        if err_val:
             results["stderr"] = results.get("stderr","") + "\n--- Raw Stderr Capture ---\n" + err_val

        redirected_stdout_exec.close()
        redirected_stderr_exec.close()
        result_queue.put(results)

# --- Secure Executor Class ---
class SecureExecutor:
    def __init__(self, timeout_seconds=10): # Increased default timeout
        self.timeout_seconds = timeout_seconds

    def execute(self, code_string, tests_string):
        result_queue = multiprocessing.Queue()
        # Ensure 'spawn' start method for better cross-platform compatibility if issues arise with 'fork'
        # On Linux, 'fork' is default and usually fine. On Windows/macOS, 'spawn' is default.
        # ctx = multiprocessing.get_context('spawn') 
        # process = ctx.Process(...)
        process = multiprocessing.Process(
            target=_execute_restricted_code_target,
            args=(code_string, tests_string, result_queue)
        )

        process.start()
        process.join(timeout=self.timeout_seconds)

        if process.is_alive():
            print(f"Process {process.pid} timed out after {self.timeout_seconds}s. Terminating.")
            process.terminate()
            process.join(timeout=1) # Give it a moment to terminate
            if process.is_alive(): # If still alive
                print(f"Process {process.pid} did not terminate gracefully. Killing.")
                process.kill() # Force kill
                process.join()
            return False, "Execution timed out.", "", "Timeout Error: Process terminated."

        try:
            # Timeout for queue.get to prevent hanging if subprocess dies unexpectedly after join
            results = result_queue.get(timeout=2) 
            return results["passed"], results["log"], results["stdout"], results.get("stderr", "")
        except multiprocessing.queues.Empty: # Using specific exception for queue empty
             # This can happen if the process terminated due to external reasons (OOM) or internal unhandled exit
            exit_code = process.exitcode
            log_message = f"Execution process {process.pid} ended unexpectedly (exit code: {exit_code}) without returning results via queue."
            print(log_message)
            return False, log_message, "", f"Process Error (exit code {exit_code})"
        except Exception as e:
            log_message = f"Failed to retrieve results from subprocess queue: {type(e).__name__}: {e}"
            print(log_message)
            return False, log_message, "", "Queue/Process Communication Error"
        finally:
            result_queue.close()
            # Waits for the queue's internal thread to finish. Important for clean exit.
            result_queue.join_thread() 


# --- Text Metric Calculation ---
def calculate_text_metrics(predictions: list[str], references: list[list[str]]):
    results = {}
    try:
        sacrebleu_metric = load_metric("sacrebleu")
        rouge_metric = load_metric("rouge")
        str_predictions = [str(p) for p in predictions]
        sacrebleu_score = sacrebleu_metric.compute(predictions=str_predictions, references=references)
        results["sacrebleu"] = sacrebleu_score["score"]
        rouge_references_flat = [ref[0] if ref else "" for ref in references]
        rouge_score = rouge_metric.compute(predictions=str_predictions, references=rouge_references_flat)
        results["rouge"] = {
            "rouge1": rouge_score["rouge1"].mid.fmeasure,
            "rouge2": rouge_score["rouge2"].mid.fmeasure,
            "rougeL": rouge_score["rougeL"].mid.fmeasure,
        }
    except Exception as e:
        print(f"Error calculating text metrics: {e}")
        results["sacrebleu"] = 0.0
        results["rouge"] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PythonMasterAI model.")
    parser.add_argument("--model_checkpoint_path", type=str, required=True, help="Path to the model checkpoint (.pt file).")
    parser.add_argument("--eval_dataset_path", type=str, required=True, help="Path to the evaluation dataset (.jsonl file).")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory to save evaluation results.")
    parser.add_argument("--timeout_seconds", type=int, default=10, help="Timeout for code execution in seconds.")

    args = parser.parse_args()
    print(f"Starting evaluation with arguments: {args}")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Model Loading ---
    print(f"Loading model from checkpoint: {args.model_checkpoint_path}")
    try:
        checkpoint = torch.load(args.model_checkpoint_path, map_location=torch.device('cpu'))
        ai_state = checkpoint.get('ai_state')
        if not ai_state:
            raise ValueError("Checkpoint does not contain 'ai_state'. Cannot determine model configuration.")
        model_config = {
            "vocab_size": ai_state.get('vocab_size', 16000),
            "n_layers": ai_state.get('n_layers', 2),
            "n_heads": ai_state.get('n_heads', 4),
            "hidden_size": ai_state.get('hidden_size', 256),
            "dropout": ai_state.get('dropout', 0.1),
            "dim_feedforward": ai_state.get('dim_feedforward', 256 * 4),
            "activation": ai_state.get('activation', 'relu')
        }
        print(f"Instantiating PythonMasterAI with configuration: {model_config}")
        model = PythonMasterAI(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Model loaded successfully. Device: {model.device}")
        model_config_id = ai_state.get('configuration_id', "unknown_config")
        model_stage_loaded = ai_state.get('stage', "unknown_stage")
    except Exception as e:
        print(f"Error loading model: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    # --- Evaluation ---
    all_results = []
    code_gen_results_summary = {"total": 0, "passed": 0} # Simplified summary
    text_gen_metrics_agg = {"sacrebleu": [], "rouge1": [], "rouge2": [], "rougeL": []}
    
    secure_executor = SecureExecutor(timeout_seconds=args.timeout_seconds)

    print(f"Processing evaluation dataset: {args.eval_dataset_path}")
    try:
        with open(args.eval_dataset_path, 'r', encoding='utf-8') as f_eval:
            for line_num, line in enumerate(f_eval):
                try:
                    task_data = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed JSON line {line_num+1} in {args.eval_dataset_path}: {e}")
                    continue

                task_id = task_data.get("task_id", f"task_{line_num+1}")
                task_type = task_data.get("task_type")
                prompt = task_data.get("prompt")
                context_code = task_data.get("reference_code") if task_type in ["code_explanation", "docstring_generation"] else None
                
                print(f"\nProcessing Task ID: {task_id}, Type: {task_type}")

                if not task_type or not prompt:
                    print(f"  Skipping task {task_id} due to missing 'task_type' or 'prompt'.")
                    all_results.append({"task_id": task_id, "status": "skipped", "reason": "Missing task_type or prompt"})
                    continue

                generated_output = model.generate_for_evaluation(
                    prompt_text=prompt,
                    task_type=task_type,
                    context_code=context_code,
                    max_gen_length=512 # Default max_gen_length for evaluation
                )
                
                task_result_detail = {"task_id": task_id, "task_type": task_type, "prompt": prompt, "generated_output": generated_output}

                if task_type == "code_generation":
                    unit_tests_str = task_data.get("unit_tests", "")
                    if not unit_tests_str:
                        print(f"  Warning: No unit tests provided for code_generation task {task_id}.")
                        task_result_detail.update({"status": "no_tests", "passed": False, "execution_log": "No unit tests provided."})
                    else:
                        passed, log, stdout_capture, stderr_capture = secure_executor.execute(generated_output, unit_tests_str)
                        task_result_detail.update({
                            "status": "executed",
                            "passed": passed,
                            "execution_log": log,
                            "execution_stdout": stdout_capture,
                            "execution_stderr": stderr_capture
                        })
                        code_gen_results_summary["total"] += 1
                        if passed:
                            code_gen_results_summary["passed"] += 1
                
                elif task_type in ["code_explanation", "docstring_generation", "concept_explanation"]:
                    reference_text_list = task_data.get("reference_text", [])
                    if not isinstance(reference_text_list, list): 
                        reference_text_list = [str(reference_text_list)] if reference_text_list else []
                    sacrebleu_references = [[ref] for ref in reference_text_list]
                    if not reference_text_list:
                        print(f"  Warning: No reference text provided for task {task_id}. Metrics will be 0.")
                        task_result_detail.update({"sacrebleu": 0.0, "rouge": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}})
                    else:
                        metrics = calculate_text_metrics(predictions=[generated_output], references=sacrebleu_references)
                        task_result_detail.update(metrics)
                        text_gen_metrics_agg["sacrebleu"].append(metrics["sacrebleu"])
                        text_gen_metrics_agg["rouge1"].append(metrics["rouge"]["rouge1"])
                        text_gen_metrics_agg["rouge2"].append(metrics["rouge"]["rouge2"])
                        text_gen_metrics_agg["rougeL"].append(metrics["rouge"]["rougeL"])
                else:
                    print(f"  Warning: Unknown task_type '{task_type}' for task {task_id}. No specific evaluation performed.")
                    task_result_detail["status"] = "unknown_task_type"
                
                all_results.append(task_result_detail)
    except FileNotFoundError:
        print(f"Error: Evaluation dataset not found at {args.eval_dataset_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing evaluation dataset: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    # --- Results Aggregation and Reporting ---
    summary = {
        "model_checkpoint_path": args.model_checkpoint_path,
        "model_configuration_id": model_config_id,
        "model_stage_loaded": model_stage_loaded,
        "evaluation_dataset": args.eval_dataset_path,
        "total_tasks_processed": len(all_results),
    }

    if code_gen_results_summary["total"] > 0:
        pass_rate = (code_gen_results_summary["passed"] / code_gen_results_summary["total"]) * 100
        summary["code_generation"] = {
            "total": code_gen_results_summary["total"],
            "passed": code_gen_results_summary["passed"],
            "pass_rate_percentage": round(pass_rate, 2)
        }
    
    if text_gen_metrics_agg["sacrebleu"]:
        summary["text_generation_avg_metrics"] = {
            "avg_sacrebleu": round(sum(text_gen_metrics_agg["sacrebleu"]) / len(text_gen_metrics_agg["sacrebleu"]), 4),
            "avg_rouge1": round(sum(text_gen_metrics_agg["rouge1"]) / len(text_gen_metrics_agg["rouge1"]), 4),
            "avg_rouge2": round(sum(text_gen_metrics_agg["rouge2"]) / len(text_gen_metrics_agg["rouge2"]), 4),
            "avg_rougeL": round(sum(text_gen_metrics_agg["rougeL"]) / len(text_gen_metrics_agg["rougeL"]), 4),
        }

    results_filename_base = f"eval_results_{model_config_id}_{model_stage_loaded}"
    detailed_results_path = os.path.join(args.output_dir, f"{results_filename_base}.jsonl")
    summary_results_path = os.path.join(args.output_dir, f"eval_summary_{model_config_id}_{model_stage_loaded}.json")

    try:
        with open(detailed_results_path, 'w', encoding='utf-8') as f_detailed:
            for result_item in all_results:
                f_detailed.write(json.dumps(result_item) + "\n")
        print(f"Detailed evaluation results saved to: {detailed_results_path}")
    except Exception as e:
        print(f"Error saving detailed results: {e}")

    try:
        with open(summary_results_path, 'w', encoding='utf-8') as f_summary:
            json.dump(summary, f_summary, indent=4)
        print(f"Summary evaluation results saved to: {summary_results_path}")
    except Exception as e:
        print(f"Error saving summary results: {e}")

    print("\n--- Evaluation Summary ---")
    print(json.dumps(summary, indent=4))
    print("Evaluation finished.")

if __name__ == "__main__":
    # This is important for multiprocessing to work correctly on some platforms (like Windows)
    multiprocessing.freeze_support() 
    main()
