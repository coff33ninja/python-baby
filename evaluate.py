import argparse
import json
import os
import torch
from python_master_ai import PythonMasterAI # Assuming it's in the same directory orPYTHONPATH
from datasets import load_metric
import traceback
import sys
from io import StringIO

# --- Unsafe Restricted Code Execution ---
class UnsafeRestrictedExec:
    """
    WARNING: This class executes arbitrary code and is NOT SAFE.
    It is intended for use ONLY in controlled environments with trusted code.
    Proper sandboxing is required for untrusted code execution.
    """
    def __init__(self, timeout_seconds=5):
        self.timeout_seconds = timeout_seconds # Placeholder, actual timeout not implemented here

    def execute(self, generated_code_str: str, unit_tests_str: str):
        """
        Executes the generated_code_str with unit_tests_str in a restricted environment.
        Returns a dictionary with 'passed' (bool) and 'output' (str).
        """
        # This is a very basic and unsafe way to execute code.
        # A real sandbox would involve Docker, seccomp, separate processes, etc.
        
        full_code_to_execute = f"{generated_code_str}\n\n{unit_tests_str}"
        
        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_stdout = StringIO()
        redirected_stderr = StringIO()
        sys.stdout = redirected_stdout
        sys.stderr = redirected_stderr

        results = {"passed": False, "output": "", "error": ""}
        
        try:
            # Create a somewhat restricted global scope
            restricted_globals = {"__builtins__": {"print": print, "range": range, "len": len, "list": list, "dict": dict, "str": str, "int": int, "float": float, "bool": bool, "AssertionError": AssertionError, "Exception": Exception, "True": True, "False": False, "None": None}}
            # For pytest-like tests, we might need to provide 'assert' or use a test runner.
            # This simple exec won't run pytest-style tests directly.
            # Assuming unit_tests_str uses basic assert statements for now.
            
            exec(full_code_to_execute, restricted_globals)
            results["passed"] = True # If exec completes without unhandled exception
            results["output"] = "Execution completed. All asserts passed (if any)."
            # Note: This doesn't capture assert failures directly unless they raise an unhandled exception.
            # A proper test runner would be needed for granular test results.

        except AssertionError as e:
            results["passed"] = False
            results["output"] = f"AssertionError: {e}"
            results["error"] = traceback.format_exc()
        except Exception as e:
            results["passed"] = False # Any other exception means failure
            results["output"] = f"Execution failed with error: {e}"
            results["error"] = traceback.format_exc()
        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Append captured output
            results["output"] += f"\n--- Captured STDOUT ---\n{redirected_stdout.getvalue()}"
            captured_stderr = redirected_stderr.getvalue()
            if captured_stderr:
                results["output"] += f"\n--- Captured STDERR ---\n{captured_stderr}"
            
            redirected_stdout.close()
            redirected_stderr.close()

        return results

# --- Text Metric Calculation ---
def calculate_text_metrics(predictions: list[str], references: list[list[str]]):
    """
    Calculates BLEU and ROUGE scores.
    predictions: A list of generated strings.
    references: A list of lists of reference strings.
    """
    results = {}
    try:
        sacrebleu_metric = load_metric("sacrebleu")
        rouge_metric = load_metric("rouge")

        # SacreBLEU expects references as list of lists of strings
        # ROUGE expects references as list of strings (taking the first reference if multiple)
        
        # Ensure predictions are strings
        str_predictions = [str(p) for p in predictions]

        sacrebleu_score = sacrebleu_metric.compute(predictions=str_predictions, references=references)
        results["sacrebleu"] = sacrebleu_score["score"]
        
        # For ROUGE, if multiple references, typically the first or best one is used.
        # Here, we'll just use the first reference for simplicity.
        rouge_references_flat = [ref[0] if ref else "" for ref in references] # Handle empty ref list
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
    # parser.add_argument("--max_generation_length", type=int, default=512, help="Max tokens for generation.") # Placeholder
    # parser.add_argument("--generation_temperature", type=float, default=0.7, help="Temperature for generation.") # Placeholder


    args = parser.parse_args()

    print(f"Starting evaluation with arguments: {args}")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Model Loading ---
    print(f"Loading model from checkpoint: {args.model_checkpoint_path}")
    try:
        checkpoint = torch.load(args.model_checkpoint_path, map_location=torch.device('cpu')) # Load to CPU first
        ai_state = checkpoint.get('ai_state')
        if not ai_state:
            raise ValueError("Checkpoint does not contain 'ai_state'. Cannot determine model configuration.")

        model_config = {
            "vocab_size": ai_state.get('vocab_size', 16000), # Default if not found
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
        model.eval() # Set model to evaluation mode
        # model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) # Move to device if needed
        print(f"Model loaded successfully. Device: {model.device}")
        model_config_id = ai_state.get('configuration_id', "unknown_config")
        model_stage_loaded = ai_state.get('stage', "unknown_stage")

    except Exception as e:
        print(f"Error loading model: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    # --- Evaluation ---
    all_results = []
    code_gen_results = {"total": 0, "passed": 0, "details": []}
    text_gen_metrics = {"sacrebleu": [], "rouge1": [], "rouge2": [], "rougeL": []}
    
    restricted_executor = UnsafeRestrictedExec()

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
                    context_code=context_code
                    # max_length=args.max_generation_length, # Add if using these
                    # temperature=args.generation_temperature
                )
                
                task_result = {"task_id": task_id, "task_type": task_type, "prompt": prompt, "generated_output": generated_output}

                if task_type == "code_generation":
                    unit_tests_str = task_data.get("unit_tests", "")
                    if not unit_tests_str:
                        print(f"  Warning: No unit tests provided for code_generation task {task_id}.")
                        task_result.update({"status": "no_tests", "passed": False, "execution_output": "No unit tests provided."})
                    else:
                        exec_result = restricted_executor.execute(generated_output, unit_tests_str)
                        task_result.update({
                            "status": "executed",
                            "passed": exec_result["passed"],
                            "execution_output": exec_result["output"],
                            "execution_error": exec_result.get("error", "")
                        })
                        code_gen_results["total"] += 1
                        if exec_result["passed"]:
                            code_gen_results["passed"] += 1
                    code_gen_results["details"].append(task_result) # Store detailed result for this task

                elif task_type in ["code_explanation", "docstring_generation", "concept_explanation"]:
                    reference_text_list = task_data.get("reference_text", [])
                    if not isinstance(reference_text_list, list): # Ensure it's a list
                        reference_text_list = [str(reference_text_list)] if reference_text_list else []
                    
                    # For sacrebleu, references should be a list of lists of strings
                    sacrebleu_references = [[ref] for ref in reference_text_list]

                    if not reference_text_list:
                        print(f"  Warning: No reference text provided for task {task_id}. Metrics will be 0.")
                        task_result.update({"sacrebleu": 0.0, "rouge": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}})
                    else:
                        metrics = calculate_text_metrics(predictions=[generated_output], references=sacrebleu_references)
                        task_result.update(metrics)
                        text_gen_metrics["sacrebleu"].append(metrics["sacrebleu"])
                        text_gen_metrics["rouge1"].append(metrics["rouge"]["rouge1"])
                        text_gen_metrics["rouge2"].append(metrics["rouge"]["rouge2"])
                        text_gen_metrics["rougeL"].append(metrics["rouge"]["rougeL"])
                else:
                    print(f"  Warning: Unknown task_type '{task_type}' for task {task_id}. No specific evaluation performed.")
                    task_result["status"] = "unknown_task_type"
                
                all_results.append(task_result)
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

    if code_gen_results["total"] > 0:
        pass_rate = (code_gen_results["passed"] / code_gen_results["total"]) * 100 if code_gen_results["total"] > 0 else 0
        summary["code_generation"] = {
            "total": code_gen_results["total"],
            "passed": code_gen_results["passed"],
            "pass_rate_percentage": round(pass_rate, 2)
        }
    
    if text_gen_metrics["sacrebleu"]: # If any text tasks were processed
        summary["text_generation_avg_metrics"] = {
            "avg_sacrebleu": round(sum(text_gen_metrics["sacrebleu"]) / len(text_gen_metrics["sacrebleu"]), 4) if text_gen_metrics["sacrebleu"] else 0.0,
            "avg_rouge1": round(sum(text_gen_metrics["rouge1"]) / len(text_gen_metrics["rouge1"]), 4) if text_gen_metrics["rouge1"] else 0.0,
            "avg_rouge2": round(sum(text_gen_metrics["rouge2"]) / len(text_gen_metrics["rouge2"]), 4) if text_gen_metrics["rouge2"] else 0.0,
            "avg_rougeL": round(sum(text_gen_metrics["rougeL"]) / len(text_gen_metrics["rougeL"]), 4) if text_gen_metrics["rougeL"] else 0.0,
        }

    # --- Output Files ---
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
    main()
