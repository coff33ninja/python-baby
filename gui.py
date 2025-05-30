# gui.py
import gradio as gr
from python_master_ai import PythonMasterAI
import json
from grow import grow_model
import os
from datetime import datetime
import subprocess
import sys
import re
import traceback
import logging  # Added for logging
from utils import (
    get_typed_config_value,
    setup_logging,
)  # Removed unused get_config_value

# --- Setup Logging ---
# Call setup_logging early, using config values.
# It's okay if utils.py's initial load_config prints a warning before this is fully set.
setup_logging()  # Removed arguments since setup_logging does not accept any.

# --- Initialize logger for this module ---
logger = logging.getLogger(__name__)


# Attempt to import PyPDF2 for PDF processing
try:
    from PyPDF2 import PdfReader

    pdf_support_available = True
    logger.info("PyPDF2 imported successfully. PDF upload support enabled.")
except ImportError:
    PdfReader = None
    pdf_support_available = False
    # This print is okay as it's an early warning before full logging might be desired by user.
    print(
        "WARNING: PyPDF2 not installed. PDF upload support will be disabled. To enable, run: pip install PyPDF2"
    )
    logger.warning("PyPDF2 not installed. PDF upload support will be disabled.")


model = PythonMasterAI()  # This will log its own config messages
MASTER_KEY = PythonMasterAI.MASTER_KEY
logger.info("PythonMasterAI model initialized.")


def validate_master_key(user_key_input, session_key_state):
    if user_key_input == MASTER_KEY:
        logger.info("Master Key validated successfully for session.")
        return user_key_input, "Master Key Validated for this session."
    elif session_key_state == MASTER_KEY:  # Key already validated in session
        logger.debug("Using session-validated Master Key.")
        return session_key_state, "Using session-validated Master Key."
    logger.warning("Invalid Master Key provided or session key not validated.")
    return None, "Invalid Master Key. Please enter the correct key."


def interact_with_ai(command, user_key, session_key_state):
    logger.debug(f"interact_with_ai called with command: '{command[:50]}...'")

    current_valid_key = (
        session_key_state if session_key_state == MASTER_KEY else user_key
    )
    if current_valid_key != MASTER_KEY:
        logger.warning("Invalid Master key attempt in interact_with_ai.")
        return "Invalid Master key"

    response = model.process_input(command, current_valid_key)
    logger.debug(f"Response from AI: {response[:100]}...")
    return response


def trigger_research():
    logger.info("Research triggered via GUI.")
    model.conduct_research()  # This method should use internal logging
    try:
        with open("research_log.json", "r") as f:
            logs = [json.loads(line) for line in f]
        logger.info("Research log successfully read.")
        return json.dumps(logs[-5:], indent=2)
    except FileNotFoundError:
        logger.warning(
            "research_log.json not found. Creating an empty research_log.json file."
        )
        with open("research_log.json", "w") as f:
            pass  # Create an empty file
        return (
            json.dumps([], indent=2)
            + "\n[No research log found yet. An empty research_log.json file has been created. It will be populated after the first research run.]"
        )
    except json.decoder.JSONDecodeError as e:
        logger.error(f"Malformed JSON in research_log.json: {e}", exc_info=True)
        return "Error: research_log.json is corrupted or contains malformed JSON. Please fix or delete the file."
    except Exception as e:
        logger.error(f"Error reading research_log.json: {e}", exc_info=True)
        return f"Error reading research_log.json: {e}"


def discover_sources():
    logger.info("Source discovery triggered via GUI.")
    new_sources = (
        model.discover_new_sources()
    )  # This method should use internal logging
    discovered_message = (
        f"Discovered sources in this run: {new_sources}\n\n"
        if new_sources
        else "No new sources discovered in this run.\n\n"
    )
    logs = []
    malformed_lines = 0
    recovered_objects = 0
    try:
        with open("source_log.json", "r") as f:
            for line in f:
                if line.strip():
                    try:
                        logs.append(json.loads(line))
                    except json.decoder.JSONDecodeError as e:
                        s = line.strip()
                        import re

                        # This regex splits at every '}{' or '}{' (not inside strings)
                        parts = re.split(r"}\s*{", s)
                        if len(parts) > 1:
                            for i, part in enumerate(parts):
                                # Re-add braces removed by split
                                if i == 0:
                                    part = part + "}"
                                elif i == len(parts) - 1:
                                    part = "{" + part
                                else:
                                    part = "{" + part + "}"
                                try:
                                    logs.append(json.loads(part))
                                    recovered_objects += 1
                                except Exception as e2:
                                    malformed_lines += 1
                                    logger.warning(
                                        f"Could not recover JSON object from split part: {e2}. Part: {part}"
                                    )
                        else:
                            malformed_lines += 1
                            logger.warning(
                                f"Malformed JSON line in source_log.json: {e}. Line content: {line.strip()}"
                            )
        logger.info(
            "Source log successfully read as JSONL (one JSON object per line, with recovery for concatenated objects)."
        )
        msg = discovered_message + "Last 5 source log entries:\n" + json.dumps(
            logs[-5:], indent=2
        )
        if malformed_lines > 0:
            msg += f"\n\nWarning: {malformed_lines} malformed line(s) were skipped in source_log.json. Please check the file for corruption."
        if recovered_objects > 0:
            msg += f"\n\nNote: {recovered_objects} JSON object(s) were recovered from lines containing multiple concatenated objects."
        return msg
    except FileNotFoundError:
        logger.error("source_log.json not found.")
        return discovered_message + "Error: source_log.json not found."
    except Exception as e:
        logger.error(f"Error reading source_log.json: {e}", exc_info=True)
        return discovered_message + f"Error reading source_log.json: {e}"


def get_status():
    logger.info("Status requested via GUI.")
    return model.get_status()  # Assumes get_status returns a string


def approve_growth(user_key, session_key_state):
    global model
    logger.info("Growth approval requested via GUI.")

    current_valid_key = (
        session_key_state if session_key_state == MASTER_KEY else user_key
    )
    if current_valid_key != MASTER_KEY:
        logger.warning("Invalid Master key attempt in approve_growth.")
        return "Invalid Master key"
    try:
        # grow_model should use internal logging for its operations
        new_model, _ = grow_model(model)
        if new_model != model:  # Check if growth actually happened
            model = new_model  # Update global model instance
            logger.info(
                f"Model growth approved and successful. New stage: {model.stage}"
            )
            return f"{model.stage.capitalize()} growth approved! Model updated."
        else:
            logger.info(
                "Model growth conditions not met or not approved by master_key service."
            )
            return "Model growth conditions not met or not approved."

    except Exception as e:
        logger.error(f"Error during growth approval: {e}", exc_info=True)
        return f"Error during growth: {e}"


def handle_manual_upload(files, target_stage, user_key, session_key_state):
    logger.info(
        f"Manual upload initiated for stage '{target_stage}' with {len(files) if files else 0} files."
    )

    current_valid_key = (
        session_key_state if session_key_state == MASTER_KEY else user_key
    )
    if current_valid_key != MASTER_KEY:
        logger.warning("Invalid Master key attempt in handle_manual_upload.")
        return "Invalid Master key"

    if not files:
        logger.warning("No files were uploaded in handle_manual_upload.")
        return "No files were uploaded."
    if not target_stage:
        logger.warning("Target stage not selected in handle_manual_upload.")
        return "Please select a target stage for the data."

    version_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage_base_dir = os.path.join("data", target_stage)
    version_data_dir = os.path.join(stage_base_dir, version_timestamp)
    os.makedirs(version_data_dir, exist_ok=True)
    logger.info(f"Created versioned directory for manual upload: {version_data_dir}")

    results = []
    saved_filenames_for_manifest = []

    for uploaded_file in files:
        actual_original_filename = uploaded_file.name
        temp_file_path = str(uploaded_file)
        logger.debug(
            f"Processing uploaded file: {actual_original_filename} (temp path: {temp_file_path})"
        )

        original_filename_basename = os.path.basename(actual_original_filename)
        content_to_save = None

        original_filename_stem, original_extension = os.path.splitext(
            original_filename_basename
        )
        safe_stem = "".join(
            c if c.isalnum() or c == "_" else "_" for c in original_filename_stem
        )
        safe_ext = "".join(
            c if c.isalnum() else "" for c in original_extension.replace(".", "")
        )
        descriptive_part = f"{safe_stem}_{safe_ext}" if safe_ext else safe_stem
        save_filename = f"manual_upload_{descriptive_part}.txt"
        save_path = os.path.join(version_data_dir, save_filename)

        if original_filename_basename.lower().endswith(".pdf"):
            if pdf_support_available and PdfReader is not None:
                try:
                    reader = PdfReader(temp_file_path)
                    pdf_text_parts = [
                        page.extract_text() or "" for page in reader.pages
                    ]
                    content_to_save = "\n".join(pdf_text_parts).strip()
                    if not content_to_save:
                        results.append(
                            f"Info: Extracted no text from PDF '{actual_original_filename}'. An empty .txt file will be saved."
                        )
                        logger.info(
                            f"Extracted no text from PDF '{actual_original_filename}'."
                        )
                except Exception as pdf_e:
                    results.append(
                        f"Error parsing PDF '{actual_original_filename}': {pdf_e}. An empty .txt file will be saved if possible."
                    )
                    logger.error(
                        f"Error parsing PDF '{actual_original_filename}': {pdf_e}",
                        exc_info=True,
                    )
                    content_to_save = ""
            else:
                results.append(
                    f"Info: PDF support not available. Skipping PDF '{actual_original_filename}'."
                )
                logger.warning(
                    f"PDF support not available. Skipping PDF '{actual_original_filename}'."
                )
                continue
        else:
            try:
                with open(
                    temp_file_path, "r", encoding="utf-8", errors="ignore"
                ) as f_in:
                    content_to_save = f_in.read()
            except Exception as text_e:
                results.append(
                    f"Error reading text file '{actual_original_filename}': {text_e}. An empty .txt file will be saved if possible."
                )
                logger.error(
                    f"Error reading text file '{actual_original_filename}': {text_e}",
                    exc_info=True,
                )
                content_to_save = ""

        if content_to_save is not None:
            try:
                with open(save_path, "w", encoding="utf-8") as f_out:
                    f_out.write(content_to_save)
                results.append(
                    f"Processed '{actual_original_filename}' and saved as '{save_filename}' in version {version_timestamp}."
                )
                saved_filenames_for_manifest.append(save_filename)
                logger.debug(
                    f"Saved content from '{actual_original_filename}' to '{save_path}'."
                )
            except Exception as save_e:
                results.append(
                    f"Error saving content from '{actual_original_filename}' to '{save_filename}': {save_e}"
                )
                logger.error(
                    f"Error saving content from '{actual_original_filename}' to '{save_filename}': {save_e}",
                    exc_info=True,
                )

    manifest_content = {
        "version_timestamp": version_timestamp,
        "dataset_path_relative_to_stage_dir": version_timestamp,
        "creation_event_type": "manual_upload",
        "stage_uploaded_for": target_stage,
        "uploaded_files_summary": {
            "count": len(saved_filenames_for_manifest),
            "original_filenames_provided": [
                f.name for f in files if hasattr(f, "name")
            ],
            "saved_filenames_in_version_dir": saved_filenames_for_manifest,
        },
    }
    manifest_filepath = os.path.join(version_data_dir, "manifest.json")
    try:
        with open(manifest_filepath, "w", encoding="utf-8") as f_manifest:
            json.dump(manifest_content, f_manifest, indent=4)
        results.append(f"Saved manifest to {manifest_filepath}")
        logger.info(f"Saved manifest to {manifest_filepath}")
    except Exception as e:
        results.append(f"Error saving manifest.json: {e}")
        logger.error(f"Error saving manifest.json: {e}", exc_info=True)

    latest_txt_path = os.path.join(stage_base_dir, "latest.txt")
    try:
        with open(latest_txt_path, "w", encoding="utf-8") as f_latest:
            f_latest.write(version_timestamp)
        results.append(
            f"Updated 'latest.txt' in {stage_base_dir} to version {version_timestamp}"
        )
        logger.info(
            f"Updated 'latest.txt' in {stage_base_dir} to version {version_timestamp}"
        )
    except Exception as e:
        results.append(f"Error updating latest.txt: {e}")
        logger.error(f"Error updating latest.txt: {e}", exc_info=True)

    final_message = "\n".join(results)
    final_message += f"\n\nNote: Manually uploaded data for stage '{target_stage}' saved to version '{version_timestamp}'. This version is now set as the latest."
    logger.info(
        f"Manual upload process completed for stage '{target_stage}', version '{version_timestamp}'."
    )
    return final_message


def run_script_in_background(command_list, user_key, session_key_state, script_name):
    current_valid_key = (
        session_key_state if session_key_state == MASTER_KEY else user_key
    )
    if current_valid_key != MASTER_KEY:
        logger.warning(f"Invalid Master Key used for attempting to run {script_name}.")
        return "Invalid Master key"

    # Log user info carefully, e.g., just the last few chars of the key if it's sensitive
    # Use current_valid_key for display
    user_key_display = (
        f"...{current_valid_key[-4:]}"
        if len(current_valid_key) >= 4
        else "provided key"
    )
    logger.info(
        f"User (key ending '{user_key_display}') attempting to run {script_name} with command: {' '.join(command_list)}"
    )

    script_timeout = get_typed_config_value(
        "gui_settings.script_execution_timeout", 3600, int
    )  # Default 1 hour

    try:
        logger.info(f"Executing command for {script_name}: {' '.join(command_list)}")

        # Environment setup for the subprocess
        # Inherit current environment and ensure Python uses UTF-8 for I/O
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"  # Ensures output is not heavily buffered

        process = subprocess.run(
            command_list,
            capture_output=True,
            text=True,  # Decodes output as text
            check=False,  # We will check returncode manually
            encoding="utf-8",  # Specify encoding for decoding
            errors="replace",  # Replace characters that can't be decoded
            env=env,  # Pass the modified environment
            timeout=script_timeout,  # Add timeout
        )

        output_message = f"--- {script_name} Execution Details ---\n"
        output_message += f"Command: {' '.join(command_list)}\n"
        output_message += f"Return Code: {process.returncode}\n\n"

        # Handle STDOUT
        if process.stdout and process.stdout.strip():
            output_message += f"--- STDOUT ---\n{process.stdout.strip()}\n\n"
            logger.debug(
                f"{script_name} STDOUT (snippet):\n{process.stdout.strip()[:500]}..."
            )
        else:
            output_message += "--- STDOUT ---\n(No standard output)\n\n"

        # Handle STDERR and create an error summary
        error_summary = ""
        if process.stderr and process.stderr.strip():
            stderr_cleaned = process.stderr.strip()
            output_message += f"--- STDERR ---\n{stderr_cleaned}\n\n"
            logger.debug(f"{script_name} STDERR:\n{stderr_cleaned}")

            stderr_lines = [
                line for line in stderr_cleaned.splitlines() if line.strip()
            ]
            if stderr_lines:
                found_specific_error = False
                for line in reversed(
                    stderr_lines
                ):  # Check from the end for more relevant error messages
                    if any(
                        err_type.lower() in line.lower()
                        for err_type in [
                            "error:",
                            "exception:",
                            "keyerror:",
                            "valueerror:",
                            "filenotfounderror:",
                            "traceback (most recent call last):",
                            "importerror:",
                            "modulenotfounderror:",
                            "attributeerror:",
                            "typeerror:",
                            "indexerror:",
                        ]
                    ):
                        idx = stderr_lines.index(line)
                        # Try to get a few lines for context, but not too many
                        error_summary_lines = stderr_lines[
                            idx : min(idx + 3, len(stderr_lines))
                        ]
                        error_summary = (
                            f"Error Summary: {' '.join(error_summary_lines)}"
                        )
                        found_specific_error = True
                        break
                if not found_specific_error:
                    error_summary = (
                        f"Error Summary (last line of error): {stderr_lines[-1]}"
                    )
        else:
            output_message += "--- STDERR ---\n(No standard error output)\n\n"

        # Final Status
        if process.returncode == 0:
            logger.info(
                f"{script_name} completed successfully. Command: {' '.join(command_list)}"
            )
            final_status_message = f"✅ {script_name} completed successfully."
            # Show warnings or deprecation notices from stderr even on success
            if error_summary and (
                "warning" in error_summary.lower()
                or "deprecated" in error_summary.lower()
            ):
                final_status_message += f"\nNotices from script: {error_summary}"
        else:
            logger.error(
                f"{script_name} failed with return code {process.returncode}. Command: {' '.join(command_list)}. Stderr (full): {process.stderr.strip()}"
            )
            final_status_message = (
                f"❌ {script_name} FAILED with return code {process.returncode}."
            )
            if error_summary:
                final_status_message += f"\n{error_summary}"
            elif process.stderr and process.stderr.strip():
                final_status_message += "\nCheck STDERR details above."
            else:
                final_status_message += (
                    "\nNo specific error message captured in STDERR."
                )

        output_message += f"--- STATUS ---\n{final_status_message}\n"
        output_message += "------------------------------------\n"
        output_message += "For full details, check the main application console and the project log file (e.g., project_ai_gui.log)."

        return output_message
    except subprocess.TimeoutExpired:
        logger.error(
            f"{script_name} timed out after {script_timeout} seconds. Command: {' '.join(command_list)}",
            exc_info=True,
        )
        return f"Error: {script_name} timed out after {script_timeout} seconds. The process was terminated. Check logs for more details."

    except FileNotFoundError as e:
        # This specifically catches if the script itself (e.g., "python" or "train.py") isn't found
        script_executable = command_list[0]
        script_path_arg = command_list[1] if len(command_list) > 1 else "N/A"
        logger.error(
            f"Failed to run {script_name}: Executable '{script_executable}' or script '{script_path_arg}' not found. Full command: {' '.join(command_list)}. Error: {e}",
            exc_info=True,
        )
        return f"Error launching {script_name}: Executable '{script_executable}' or script '{script_path_arg}' not found. Please ensure it exists and the path is correct."
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in the GUI script while trying to run {script_name}. Command: {' '.join(command_list)}",
            exc_info=True,
        )
        return f"Error in GUI while launching {script_name}: {type(e).__name__}: {e}\nTraceback: {traceback.format_exc()}\nThis is likely an issue in the GUI or environment, not the script itself."


def run_train_script_gui(stage, user_key, session_key_state):
    if not stage:
        logger.warning("Train script run attempted without selecting a stage.")
        return "Please select a stage for training."
    # Always repair dataset structure before training
    auto_repair_and_manage_datasets()
    data_dir = os.path.join("data", stage)
    # Get all available versions, newest first
    version_dirs = [d for d in os.listdir(data_dir) if re.match(r"^\d{8}_\d{6}$", d) and os.path.isdir(os.path.join(data_dir, d))]
    version_dirs.sort(reverse=True)
    if not version_dirs:
        return "No dataset versions found for this stage. Please upload data."
    # Try training on each version, rolling back if error detected
    for version in version_dirs:
        # Set latest.txt to this version
        latest_txt_path = os.path.join(data_dir, "latest.txt")
        with open(latest_txt_path, "w", encoding="utf-8") as f:
            f.write(version + "\n")
        command = [sys.executable, "train.py", "--stage", stage]
        result = run_script_in_background(
            command, user_key, session_key_state, f"Training Script (train.py) [version {version}]"
        )
        # Check for error/knowledge gap in result (simple heuristic: look for 'cannot proceed', 'error', or 'knowledge gap')
        if any(s in result.lower() for s in ["cannot proceed", "error", "knowledge gap", "no dataset version"]):
            logger.warning(f"Training failed or knowledge gap detected for version {version}. Trying previous version if available.")
            continue
        # If training succeeded, return result
        return result + f"\n[Trained on dataset version: {version}]"
    # If all versions failed, trigger research ("google"), repair datasets, and retry latest
    logger.warning("Training failed or knowledge gap detected for all available dataset versions. Triggering research and retrying training on latest version.")
    model.conduct_research()  # This will 'google' for new sources
    auto_repair_and_manage_datasets()  # Repair after research
    # Get latest version again (in case new data was added)
    version_dirs = [d for d in os.listdir(data_dir) if re.match(r"^\d{8}_\d{6}$", d) and os.path.isdir(os.path.join(data_dir, d))]
    version_dirs.sort(reverse=True)
    if not version_dirs:
        return "No dataset versions found after research. Please upload or discover new data."
    latest_version = version_dirs[0]
    latest_txt_path = os.path.join(data_dir, "latest.txt")
    with open(latest_txt_path, "w", encoding="utf-8") as f:
        f.write(latest_version + "\n")
    command = [sys.executable, "train.py", "--stage", stage]
    result = run_script_in_background(
        command, user_key, session_key_state, f"Training Script (train.py) [version {latest_version} after research]"
    )
    # Enhanced error handling for PyTorch embedding index error
    if "index out of range in self" in result.lower():
        logger.error("Training failed due to a data/model mismatch: PyTorch embedding index out of range. This is likely a tokenization or vocabulary bug, not a knowledge gap.")
        return (
            "Training failed due to a data/model mismatch: PyTorch embedding index out of range.\n"
            "This is likely caused by out-of-vocabulary tokens or a bug in your preprocessing/tokenization pipeline.\n"
            "Please check your dataset, tokenization, and model vocabulary size.\n\n"
            "Recommended debug steps:\n"
            "1. Check your embedding layer's vocab size in python_master_ai.py.\n"
            "2. Print the max/min token indices in your dataset just before model training.\n"
            "3. Ensure your tokenizer maps all unknown tokens to a valid <unk> index within the vocab.\n"
            "4. Add an assertion in your data loader: assert (x >= 0).all() and (x < vocab_size).all()\n"
            "5. If using a pre-trained tokenizer, ensure your data is compatible with its vocab.\n\n"
            + result
        )
    if any(s in result.lower() for s in ["cannot proceed", "error", "knowledge gap", "no dataset version"]):
        logger.error("Training still failed after research and retry. Human intervention may be required.")
        return (
            "Training failed or knowledge gap detected for all available dataset versions, even after research. "
            "Please check your data, sources, or consider manual intervention.\n" + result
        )
    return (
        result
        + f"\n[Trained on dataset version: {latest_version} after research/googling]"
        + "\nNote: Research was automatically triggered due to insufficient/inefficient data."
    )


def run_scrape_data_script_gui(
    stage, sources_str, urls_str, user_key, session_key_state
):
    if not stage:
        logger.warning("Scraper script run attempted without selecting a stage.")
        return "Please select a stage for scraping."
    if not sources_str.strip() or not urls_str.strip():
        logger.warning("Scraper script run attempted with empty sources or URLs.")
        return "Sources and URLs cannot be empty."

    sources_list = [s.strip() for s in sources_str.split(",") if s.strip()]
    urls_list = [u.strip() for u in urls_str.split(",") if u.strip()]

    if not sources_list or not urls_list:
        logger.warning(
            "Scraper script run attempted with empty sources or URLs after stripping."
        )
        return "After stripping, sources or URLs list is empty."

    if len(sources_list) != len(urls_list):
        logger.warning(
            f"Mismatch in number of sources and URLs for scraper script: {len(sources_list)} vs {len(urls_list)}"
        )
        return f"Mismatch between number of sources ({len(sources_list)}) and URLs ({len(urls_list)})."

    command = [sys.executable, "scrape_data.py", stage] + [
        item for pair in zip(sources_list, urls_list) for item in pair
    ]
    return run_script_in_background(
        command, user_key, session_key_state, "Scraping Script (scrape_data.py)"
    )


def run_evaluate_script_gui(
    model_checkpoint_path,
    eval_dataset_path,
    selected_stage_option,
    user_key,
    session_key_state,
):
    logger.info(
        f"Evaluation script run requested via GUI. Checkpoint: {model_checkpoint_path}, Dataset: {eval_dataset_path}, Stage Option: {selected_stage_option}"
    )
    if not model_checkpoint_path:
        return "Please provide the path to the model checkpoint."
    if not eval_dataset_path:
        return "Please provide the path to the evaluation dataset (.jsonl file)."

    command = [
        sys.executable,
        "evaluate.py",
        "--model_checkpoint_path",
        model_checkpoint_path,
        "--eval_dataset_path",
        eval_dataset_path,
    ]
    # Only add the --stage argument if a specific stage is selected (not the "Evaluate All Stages" option)
    if selected_stage_option and selected_stage_option != "(Evaluate All Stages)":
        command.extend(["--stage", selected_stage_option])

    return run_script_in_background(
        command, user_key, session_key_state, "Evaluation Script (evaluate.py)"
    )


def load_latest_model_from_checkpoint_gui(user_key, session_key_state):
    logger.info("Load latest model from checkpoint requested via GUI.")

    current_valid_key = (
        session_key_state if session_key_state == MASTER_KEY else user_key
    )
    if current_valid_key != MASTER_KEY:
        logger.warning(
            "Invalid Master key attempt in load_latest_model_from_checkpoint_gui."
        )
        return "Invalid Master key. Model reload aborted."

    status_message = (
        model._try_load_latest_checkpoint()
    )  # This method now logs internally too
    logger.info(f"Checkpoint load attempt status: {status_message}")
    return status_message


# --- Dataset Management Handler Functions ---
# These functions primarily return data for the UI, logging is for errors or important info.


def get_dataset_versions(stage_name: str):
    logger.debug(f"Fetching dataset versions for stage: {stage_name}")
    if not stage_name:
        return []
    data_dir = os.path.join("data", stage_name)
    if not os.path.isdir(data_dir):
        logger.warning(f"Stage directory not found for dataset versions: {data_dir}")
        return [
            {
                "id": "Error",
                "is_latest": False,
                "manifest_type": f"Stage directory not found: {data_dir}",
                "num_files": 0,
                "timestamp_from_manifest": "N/A",
            }
        ]
    latest_version = None
    latest_txt_path = os.path.join(data_dir, "latest.txt")
    if os.path.exists(latest_txt_path):
        with open(latest_txt_path, "r") as f:
            latest_version = f.read().strip()
    versions = []
    try:
        for version_dir_name in os.listdir(data_dir):
            version_path = os.path.join(data_dir, version_dir_name)
            if (
                os.path.isdir(version_path)
                and version_dir_name.replace("_", "").isdigit()
            ):
                manifest_path = os.path.join(version_path, "manifest.json")
                manifest_data = {}
                num_txt_files = 0
                if os.path.exists(manifest_path):
                    try:
                        with open(manifest_path, "r") as f_manifest:
                            manifest_data = json.load(f_manifest)
                    except json.decoder.JSONDecodeError as e:
                        manifest_data = {"error": f"Malformed manifest.json: {e}"}
                        logger.error(
                            f"Malformed JSON in manifest {manifest_path}: {e}",
                            exc_info=True,
                        )
                    except Exception as e:
                        manifest_data = {"error": f"Could not read manifest: {e}"}
                        logger.error(
                            f"Error reading manifest {manifest_path}: {e}",
                            exc_info=True,
                        )
                try:  # Try to get the number of txt files in the version
                    txt_files = [
                        f
                        for f in os.listdir(version_path)
                        if f.endswith(".txt") and f != "manifest.json"
                    ]
                    num_txt_files = len(txt_files)
                except Exception as e:
                    logger.error(
                        f"Error listing .txt files in {version_path}: {e}",
                        exc_info=True,
                    )
                versions.append(
                    {
                        "id": version_dir_name,
                        "is_latest": version_dir_name == latest_version,
                        "manifest_type": manifest_data.get(
                            "creation_event_type", manifest_data.get("error", "N/A")
                        ),
                        "num_files": num_txt_files,
                        "timestamp_from_manifest": manifest_data.get(
                            "version_timestamp", "N/A"
                        ),
                    }
                )
        versions.sort(key=lambda x: x.get("id"), reverse=True)
    except Exception as e:
        logger.error(
            f"Error listing versions for stage {stage_name}: {e}", exc_info=True
        )
        return [
            {
                "id": "Error",
                "is_latest": False,
                "manifest_type": f"Error listing versions: {e}",
                "num_files": 0,
                "timestamp_from_manifest": "N/A",
            }
        ]
    logger.debug(f"Found {len(versions)} versions for stage {stage_name}.")
    return versions


def get_files_in_version(stage_name: str, version_timestamp: str):
    logger.debug(
        f"Fetching files for stage '{stage_name}', version '{version_timestamp}'."
    )
    if not stage_name or not version_timestamp:
        return []
    version_dir = os.path.join("data", stage_name, version_timestamp)
    if not os.path.isdir(version_dir):
        logger.warning(f"Version directory not found: {version_dir}")
        return [
            {
                "name": "Error: Version directory not found.",
                "size": 0,
                "is_excluded": False,
            }
        ]
    manifest_path = os.path.join(version_dir, "manifest.json")
    excluded_files = []
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f_manifest:
                manifest_data = json.load(f_manifest)
            excluded_files = manifest_data.get("excluded_files", [])
        except json.decoder.JSONDecodeError as e:
            logger.warning(
                f"Malformed manifest.json for exclusions in {version_dir}: {e}",
                exc_info=True,
            )
            excluded_files = []
        except Exception as e:
            logger.warning(
                f"Could not read manifest for exclusions in {version_dir}: {e}",
                exc_info=True,
            )
    files_info = []
    try:
        for filename in os.listdir(version_dir):
            if filename.endswith(".txt") and filename != "manifest.json":
                file_path = os.path.join(version_dir, filename)
                try:
                    size = os.path.getsize(file_path)
                    files_info.append(
                        {
                            "name": filename,
                            "size": size,
                            "is_excluded": filename in excluded_files,
                        }
                    )
                except OSError as e:
                    logger.error(
                        f"Error accessing file {file_path}: {e}", exc_info=True
                    )
                    files_info.append(
                        {
                            "name": f"{filename} (Error accessing)",
                            "size": 0,
                            "is_excluded": filename in excluded_files,
                        }
                    )
        files_info.sort(key=lambda x: x["name"])
    except Exception as e:
        logger.error(f"Error listing files in {version_dir}: {e}", exc_info=True)
        return [{"name": f"Error listing files: {e}", "size": 0, "is_excluded": False}]
    logger.debug(f"Found {len(files_info)} files in {version_dir}.")
    return files_info


def get_file_content_gui(stage_name: str, version_timestamp: str, filename: str):
    logger.debug(
        f"Fetching content for file '{filename}' in stage '{stage_name}', version '{version_timestamp}'."
    )
    if not stage_name or not version_timestamp or not filename:
        return "Missing stage, version, or filename."
    file_path = os.path.join("data", stage_name, version_timestamp, filename)
    base_dir = os.path.abspath(os.path.join("data", stage_name, version_timestamp))
    abs_file_path = os.path.abspath(file_path)
    if os.path.commonpath([abs_file_path, base_dir]) != base_dir:
        logger.warning(f"Attempted directory traversal: {file_path}")
        return "Error: File path is outside the allowed directory."
    if not os.path.exists(file_path):
        logger.warning(f"File not found for content preview: {file_path}")
        return f"Error: File not found: {file_path}"
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:  # type: ignore # errors='ignore' for robustness
            content_lines = []
            total_bytes = 0

            # Get truncation limits from config, with defaults
            default_max_bytes = 1024 * 1024  # 1MB
            default_max_lines = 200
            max_bytes_limit = get_typed_config_value(
                "gui_settings.file_preview.max_bytes", default_max_bytes, int
            )
            max_lines_limit = get_typed_config_value(
                "gui_settings.file_preview.max_lines", default_max_lines, int
            )

            for i, line in enumerate(f):
                if i >= max_lines_limit:
                    content_lines.append(
                        f"...\n[Preview truncated: Max lines ({max_lines_limit}) reached]"
                    )
                    break
                line_bytes = len(line.encode("utf-8"))
                # Check if adding this line would exceed the byte limit
                # Ensure at least one line is processed if it's huge, unless it's the very first line and it alone exceeds.
                if (
                    total_bytes + line_bytes > max_bytes_limit and i > 0
                ):  # If not the first line and adding it exceeds
                    content_lines.append(
                        f"...\n[Preview truncated: Max bytes ({max_bytes_limit}) reached before this line]"
                    )
                    break
                content_lines.append(line)
                total_bytes += line_bytes
                if (
                    total_bytes > max_bytes_limit and i == 0
                ):  # If the very first line itself is too big
                    content_lines[-1] = content_lines[-1][
                        :max_bytes_limit
                    ]  # Approximate truncation
                    content_lines.append(
                        f"...\n[Preview truncated: First line exceeded max bytes ({max_bytes_limit})]"
                    )
                    break
                elif (
                    total_bytes > max_bytes_limit
                ):  # If accumulated size exceeds after adding current line
                    content_lines.append(
                        f"...\n[Preview truncated: Max bytes ({max_bytes_limit}) reached]"
                    )
                    break
            return "".join(content_lines)
    except Exception as e:
        logger.error(f"Error reading file {filename} for preview: {e}", exc_info=True)
        return f"Error reading file {filename}: {e}"


def get_manifest_content_gui(stage_name: str, version_timestamp: str):
    logger.debug(
        f"Fetching manifest for stage '{stage_name}', version '{version_timestamp}'."
    )
    if not stage_name or not version_timestamp:
        return "Missing stage or version."
    manifest_path = os.path.join("data", stage_name, version_timestamp, "manifest.json")
    if not os.path.exists(manifest_path):
        logger.warning(f"Manifest.json not found: {manifest_path}")
        return f"Error: manifest.json not found in {os.path.dirname(manifest_path)}"
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        return json.dumps(manifest_data, indent=4)
    except json.decoder.JSONDecodeError as e:
        logger.error(
            f"Malformed JSON in manifest.json {manifest_path}: {e}", exc_info=True
        )
        return f"Error: manifest.json is corrupted or contains malformed JSON. Please fix or delete the file. Details: {e}"
    except Exception as e:
        logger.error(f"Error reading manifest.json {manifest_path}: {e}", exc_info=True)
        return f"Error reading manifest.json: {e}"


def set_latest_version_gui(
    stage_name: str,
    version_timestamp: str,
    master_key_input: str,
    session_key_state: gr.State,
):
    logger.info(
        f"Attempting to set version '{version_timestamp}' as latest for stage '{stage_name}'."
    )

    current_valid_key = session_key_state if session_key_state == MASTER_KEY else master_key_input  # type: ignore
    if current_valid_key != MASTER_KEY:
        logger.warning("Invalid Master key in set_latest_version_gui.")
        return "Invalid Master key."
    if not stage_name or not version_timestamp:  # type: ignore
        return "Stage name and version timestamp must be provided."
    stage_data_dir = os.path.join("data", stage_name)
    version_path_to_check = os.path.join(stage_data_dir, version_timestamp)
    if not os.path.isdir(version_path_to_check):
        logger.error(f"Version directory '{version_path_to_check}' does not exist.")
        return f"Error: Version directory '{version_timestamp}' does not exist in stage '{stage_name}'."
    latest_txt_path = os.path.join(stage_data_dir, "latest.txt")
    try:
        with open(latest_txt_path, "w", encoding="utf-8") as f:
            f.write(version_timestamp)
        logger.info(
            f"Successfully set version '{version_timestamp}' as latest for stage '{stage_name}'."
        )
        return f"Successfully set version '{version_timestamp}' as latest for stage '{stage_name}'."
    except Exception as e:
        logger.error(
            f"Error updating latest.txt for stage '{stage_name}': {e}", exc_info=True
        )
        return f"Error updating latest.txt: {e}"


def toggle_file_exclusion_gui(
    stage_name: str,
    version_timestamp: str,
    filename: str,
    master_key_input: str,
    session_key_state: gr.State,
):
    logger.info(
        f"Attempting to toggle exclusion for file '{filename}' in stage '{stage_name}', version '{version_timestamp}'."
    )

    current_valid_key = session_key_state if session_key_state == MASTER_KEY else master_key_input  # type: ignore
    if current_valid_key != MASTER_KEY:
        logger.warning("Invalid Master key in toggle_file_exclusion_gui.")
        return "Invalid Master key.", False

    if not stage_name or not version_timestamp or not filename:  # type: ignore
        return "Stage, version, and filename must be provided.", False
    manifest_path = os.path.join("data", stage_name, version_timestamp, "manifest.json")
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest.json not found for toggling exclusion: {manifest_path}")
        return (
            f"Error: manifest.json not found in {os.path.dirname(manifest_path)}.",
            False,
        )
    try:
        with open(manifest_path, "r+", encoding="utf-8") as f:
            try:
                manifest_data = json.load(f)
            except json.decoder.JSONDecodeError as e:
                logger.error(
                    f"Malformed JSON in manifest.json {manifest_path}: {e}",
                    exc_info=True,
                )
                return (
                    f"Error: manifest.json is corrupted or contains malformed JSON. Please fix or delete the file. Details: {e}",
                    False,
                )
            excluded_files = manifest_data.get("excluded_files", [])
            new_exclusion_status = False
            if filename in excluded_files:
                excluded_files.remove(filename)
                action_message = f"File '{filename}' UNEXCLUDED."
            else:
                excluded_files.append(filename)
                action_message = f"File '{filename}' EXCLUDED."
                new_exclusion_status = True
            manifest_data["excluded_files"] = sorted(list(set(excluded_files)))
            f.seek(0)
            json.dump(manifest_data, f, indent=4)
            f.truncate()
        logger.info(
            f"Successfully updated exclusion for '{filename}'. {action_message}"
        )
        return (
            f"Successfully updated exclusion for '{filename}'. {action_message}",
            new_exclusion_status,
        )
    except Exception as e:
        logger.error(f"Error toggling exclusion for '{filename}': {e}", exc_info=True)
        return f"Error toggling exclusion for '{filename}': {e}", False


def list_manual_uploads(
    target_stage, user_key, session_key_state
):  # This function is mostly for direct user feedback.
    logger.debug(f"Listing manual uploads for stage: {target_stage}")

    current_valid_key = (
        session_key_state if session_key_state == MASTER_KEY else user_key
    )
    if current_valid_key != MASTER_KEY:
        return "Invalid Master key"
    if not target_stage:
        return "Please select a stage to view uploads for."
    data_dir = f"data/{target_stage}"
    if not os.path.exists(data_dir):
        return (
            f"No data directory found for stage '{target_stage}'. No uploads to list."
        )
    manual_files = []
    try:
        for item_name in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item_name)
            if (
                os.path.isfile(item_path)
                and item_name.startswith("manual_upload_")
                and item_name.endswith(".txt")
            ):
                manual_files.append(item_name)
    except Exception as e:
        logger.error(
            f"Error reading data directory for stage '{target_stage}': {e}",
            exc_info=True,
        )
        return f"Error reading data directory for stage '{target_stage}': {e}"
    if not manual_files:
        return f"No manually uploaded files found directly in '{data_dir}' matching 'manual_upload_*.txt'."
    else:
        return f"Manually uploaded files directly in '{data_dir}':\n" + "\n".join(
            manual_files
        )


available_stages = list(model.define_growth_tasks().keys())
logger.info(f"Available stages for GUI: {available_stages}")

with gr.Blocks(title="PythonMasterAI: Serving Master Daddy") as iface:
    gr.Markdown("## PythonMasterAI Control Panel")

    # Master Key Management
    session_master_key = gr.State(None)  # Stores validated master key for the session
    with gr.Row(variant="panel"):
        gr.Markdown("### Master Key Validation (Required for most actions)")
        master_key_input_global = gr.Textbox(
            label="Enter Master Key",
            type="password",
            placeholder="Enter Master Key to enable actions",
        )
        validate_key_button = gr.Button("Validate Master Key for Session")
        key_validation_status_output = gr.Textbox(label="Key Status", interactive=False)
        validate_key_button.click(
            validate_master_key,
            inputs=[master_key_input_global, session_master_key],
            outputs=[session_master_key, key_validation_status_output],
        )

    with gr.Tab("Master Commands"):
        cmd_input = gr.Textbox(
            label="Command", placeholder="Enter MASTER: or HALT_TEEN"
        )
        cmd_output = gr.Textbox(label="Response")
        cmd_button = gr.Button("Execute")
        cmd_button.click(
            interact_with_ai,
            inputs=[cmd_input, master_key_input_global, session_master_key],
            outputs=cmd_output,
        )
    with gr.Tab("Research"):
        research_button = gr.Button("Trigger Research")
        research_output = gr.Textbox(label="Research Logs")
        research_button.click(trigger_research, outputs=research_output)
    with gr.Tab("Source Discovery"):
        source_button = gr.Button("Discover Sources")
        source_output = gr.Textbox(label="Source Logs")
        source_button.click(discover_sources, outputs=source_output)
    with gr.Tab("Status"):
        status_button = gr.Button("Get Status")
        status_output = gr.Textbox(label="AI Status")
        status_button.click(get_status, outputs=status_output)
    with gr.Tab("Growth"):
        growth_button = gr.Button("Approve Growth")
        growth_output = gr.Textbox(label="Growth Status")
        growth_button.click(
            approve_growth,
            inputs=[master_key_input_global, session_master_key],
            outputs=growth_output,
        )
    with gr.Tab("Manual Data Upload"):
        gr.Markdown(
            "Upload training documents (e.g., .txt, .py, .md, .pdf). Content will be extracted and saved as .txt files within a new versioned dataset."
        )
        upload_files = gr.File(label="Upload Training Files", file_count="multiple")
        upload_stage_select = gr.Dropdown(
            choices=available_stages,
            label="Target Stage for Uploaded Data",
            value=model.stage,
        )
        upload_button = gr.Button("Upload and Save Files")
        upload_output = gr.Textbox(label="Upload Status", lines=5)
        upload_button.click(
            handle_manual_upload,
            inputs=[
                upload_files,
                upload_stage_select,
                master_key_input_global,
                session_master_key,
            ],
            outputs=upload_output,
        )
        gr.Markdown("---")
        gr.Markdown(
            "The `list_manual_uploads` button below might be deprecated or refer to non-versioned uploads. Use Dataset Management tab for versioned data."
        )
        view_uploads_button = gr.Button(
            "View Legacy Manually Uploaded Files for Selected Stage"
        )
        view_uploads_output = gr.Textbox(
            label="List of Legacy Uploaded Files", lines=5, interactive=False
        )
        view_uploads_button.click(
            list_manual_uploads,
            inputs=[upload_stage_select, master_key_input_global, session_master_key],
            outputs=view_uploads_output,
        )
    with gr.Tab("Run Training Script"):
        gr.Markdown(
            "Run the `train.py` script. This will execute in a separate process."
        )
        train_stage_select = gr.Dropdown(
            choices=available_stages,
            label="Select Stage for Training",
            value=model.stage,
        )
        train_run_button = gr.Button("Start Training Script")
        train_run_output = gr.Textbox(
            label="Training Script Output", lines=10, interactive=False
        )
        train_run_button.click(
            run_train_script_gui,
            inputs=[train_stage_select, master_key_input_global, session_master_key],
            outputs=train_run_output,
        )
    with gr.Tab("Run Scraper Script"):
        gr.Markdown(
            "Run the `scrape_data.py` script. This will execute in a separate process."
        )
        scrape_stage_select = gr.Dropdown(
            choices=available_stages,
            label="Select Stage for Scraping",
            value=model.stage,
        )
        scrape_sources_input = gr.Textbox(
            label="Sources (comma-separated)",
            placeholder="e.g., github_beginner,study_guides",
        )
        scrape_urls_input = gr.Textbox(
            label="URLs (comma-separated, in same order as sources)",
            placeholder="e.g., http://url1.com,http://url2.com",
        )
        scrape_run_button = gr.Button("Start Scraper Script")
        scrape_run_output = gr.Textbox(
            label="Scraper Script Output", lines=10, interactive=False
        )
        scrape_run_button.click(
            run_scrape_data_script_gui,
            inputs=[
                scrape_stage_select,
                scrape_sources_input,
                scrape_urls_input,
                master_key_input_global,
                session_master_key,
            ],
            outputs=scrape_run_output,
        )

    with gr.Tab("Run Evaluation Script"):
        gr.Markdown(
            "Run the `evaluate.py` script against a specified model checkpoint and evaluation dataset. This will execute in a separate process."
        )
        eval_checkpoint_path_input = gr.Textbox(
            label="Model Checkpoint Path (.pt)",
            placeholder="e.g., checkpoints/model_stage_baby_config_xxxx_latest.pt",
        )
        eval_dataset_path_input = gr.Textbox(
            label="Evaluation Dataset Path (.jsonl)",
            placeholder="e.g., sample_evaluation_dataset.jsonl or custom_eval_data.jsonl",
        )
        eval_stage_choices = ["(Evaluate All Stages)"] + available_stages
        eval_stage_select = gr.Dropdown(
            choices=eval_stage_choices,
            label="Select Stage to Filter (Optional)",
            value="(Evaluate All Stages)",
        )  # Default to evaluating all
        eval_run_button = gr.Button("Start Evaluation Script")
        eval_run_output = gr.Textbox(
            label="Evaluation Script Output", lines=10, interactive=False
        )
        eval_run_button.click(
            run_evaluate_script_gui,
            inputs=[
                eval_checkpoint_path_input,
                eval_dataset_path_input,
                eval_stage_select,
                master_key_input_global,
                session_master_key,
            ],
            outputs=eval_run_output,
        )

    with gr.Tab("Model Management"):
        gr.Markdown(
            "Manage model checkpoints. Note: The model automatically attempts to load the latest compatible checkpoint on startup."
        )
        load_checkpoint_button = gr.Button(
            "Load Latest Model from Checkpoint for Current Stage & Configuration"
        )
        load_checkpoint_status_output = gr.Textbox(
            label="Load Status", interactive=False, lines=3
        )
        load_checkpoint_button.click(
            load_latest_model_from_checkpoint_gui,
            inputs=[master_key_input_global, session_master_key],
            outputs=[load_checkpoint_status_output],
        )
    with gr.Tab("Dataset Management"):
        gr.Markdown("## Dataset Version Explorer and Management")
        with gr.Row():
            dm_stage_select = gr.Dropdown(
                choices=available_stages, label="Select Stage", value=model.stage
            )
            dm_refresh_versions_btn = gr.Button("Refresh Versions List")
        dm_versions_df = gr.DataFrame(
            headers=[
                "Version ID",
                "Is Latest?",
                "Manifest Type",
                "# Files",
                "Manifest Timestamp",
            ],
            datatype=["str", "bool", "str", "number", "str"],
            label="Available Dataset Versions for Stage",
        )
        dm_selected_version_id_state = gr.State(None)
        dm_stage_select.change(
            get_dataset_versions, inputs=[dm_stage_select], outputs=[dm_versions_df]
        )
        dm_refresh_versions_btn.click(
            get_dataset_versions, inputs=[dm_stage_select], outputs=[dm_versions_df]
        )
        with gr.Row():
            dm_version_select_for_actions = gr.Dropdown(
                label="Select Version (for actions below)", interactive=True
            )
            dm_set_latest_btn = gr.Button("Set Selected Version as Latest")
            dm_view_manifest_btn = gr.Button("View Full Manifest of Selected Version")
        dm_set_latest_status_text = gr.Textbox(
            label="Set Latest Status", interactive=False
        )
        dm_manifest_content_text = gr.Textbox(
            label="Manifest Content", lines=10, interactive=False
        )

        def update_version_action_dropdown(versions_data):
            if versions_data is not None and not versions_data.empty:
                version_ids = versions_data["Version ID"].tolist()
                return gr.Dropdown.update(choices=version_ids, value=version_ids[0] if version_ids else None)  # type: ignore[attr-defined]
            return gr.Dropdown.update(choices=[], value=None)  # type: ignore[attr-defined]

        dm_versions_df.change(update_version_action_dropdown, inputs=[dm_versions_df], outputs=[dm_version_select_for_actions])  # type: ignore
        dm_set_latest_btn.click(
            set_latest_version_gui,
            inputs=[
                dm_stage_select,
                dm_version_select_for_actions,
                master_key_input_global,
                session_master_key,
            ],
            outputs=[dm_set_latest_status_text],
        ).then(get_dataset_versions, inputs=[dm_stage_select], outputs=[dm_versions_df])
        dm_view_manifest_btn.click(
            get_manifest_content_gui,
            inputs=[dm_stage_select, dm_version_select_for_actions],
            outputs=[dm_manifest_content_text],
        )
        gr.Markdown("---")
        gr.Markdown("### Files in Selected Version")
        dm_files_df = gr.DataFrame(
            headers=["Filename", "Size (bytes)", "Excluded?"],
            datatype=["str", "number", "bool"],
            label="Files in Selected Dataset Version",
        )
        dm_version_select_for_actions.change(
            lambda stage, version: (get_files_in_version(stage, version), version),
            inputs=[dm_stage_select, dm_version_select_for_actions],
            outputs=[dm_files_df, dm_selected_version_id_state],
        )
        with gr.Row():
            dm_file_select_for_actions = gr.Dropdown(
                label="Select File (for actions below)", interactive=True
            )
            dm_toggle_exclusion_btn = gr.Button("Toggle Exclusion of Selected File")
        dm_toggle_exclusion_status_text = gr.Textbox(
            label="Toggle Exclusion Status", interactive=False
        )
        dm_file_content_text = gr.Textbox(
            label="File Content Preview", lines=15, interactive=False
        )

        def update_file_action_dropdown(files_data):
            if files_data is not None and not files_data.empty:
                filenames = files_data["Filename"].tolist()
                valid_filenames = [
                    fn for fn in filenames if not fn.startswith("Error:")
                ]
                return gr.Dropdown.update(choices=valid_filenames, value=valid_filenames[0] if valid_filenames else None)  # type: ignore[attr-defined]
            return gr.Dropdown.update(choices=[], value=None)  # type: ignore[attr-defined]

        dm_files_df.change(
            update_file_action_dropdown,
            inputs=[dm_files_df],
            outputs=[dm_file_select_for_actions],
        )
        dm_file_select_for_actions.change(get_file_content_gui, inputs=[dm_stage_select, dm_selected_version_id_state, dm_file_select_for_actions], outputs=[dm_file_content_text])  # type: ignore
        dm_toggle_exclusion_btn.click(
            toggle_file_exclusion_gui,
            inputs=[
                dm_stage_select,
                dm_selected_version_id_state,
                dm_file_select_for_actions,
                master_key_input_global,
                session_master_key,
            ],
            outputs=[dm_toggle_exclusion_status_text],
        ).then(
            lambda stage, version: get_files_in_version(stage, version),
            inputs=[dm_stage_select, dm_selected_version_id_state],
            outputs=[dm_files_df],
        )

def auto_repair_and_manage_datasets(stage_dir_base="data"):    
    """
    Scan all stage directories, ensure each versioned dataset dir has a manifest.json, and latest.txt points to the newest version.
    Optionally, can be extended to archive used data and support rollback.
    """
    import os
    import json
    import re
    if not os.path.isdir(stage_dir_base):
        return
    for stage in os.listdir(stage_dir_base):
        stage_dir = os.path.join(stage_dir_base, stage)
        if not os.path.isdir(stage_dir):
            continue
        # Find all versioned subdirs (YYYYMMDD_HHMMSS)
        version_dirs = [d for d in os.listdir(stage_dir) if re.match(r"^\d{8}_\d{6}$", d) and os.path.isdir(os.path.join(stage_dir, d))]
        if not version_dirs:
            continue
        version_dirs.sort()
        latest_version = version_dirs[-1]
        # Ensure manifest.json exists in each version dir
        for vdir in version_dirs:
            vdir_path = os.path.join(stage_dir, vdir)
            manifest_path = os.path.join(vdir_path, "manifest.json")
            if not os.path.exists(manifest_path):
                txt_files = [f for f in os.listdir(vdir_path) if f.endswith(".txt")]
                manifest = {
                    "version_timestamp": vdir,
                    "dataset_path_relative_to_stage_dir": vdir,
                    "creation_event_type": "auto_repair",
                    "stage_uploaded_for": stage,
                    "uploaded_files_summary": {
                        "count": len(txt_files),
                        "original_filenames_provided": [],
                        "saved_filenames_in_version_dir": txt_files,
                    },
                    "excluded_files": []
                }
                with open(manifest_path, "w", encoding="utf-8") as mf:
                    json.dump(manifest, mf, indent=4)
        # Ensure latest.txt exists and is correct
        latest_txt_path = os.path.join(stage_dir, "latest.txt")
        try:
            with open(latest_txt_path, "w", encoding="utf-8") as f:
                f.write(latest_version + "\n")
        except Exception as e:
            print(f"[auto_repair] Could not update latest.txt for {stage}: {e}")

# Call at startup
auto_repair_and_manage_datasets()

logger.info("Launching Gradio interface...")
iface.launch()
