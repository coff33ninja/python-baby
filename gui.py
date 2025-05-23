# gui.py
import gradio as gr
from python_master_ai import PythonMasterAI
import json
from grow import grow_model
import os # Added for path operations
from datetime import datetime # Added for dataset versioning
import subprocess
import sys
import traceback

# Attempt to import PyPDF2 for PDF processing
try:
    from PyPDF2 import PdfReader
    pdf_support_available = True
except ImportError:
    PdfReader = None # Define for type hinting and conditional checks
    pdf_support_available = False
    print("WARNING: PyPDF2 not installed. PDF upload support will be disabled. To enable, run: pip install PyPDF2")

model = PythonMasterAI()
MASTER_KEY = PythonMasterAI.MASTER_KEY


def interact_with_ai(command, user_key):
    if user_key != MASTER_KEY:
        return "Invalid Master key"
    return model.process_input(command, MASTER_KEY)


def trigger_research():
    model.conduct_research()
    with open("research_log.json", "r") as f:
        logs = [json.loads(line) for line in f]
    return json.dumps(logs[-5:], indent=2)


def discover_sources():
    new_sources = model.discover_new_sources()
    discovered_message = f"Discovered sources in this run: {new_sources}\n\n" if new_sources else "No new sources discovered in this run.\n\n"
    with open("source_log.json", "r") as f:
        logs = [json.loads(line) for line in f]
    return discovered_message + "Last 5 source log entries:\n" + json.dumps(logs[-5:], indent=2)


def get_status():
    return model.get_status()


def approve_growth(user_key):
    global model # Declare intent to use the global 'model' variable
    if user_key != MASTER_KEY:
        return "Invalid Master key"
    model, _ = grow_model(model)
    return f"{model.stage.capitalize()} growth approved!"

def handle_manual_upload(files, target_stage, user_key):
    if user_key != MASTER_KEY:
        return "Invalid Master key"
    if not files:
        return "No files were uploaded."
    if not target_stage:
        return "Please select a target stage for the data."

    # --- Dataset Versioning ---
    version_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage_base_dir = os.path.join("data", target_stage)
    version_data_dir = os.path.join(stage_base_dir, version_timestamp)
    os.makedirs(version_data_dir, exist_ok=True)
    
    results = []
    saved_filenames_for_manifest = []

    for uploaded_file in files:
        actual_original_filename = uploaded_file.name 
        temp_file_path = str(uploaded_file) # Gradio provides a temp path for the uploaded file

        original_filename_basename = os.path.basename(actual_original_filename)
        content_to_save = None

        # Generate a descriptive .txt filename (remains the same logic)
        original_filename_stem, original_extension = os.path.splitext(original_filename_basename)
        safe_stem = "".join(c if c.isalnum() or c == '_' else '_' for c in original_filename_stem)
        safe_ext = "".join(c if c.isalnum() else '' for c in original_extension.replace('.', ''))
        
        # Ensure unique filename within the versioned directory, though original logic might be sufficient
        # if safe_stem + safe_ext is unique enough for a single upload batch.
        # We'll save it with a generic "manual_upload_" prefix for clarity.
        descriptive_part = f"{safe_stem}_{safe_ext}" if safe_ext else safe_stem
        save_filename = f"manual_upload_{descriptive_part}.txt"
        # Save path is now within the version_data_dir
        save_path = os.path.join(version_data_dir, save_filename)


        if original_filename_basename.lower().endswith(".pdf"):
            if pdf_support_available and PdfReader is not None:
                try:
                    reader = PdfReader(temp_file_path)
                    pdf_text_parts = [page.extract_text() or "" for page in reader.pages]
                    content_to_save = "\n".join(pdf_text_parts).strip()
                    if not content_to_save:
                        results.append(f"Info: Extracted no text from PDF '{actual_original_filename}'. An empty .txt file will be saved.")
                except Exception as pdf_e:
                    results.append(f"Error parsing PDF '{actual_original_filename}': {pdf_e}. An empty .txt file will be saved if possible.")
                    content_to_save = "" # Fallback to save an empty file
            else:
                results.append(f"Info: PDF support not available (PyPDF2 not installed). Skipping PDF '{actual_original_filename}'.")
                continue # Skip to the next file
        else:  # Assume other files are text-based
            try:
                with open(temp_file_path, "r", encoding="utf-8", errors="ignore") as f_in:
                    content_to_save = f_in.read()
            except Exception as text_e:
                results.append(f"Error reading text file '{actual_original_filename}': {text_e}. An empty .txt file will be saved if possible.")
                content_to_save = "" # Fallback to save an empty file

        # Proceed to save if content_to_save has been set (even if it's an empty string from an error)
        if content_to_save is not None:
            try:
                with open(save_path, "w", encoding="utf-8") as f_out:
                    f_out.write(content_to_save)
                results.append(f"Processed '{actual_original_filename}' and saved as '{save_filename}' in version {version_timestamp}.")
                saved_filenames_for_manifest.append(save_filename)
            except Exception as save_e:
                results.append(f"Error saving content from '{actual_original_filename}' to '{save_filename}': {save_e}")
        
    # --- Manifest and latest.txt Creation ---
    manifest_content = {
        "version_timestamp": version_timestamp,
        "dataset_path_relative_to_stage_dir": version_timestamp,
        "creation_event_type": "manual_upload",
        "stage_uploaded_for": target_stage,
        "uploaded_files_summary": {
            "count": len(saved_filenames_for_manifest),
            "original_filenames_provided": [f.name for f in files], # Original names from upload
            "saved_filenames_in_version_dir": saved_filenames_for_manifest
        }
    }
    manifest_filepath = os.path.join(version_data_dir, "manifest.json")
    try:
        with open(manifest_filepath, "w", encoding="utf-8") as f_manifest:
            json.dump(manifest_content, f_manifest, indent=4)
        results.append(f"Saved manifest to {manifest_filepath}")
    except Exception as e:
        results.append(f"Error saving manifest.json: {e}")

    latest_txt_path = os.path.join(stage_base_dir, "latest.txt")
    try:
        with open(latest_txt_path, "w", encoding="utf-8") as f_latest:
            f_latest.write(version_timestamp)
        results.append(f"Updated 'latest.txt' in {stage_base_dir} to version {version_timestamp}")
    except Exception as e:
        results.append(f"Error updating latest.txt: {e}")

    final_message = "\n".join(results)
    final_message += f"\n\nNote: Manually uploaded data for stage '{target_stage}' saved to version '{version_timestamp}'. This version is now set as the latest."
    return final_message

def run_script_in_background(command_list, user_key, script_name):
    if user_key != MASTER_KEY:
        return "Invalid Master key"
    try:
        print(f"Running command: {' '.join(command_list)}")
        process = subprocess.run(command_list, capture_output=True, text=True, check=False, encoding='utf-8')
        output = f"--- {script_name} STDOUT ---\n{process.stdout}\n"
        if process.stderr:
            output += f"--- {script_name} STDERR ---\n{process.stderr}\n"
        
        if process.returncode == 0:
            return f"{script_name} completed successfully.\n{output}"
        else:
            return f"{script_name} failed with return code {process.returncode}.\n{output}"
    except Exception as e:
        return f"Failed to run {script_name}: {e}\nTraceback: {traceback.format_exc()}"

def run_train_script_gui(stage, user_key):
    if not stage:
        return "Please select a stage for training."
    command = [sys.executable, "train.py", "--stage", stage]
    return run_script_in_background(command, user_key, "Training Script (train.py)")

def run_scrape_data_script_gui(stage, sources_str, urls_str, user_key):
    if not stage:
        return "Please select a stage for scraping."
    if not sources_str.strip() or not urls_str.strip():
        return "Sources and URLs cannot be empty. Please provide at least one source and its corresponding URL."
        
    sources_list = [s.strip() for s in sources_str.split(',') if s.strip()]
    urls_list = [u.strip() for u in urls_str.split(',') if u.strip()]

    if not sources_list or not urls_list:
        return "After stripping, sources or URLs list is empty. Please provide valid, comma-separated values."

    if len(sources_list) != len(urls_list):
        return f"Mismatch between number of sources ({len(sources_list)}) and URLs ({len(urls_list)}). Please provide one URL per source."

    command = [sys.executable, "scrape_data.py", stage]
    for source, url in zip(sources_list, urls_list):
        command.append(source)
        command.append(url)
    
    return run_script_in_background(command, user_key, "Scraping Script (scrape_data.py)")

def load_latest_model_from_checkpoint_gui(user_key):
    if user_key != MASTER_KEY:
        return "Invalid Master key. Model reload aborted."
    
    # model is the global PythonMasterAI instance
    # _try_load_latest_checkpoint now returns a status message
    status_message = model._try_load_latest_checkpoint()
    return status_message

# --- Dataset Management Handler Functions ---

def get_dataset_versions(stage_name: str):
    if not stage_name:
        return [] # Or an error message suitable for Gradio output
    data_dir = os.path.join("data", stage_name)
    if not os.path.isdir(data_dir):
        return [{"id": "Error", "is_latest": False, "manifest": f"Stage directory not found: {data_dir}", "num_files": 0}]

    latest_version = None
    latest_txt_path = os.path.join(data_dir, "latest.txt")
    if os.path.exists(latest_txt_path):
        with open(latest_txt_path, "r") as f:
            latest_version = f.read().strip()

    versions = []
    try:
        for version_dir_name in os.listdir(data_dir):
            version_path = os.path.join(data_dir, version_dir_name)
            if os.path.isdir(version_path) and version_dir_name.replace("_", "").isdigit(): # Basic check for version format
                manifest_path = os.path.join(version_path, "manifest.json")
                manifest_data = {}
                num_txt_files = 0
                if os.path.exists(manifest_path):
                    try:
                        with open(manifest_path, "r") as f_manifest:
                            manifest_data = json.load(f_manifest)
                    except Exception as e:
                        manifest_data = {"error": f"Could not read manifest: {e}"}
                
                try:
                    txt_files = [f for f in os.listdir(version_path) if f.endswith(".txt") and f != "manifest.json"]
                    num_txt_files = len(txt_files)
                except Exception:
                    pass # num_txt_files remains 0

                versions.append({
                    "id": version_dir_name,
                    "is_latest": version_dir_name == latest_version,
                    "manifest_type": manifest_data.get("creation_event_type", "N/A"),
                    "num_files": num_txt_files,
                    "timestamp_from_manifest": manifest_data.get("version_timestamp", "N/A") # For cross-check
                })
        versions.sort(key=lambda x: x.get("id"), reverse=True) # Show newest first
    except Exception as e:
        return [{"id": "Error", "is_latest": False, "manifest": f"Error listing versions: {e}", "num_files": 0}]
    return versions


def get_files_in_version(stage_name: str, version_timestamp: str):
    if not stage_name or not version_timestamp:
        return []
    version_dir = os.path.join("data", stage_name, version_timestamp)
    if not os.path.isdir(version_dir):
        return [{"name": "Error: Version directory not found.", "size": 0, "is_excluded": False}]

    manifest_path = os.path.join(version_dir, "manifest.json")
    excluded_files = []
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f_manifest:
                manifest_data = json.load(f_manifest)
                excluded_files = manifest_data.get("excluded_files", [])
        except Exception as e:
            # Allow proceeding to list files, but exclusion info might be missing
            print(f"Warning: Could not read manifest for exclusions in {version_dir}: {e}")


    files_info = []
    try:
        for filename in os.listdir(version_dir):
            if filename.endswith(".txt") and filename != "manifest.json": # Exclude manifest itself
                file_path = os.path.join(version_dir, filename)
                try:
                    size = os.path.getsize(file_path)
                    files_info.append({
                        "name": filename,
                        "size": size,
                        "is_excluded": filename in excluded_files
                    })
                except OSError: # File might be a broken symlink or inaccessible
                    files_info.append({
                        "name": f"{filename} (Error accessing)",
                        "size": 0,
                        "is_excluded": filename in excluded_files 
                    })
        files_info.sort(key=lambda x: x["name"])
    except Exception as e:
        return [{"name": f"Error listing files: {e}", "size": 0, "is_excluded": False}]
    return files_info


def get_file_content_gui(stage_name: str, version_timestamp: str, filename: str):
    if not stage_name or not version_timestamp or not filename:
        return "Missing stage, version, or filename."
    file_path = os.path.join("data", stage_name, version_timestamp, filename)
    
    # Prevent directory traversal
    base_dir = os.path.abspath(os.path.join("data", stage_name, version_timestamp))
    abs_file_path = os.path.abspath(file_path)
    if os.path.commonpath([abs_file_path, base_dir]) != base_dir:
        return "Error: File path is outside the allowed directory."

    if not os.path.exists(file_path):
        return f"Error: File not found: {file_path}"
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            # Limit preview size (e.g., first 200 lines or 1MB)
            content_lines = []
            total_bytes = 0
            max_bytes = 1024 * 1024 # 1MB
            max_lines = 200
            for i, line in enumerate(f):
                if i >= max_lines:
                    content_lines.append("...\n[Preview truncated: Too many lines]")
                    break
                line_bytes = len(line.encode('utf-8'))
                if total_bytes + line_bytes > max_bytes:
                    content_lines.append("...\n[Preview truncated: File size limit reached]")
                    break
                content_lines.append(line)
                total_bytes += line_bytes
            return "".join(content_lines)
    except Exception as e:
        return f"Error reading file {filename}: {e}"


def get_manifest_content_gui(stage_name: str, version_timestamp: str):
    if not stage_name or not version_timestamp:
        return "Missing stage or version."
    manifest_path = os.path.join("data", stage_name, version_timestamp, "manifest.json")
    if not os.path.exists(manifest_path):
        return f"Error: manifest.json not found in {os.path.dirname(manifest_path)}"
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        return json.dumps(manifest_data, indent=4)
    except Exception as e:
        return f"Error reading manifest.json: {e}"


def set_latest_version_gui(stage_name: str, version_timestamp: str, master_key: str):
    if master_key != MASTER_KEY:
        return "Invalid Master key."
    if not stage_name or not version_timestamp:
        return "Stage name and version timestamp must be provided."

    stage_data_dir = os.path.join("data", stage_name)
    version_path_to_check = os.path.join(stage_data_dir, version_timestamp)

    if not os.path.isdir(version_path_to_check):
        return f"Error: Version directory '{version_timestamp}' does not exist in stage '{stage_name}'."

    latest_txt_path = os.path.join(stage_data_dir, "latest.txt")
    try:
        with open(latest_txt_path, "w", encoding="utf-8") as f:
            f.write(version_timestamp)
        return f"Successfully set version '{version_timestamp}' as latest for stage '{stage_name}'."
    except Exception as e:
        return f"Error updating latest.txt: {e}"


def toggle_file_exclusion_gui(stage_name: str, version_timestamp: str, filename: str, master_key: str):
    if master_key != MASTER_KEY:
        return "Invalid Master key.", False # Message, new_exclusion_status
    if not stage_name or not version_timestamp or not filename:
        return "Stage, version, and filename must be provided.", False

    manifest_path = os.path.join("data", stage_name, version_timestamp, "manifest.json")
    if not os.path.exists(manifest_path):
        return f"Error: manifest.json not found in {os.path.dirname(manifest_path)}.", False

    try:
        with open(manifest_path, "r+", encoding="utf-8") as f:
            manifest_data = json.load(f)
            excluded_files = manifest_data.get("excluded_files", [])
            
            new_exclusion_status = False
            if filename in excluded_files:
                excluded_files.remove(filename)
                action_message = f"File '{filename}' UNEXCLUDED."
                new_exclusion_status = False
            else:
                excluded_files.append(filename)
                action_message = f"File '{filename}' EXCLUDED."
                new_exclusion_status = True
            
            manifest_data["excluded_files"] = sorted(list(set(excluded_files))) # Ensure unique and sorted
            
            f.seek(0) # Rewind to overwrite
            json.dump(manifest_data, f, indent=4)
            f.truncate() # Remove any trailing content if new manifest is shorter
            
        return f"Successfully updated exclusion for '{filename}'. {action_message}", new_exclusion_status
    except Exception as e:
        return f"Error toggling exclusion for '{filename}': {e}", False


def list_manual_uploads(target_stage, user_key):
    if user_key != MASTER_KEY:
        return "Invalid Master key"
    if not target_stage:
        return "Please select a stage to view uploads for."

    data_dir = f"data/{target_stage}"
    if not os.path.exists(data_dir):
        return f"No data directory found for stage '{target_stage}'. No uploads to list."

    manual_files = []
    try:
        for filename in os.listdir(data_dir):
            if filename.startswith("manual_upload_") and filename.endswith(".txt"):
                manual_files.append(filename)
    except Exception as e:
        return f"Error reading data directory for stage '{target_stage}': {e}"

    if not manual_files:
        return f"No manually uploaded files found in '{data_dir}' that match the 'manual_upload_*.txt' pattern."
    else:
        return f"Manually uploaded files in '{data_dir}':\n" + "\n".join(manual_files)

# Get stages from the model's definition for dropdowns
available_stages = list(model.define_growth_tasks().keys())

with gr.Blocks(title="PythonMasterAI: Serving Master Daddy") as iface:
    gr.Markdown("## PythonMasterAI Control Panel")
    with gr.Tab("Master Commands"):
        cmd_input = gr.Textbox(
            label="Command", placeholder="Enter MASTER: or HALT_TEEN"
        )
        key_input = gr.Textbox(label="Master Key", type="password")
        cmd_output = gr.Textbox(label="Response")
        cmd_button = gr.Button("Execute")
        cmd_button.click(
            interact_with_ai, inputs=[cmd_input, key_input], outputs=cmd_output
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
        growth_key_input = gr.Textbox(label="Master Key", type="password")
        growth_button = gr.Button("Approve Growth")
        growth_output = gr.Textbox(label="Growth Status")
        growth_button.click(
            approve_growth, inputs=growth_key_input, outputs=growth_output
        )
    with gr.Tab("Manual Data Upload"):
        gr.Markdown("Upload training documents (e.g., .txt, .py, .md, .pdf). Content will be extracted and saved as .txt files.")
        upload_files = gr.File(label="Upload Training Files", file_count="multiple")
        upload_stage_select = gr.Dropdown(choices=available_stages, label="Target Stage for Uploaded Data", value=model.stage)
        upload_key_input = gr.Textbox(label="Master Key", type="password")
        upload_button = gr.Button("Upload and Save Files")
        upload_output = gr.Textbox(label="Upload Status", lines=5)
        upload_button.click(
            handle_manual_upload,
            inputs=[upload_files, upload_stage_select, upload_key_input],
            outputs=upload_output
        )
        gr.Markdown("---") # Separator
        view_uploads_button = gr.Button("View Manually Uploaded Files for Selected Stage")
        view_uploads_output = gr.Textbox(label="List of Uploaded Files", lines=5, interactive=False)
        view_uploads_button.click(
            list_manual_uploads,
            inputs=[upload_stage_select, upload_key_input], # Re-use key input and stage select from above
            outputs=view_uploads_output
        )
    with gr.Tab("Run Training Script"):
        gr.Markdown("Run the `train.py` script. This will execute in a separate process.")
        train_stage_select = gr.Dropdown(choices=available_stages, label="Select Stage for Training", value=model.stage)
        train_key_input = gr.Textbox(label="Master Key", type="password")
        train_run_button = gr.Button("Start Training Script")
        train_run_output = gr.Textbox(label="Training Script Output", lines=10, interactive=False)
        train_run_button.click(
            run_train_script_gui,
            inputs=[train_stage_select, train_key_input],
            outputs=train_run_output
        )
    with gr.Tab("Run Scraper Script"):
        gr.Markdown("Run the `scrape_data.py` script. This will execute in a separate process.")
        scrape_stage_select = gr.Dropdown(choices=available_stages, label="Select Stage for Scraping", value=model.stage)
        scrape_sources_input = gr.Textbox(label="Sources (comma-separated)", placeholder="e.g., github_beginner,study_guides")
        scrape_urls_input = gr.Textbox(label="URLs (comma-separated, in same order as sources)", placeholder="e.g., http://url1.com,http://url2.com")
        scrape_key_input = gr.Textbox(label="Master Key", type="password")
        scrape_run_button = gr.Button("Start Scraper Script")
        scrape_run_output = gr.Textbox(label="Scraper Script Output", lines=10, interactive=False)
        scrape_run_button.click(
            run_scrape_data_script_gui,
            inputs=[scrape_stage_select, scrape_sources_input, scrape_urls_input, scrape_key_input],
            outputs=scrape_run_output
        )
    with gr.Tab("Model Management"):
        gr.Markdown("Manage model checkpoints. Note: The model automatically attempts to load the latest compatible checkpoint on startup.")
        load_checkpoint_key_input = gr.Textbox(label="Master Key", type="password", placeholder="Enter Master Key to enable loading")
        load_checkpoint_button = gr.Button("Load Latest Model from Checkpoint for Current Stage & Configuration")
        load_checkpoint_status_output = gr.Textbox(label="Load Status", interactive=False, lines=3)
        
        load_checkpoint_button.click(
            load_latest_model_from_checkpoint_gui,
            inputs=[load_checkpoint_key_input],
            outputs=[load_checkpoint_status_output]
        )
    with gr.Tab("Dataset Management"):
        gr.Markdown("## Dataset Version Explorer and Management")
        dm_master_key_input = gr.Textbox(label="Master Key (for actions like Set Latest / Exclude)", type="password")
        
        with gr.Row():
            dm_stage_select = gr.Dropdown(choices=available_stages, label="Select Stage", value=model.stage)
            dm_refresh_versions_btn = gr.Button("Refresh Versions List")
        
        dm_versions_df = gr.DataFrame(
            headers=["Version ID", "Is Latest?", "Manifest Type", "# Files", "Manifest Timestamp"], 
            datatype=["str", "bool", "str", "number", "str"],
            label="Available Dataset Versions for Stage"
        )
        
        # Hidden storage for selected version to trigger file list update
        dm_selected_version_id_state = gr.State(None)

        # Update DataFrame when stage changes or refresh button is clicked
        dm_stage_select.change(get_dataset_versions, inputs=[dm_stage_select], outputs=[dm_versions_df])
        dm_refresh_versions_btn.click(get_dataset_versions, inputs=[dm_stage_select], outputs=[dm_versions_df])

        with gr.Row():
            dm_version_select_for_actions = gr.Dropdown(label="Select Version (for actions below)", interactive=True)
            dm_set_latest_btn = gr.Button("Set Selected Version as Latest")
            dm_view_manifest_btn = gr.Button("View Full Manifest of Selected Version")
        
        dm_set_latest_status_text = gr.Textbox(label="Set Latest Status", interactive=False)
        dm_manifest_content_text = gr.Textbox(label="Manifest Content", lines=10, interactive=False)

        # When DataFrame is updated, update the version dropdown for actions
        def update_version_action_dropdown(versions_data):
            if versions_data is not None and not versions_data.empty: # Check if DataFrame is not None and not empty
                version_ids = versions_data["Version ID"].tolist()
                return gr.Dropdown.update(choices=version_ids, value=version_ids[0] if version_ids else None)
            return gr.Dropdown.update(choices=[], value=None)

        dm_versions_df.change(update_version_action_dropdown, inputs=[dm_versions_df], outputs=[dm_version_select_for_actions])

        dm_set_latest_btn.click(
            set_latest_version_gui, 
            inputs=[dm_stage_select, dm_version_select_for_actions, dm_master_key_input], 
            outputs=[dm_set_latest_status_text]
        ).then(
            get_dataset_versions, # Refresh versions list to show new "is_latest"
            inputs=[dm_stage_select], 
            outputs=[dm_versions_df]
        )
        
        dm_view_manifest_btn.click(get_manifest_content_gui, inputs=[dm_stage_select, dm_version_select_for_actions], outputs=[dm_manifest_content_text])

        gr.Markdown("---")
        gr.Markdown("### Files in Selected Version")
        
        dm_files_df = gr.DataFrame(
            headers=["Filename", "Size (bytes)", "Excluded?"], 
            datatype=["str", "number", "bool"],
            label="Files in Selected Dataset Version"
        )
        
        # When version selection dropdown changes, update files list
        dm_version_select_for_actions.change(
            lambda stage, version: (get_files_in_version(stage, version), version), # Pass version to state
            inputs=[dm_stage_select, dm_version_select_for_actions], 
            outputs=[dm_files_df, dm_selected_version_id_state]
        )

        with gr.Row():
            dm_file_select_for_actions = gr.Dropdown(label="Select File (for actions below)", interactive=True)
            dm_toggle_exclusion_btn = gr.Button("Toggle Exclusion of Selected File")
        
        dm_toggle_exclusion_status_text = gr.Textbox(label="Toggle Exclusion Status", interactive=False)
        dm_file_content_text = gr.Textbox(label="File Content Preview", lines=15, interactive=False)

        # Update file selection dropdown when files_df changes
        def update_file_action_dropdown(files_data):
            if files_data is not None and not files_data.empty:
                filenames = files_data["Filename"].tolist()
                # Filter out error messages if any
                valid_filenames = [fn for fn in filenames if not fn.startswith("Error:")]
                return gr.Dropdown.update(choices=valid_filenames, value=valid_filenames[0] if valid_filenames else None)
            return gr.Dropdown.update(choices=[], value=None)

        dm_files_df.change(update_file_action_dropdown, inputs=[dm_files_df], outputs=[dm_file_select_for_actions])
        
        # Action for viewing file content (triggered by selecting a file from dropdown)
        dm_file_select_for_actions.change(
            get_file_content_gui, 
            inputs=[dm_stage_select, dm_selected_version_id_state, dm_file_select_for_actions], # Use stored selected version
            outputs=[dm_file_content_text]
        )

        dm_toggle_exclusion_btn.click(
            toggle_file_exclusion_gui,
            inputs=[dm_stage_select, dm_selected_version_id_state, dm_file_select_for_actions, dm_master_key_input],
            outputs=[dm_toggle_exclusion_status_text] # Assuming toggle_file_exclusion_gui returns (message, new_status_bool)
                                                        # We might need to adjust how this output is handled or add another output for the boolean
        ).then(
            lambda stage, version: get_files_in_version(stage, version), # Refresh file list to show new exclusion status
            inputs=[dm_stage_select, dm_selected_version_id_state],
            outputs=[dm_files_df]
        )


iface.launch()
