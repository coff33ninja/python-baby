# gui.py
import gradio as gr
from python_master_ai import PythonMasterAI
import json
from grow import grow_model
import os # Added for path operations
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

    data_dir = f"data/{target_stage}"
    os.makedirs(data_dir, exist_ok=True)

    results = []
    for uploaded_file in files:
        # Assuming 'uploaded_file' is an object where:
        # .name is the original filename
        # str(uploaded_file) is the temporary file path
        actual_original_filename = uploaded_file.name 
        temp_file_path = str(uploaded_file)

        original_filename_basename = os.path.basename(actual_original_filename)
        content_to_save = None  # Initialize to None, indicating content not yet processed

        # Generate a descriptive .txt filename based on the original
        original_filename_stem, original_extension = os.path.splitext(original_filename_basename)
        safe_stem = "".join(c if c.isalnum() or c == '_' else '_' for c in original_filename_stem)
        safe_ext = "".join(c if c.isalnum() else '' for c in original_extension.replace('.', ''))

        if safe_ext:
            save_filename_base = f"manual_upload_{safe_stem}_{safe_ext}"
        else:
            save_filename_base = f"manual_upload_{safe_stem}"
        save_filename = f"{save_filename_base}.txt"
        save_path = os.path.join(data_dir, save_filename)

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
                results.append(f"Processed '{actual_original_filename}' and saved as '{save_filename}'.")
            except Exception as save_e:
                results.append(f"Error saving content from '{actual_original_filename}' to '{save_filename}': {save_e}")
        # If content_to_save is None, it means the file was skipped (e.g., PDF without PyPDF2), and a message was already added.
    final_message = "\n".join(results)
    final_message += f"\n\nNote: Uploaded files in 'data/{target_stage}/' will be automatically included in the next training run for stage '{target_stage}'."
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


iface.launch()
