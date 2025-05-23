# train.py
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset as TorchDataset # Added TorchDataset
import torch.nn as nn
from python_master_ai import PythonMasterAI
from datasets import load_dataset
from transformers import AutoTokenizer
from scrape_data import scrape_data
import pytest
from typing import cast, DefaultDict # Added cast and DefaultDict
import os # Added for checkpointing
import torch # Added for checkpointing
import json 
import glob # Added for file exclusion logic

model = PythonMasterAI()
tokenizer = AutoTokenizer.from_pretrained("gpt2")
optimizer = Adam(model.parameters(), lr=1e-4)

# --- Checkpoint Directory ---
CHECKPOINT_DIR = "checkpoints"


def run_unit_tests(code):
    with open("test_temp.py", "w") as f:
        f.write(code)
    result = pytest.main(["test_temp.py", "--quiet"])
    return result == 0


def train(stage: str):
    print(f"Train function initiated for stage: {stage}")

    # Ensure the model instance used by this train function is aligned with the target stage
    if model.stage != stage:
        print(f"Updating model's current stage from '{model.stage}' to '{stage}' for this training run.")
        model.stage = stage
        # Reset progress for the new stage if it's being set explicitly for a training run
        model.task_progress = DefaultDict(int)
        model.knowledge_gaps = []

    scrape_targets_for_research = []
    if model.assess_performance()["needs_research"]:
        # Get research targets but don't scrape yet
        scrape_targets_for_research = model.get_research_scrape_targets()
        model.discover_new_sources()

    # Get general sources to scrape based on priority
    priority_source_names = model.prioritize_scraping()
    priority_scrape_targets = []
    for source_name in priority_source_names:
        if source_name in model.known_sources and model.known_sources[source_name]:
            priority_scrape_targets.append((source_name, model.known_sources[source_name][0])) # Assuming first URL
        else:
            print(f"Warning: Priority source '{source_name}' not in known_sources or has no URL.")

    # Combine and deduplicate all scrape targets
    # Use a dictionary to ensure each source is targeted only once, prioritizing research targets
    all_scrape_targets_dict = {}
    for source, url in priority_scrape_targets: # Add priority targets first
        all_scrape_targets_dict[source] = url
    for source, url in scrape_targets_for_research: # Override with/add research targets
        all_scrape_targets_dict[source] = url

    final_sources_to_scrape = list(all_scrape_targets_dict.keys())
    final_urls_to_scrape = list(all_scrape_targets_dict.values())

    if final_sources_to_scrape:
        print(f"Consolidated scraping for stage '{stage}': {final_sources_to_scrape} with URLs: {final_urls_to_scrape}")
        scrape_data(stage, final_sources_to_scrape, final_urls_to_scrape)
        # Previously, process_scraped_research_data was only called if new scraping occurred.

    # Always attempt to process research data.
    # The method `process_scraped_research_data` will check internally if there are
    # active research queries (knowledge gaps) to address.
    # This ensures that even if no new scraping happened in this run,
    # existing data can be used to try and resolve pending knowledge gaps.
    print(f"Attempting to process research data for stage '{stage}' using existing/newly scraped files.")
    model.process_scraped_research_data(stage)

    # Load the 'train' split directly to get a datasets.Dataset object.
    # This makes the type clearer for Pylance and avoids indexing a potentially
    # misinterpreted DatasetDict (which Pylance thought was an IterableDataset).
    # datasets.Dataset is map-style by default when streaming=False (default).
    
    # --- Determine Data Path using versioning ---
    current_dataset_path = model.get_latest_dataset_path(stage)
    if not current_dataset_path:
        print(f"Error: Could not determine latest dataset path for stage '{stage}'. Training cannot proceed.")
        return # Exit the training function
    
    print(f"Training using dataset version: {current_dataset_path} for stage '{stage}'")
    # Store the dataset version (timestamp) in the model's state for checkpointing
    model.current_dataset_version = os.path.basename(current_dataset_path)

    # --- Handle file exclusions based on manifest.json ---
    manifest_path = os.path.join(current_dataset_path, "manifest.json")
    excluded_files_list = []
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
            excluded_files_list = manifest_data.get("excluded_files", [])
            if excluded_files_list:
                print(f"Excluding files from training based on manifest: {excluded_files_list}")
        except Exception as e:
            print(f"Warning: Could not read or parse manifest.json at {manifest_path}. Proceeding without exclusions. Error: {e}")
    
    all_txt_files_in_dir = glob.glob(os.path.join(current_dataset_path, "*.txt"))
    
    # Filter out manifest.json itself from the list of all text files, just in case
    all_txt_files_in_dir = [f for f in all_txt_files_in_dir if os.path.basename(f) != "manifest.json"]

    data_files_for_load_dataset = []
    for f_path in all_txt_files_in_dir:
        if os.path.basename(f_path) not in excluded_files_list:
            data_files_for_load_dataset.append(f_path)
        else:
            print(f"  Skipping excluded file: {os.path.basename(f_path)}")

    if not data_files_for_load_dataset:
        print(f"Error: No data files remaining for training in {current_dataset_path} after applying exclusions. Aborting training.")
        return

    try:
        # Using data_files argument
        train_dataset = load_dataset("text", data_files=data_files_for_load_dataset, split="train")
        if not train_dataset or len(train_dataset) == 0: # Should be redundant due to the check above, but good for safety
            print(f"Warning: The loaded dataset from files {data_files_for_load_dataset} is empty. Training will proceed but may not be effective.")
    except Exception as e:
        print(f"Error loading dataset from specific files {data_files_for_load_dataset}: {e}")
        print("Please ensure the dataset files are correctly formatted and accessible.")
        return # Exit training

    # Initialize loss variable outside the loop to ensure it's available for checkpointing 
    # if the dataset is empty or loop doesn't run.
    current_loss_value = None

    for epoch in range(5): # Assuming 5 epochs as in the original code
        # datasets.Dataset is compatible with torch.utils.data.DataLoader.
        # We use typing.cast to inform Pylance of this compatibility if it struggles
        # to recognize it directly, addressing the reportArgumentType issue.
        for batch in DataLoader(
            cast(TorchDataset, train_dataset), batch_size=4
        ):  # Use the extracted train_dataset
            tokenized_inputs = tokenizer(
                batch["text"], return_tensors="pt", padding=True, truncation=True
            )
            # Move inputs to the model's device
            inputs = {k: v.to(model.device) for k, v in tokenized_inputs.items()}
            outputs = model(inputs["input_ids"], src_key_padding_mask=inputs.get("attention_mask")) # Pass attention mask if tokenizer provides it and model uses it
            loss = nn.CrossEntropyLoss()(
                outputs.view(-1, outputs.size(-1)), inputs["input_ids"].view(-1)
            )
            loss.backward()
            optimizer.step()
            model.log_performance("loss", loss.item())
            current_loss_value = loss.item() # Keep track of the latest loss
            code = model.generate_code("write function")
            if run_unit_tests(code):
                model.log_task_progress("unit_test_accuracy", success=True)
        print(f"Epoch {epoch+1}, Loss: {current_loss_value}")

        # --- Checkpoint Saving Logic ---
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        ai_state_for_checkpoint = model.get_state_for_checkpoint()

        checkpoint_data = {
            'epoch': epoch + 1, # epoch is 0-indexed, so +1 for human-readable 1-indexed epoch
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': current_loss_value, # Use the loss from the end of the epoch
            'ai_state': ai_state_for_checkpoint
        }

        # model.stage and model.configuration_id are accessed from the model instance
        # The epoch number in the filename should be 1-indexed to match the 'epoch' key in checkpoint_data
        epoch_checkpoint_filename = os.path.join(CHECKPOINT_DIR, f"model_stage_{model.stage}_config_{model.configuration_id}_epoch_{epoch+1}.pt")
        latest_checkpoint_filename = os.path.join(CHECKPOINT_DIR, f"model_stage_{model.stage}_config_{model.configuration_id}_latest.pt")

        torch.save(checkpoint_data, epoch_checkpoint_filename)
        torch.save(checkpoint_data, latest_checkpoint_filename)
        print(f"Saved checkpoint for epoch {epoch+1} to {epoch_checkpoint_filename} and {latest_checkpoint_filename}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the PythonMasterAI model.")
    parser.add_argument("--stage", type=str, default=model.stage, # Default to current model's initial stage
                        choices=list(model.define_growth_tasks().keys()),
                        help="The training stage.")
    args = parser.parse_args()
    
    print(f"Starting training script for stage: {args.stage}")
    train(stage=args.stage)
