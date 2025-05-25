# train.py
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR # Added for learning rate scheduling
from torch.utils.data import DataLoader, Dataset as TorchDataset
import torch.nn as nn
from python_master_ai import PythonMasterAI
from datasets import load_dataset
from transformers import AutoTokenizer
from scrape_data import scrape_data
import pytest # type: ignore
from typing import DefaultDict, Sized, cast
import os
import torch
import json
import glob
import logging  # Added for logging
from utils import get_typed_config_value, setup_logging, get_config_value # Updated import

# --- Initialize logger for this module ---
# Note: setup_logging() will be called in if __name__ == "__main__" or by an importing module.
# If this script is run directly, logging might be unconfigured until main().
# If imported, the importing module should have configured logging.
logger = logging.getLogger(__name__)

# --- Initialize Model, Tokenizer, Optimizer with config values ---
# These are initialized at the module level because they might be used by main or helper functions.
# setup_logging() should ideally be called before these initializations if they log anything.
# For now, assuming PythonMasterAI's __init__ uses logger internally if needed.
model = PythonMasterAI()
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None: # Ensure pad_token is set for tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer pad_token set to eos_token as it was None.")


default_lr = get_typed_config_value("training_defaults.learning_rate", 1e-4, float)
optimizer = Adam(model.parameters(), lr=default_lr)

# Learning Rate Scheduler
lr_scheduler_step_size = get_typed_config_value("training_defaults.lr_scheduler.step_size", 1, int)
lr_scheduler_gamma = get_typed_config_value("training_defaults.lr_scheduler.gamma", 0.7, float)
scheduler = StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)
# logger.info(f"Initialized Adam optimizer with learning rate: {default_lr}") # Logger might not be set up yet if not main

CHECKPOINT_DIR = str(get_config_value("checkpointing.checkpoint_dir", "checkpoints"))
# logger.info(f"Using checkpoint directory: {CHECKPOINT_DIR}") # Logger might not be set up yet

# TensorBoard Writer (initialized in train() or main if used)
writer = None


def run_unit_tests(code):
    with open("test_temp.py", "w", encoding="utf-8") as f:
        f.write(code)
    result = pytest.main(["test_temp.py", "--quiet"])
    return result == 0


def train(stage: str):
    global writer  # Access the global writer
    if writer is None:  # Initialize TensorBoard writer if not already done
        tb_log_dir_val = get_config_value(
            "logging.tensorboard_log_dir", "runs/python_master_ai_experiment"
        )
        # Ensure tb_log_dir_val is a string before .lower() and use
        is_valid_tb_path_string = isinstance(tb_log_dir_val, str) and \
                                  tb_log_dir_val.lower() not in ["none", "null"]

        if is_valid_tb_path_string:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=tb_log_dir_val) # Use tb_log_dir_val here
            logger.info(f"TensorBoard logging to: {tb_log_dir_val}") # And here
        else:
            if not isinstance(tb_log_dir_val, str):
                logger.warning(
                    f"TensorBoard log_dir from config is not a valid string: '{tb_log_dir_val}'. Disabling TensorBoard."
                )
            logger.info("TensorBoard logging is disabled via config.")

    logger.info(f"Train function initiated for stage: {stage}")

    if model.stage != stage:
        logger.info(
            f"Updating model's current stage from '{model.stage}' to '{stage}' for this training run."
        )
        model.stage = stage
        model.task_progress = DefaultDict(float)
        model.knowledge_gaps = []

    scrape_targets_for_research = []
    if model.assess_performance()["needs_research"]:
        scrape_targets_for_research = model.get_research_scrape_targets()
        model.discover_new_sources()

    priority_source_names = model.prioritize_scraping()
    priority_scrape_targets = []
    for source_name in priority_source_names:
        if source_name in model.known_sources and model.known_sources[source_name]:
            priority_scrape_targets.append(
                (source_name, model.known_sources[source_name][0])
            )
        else:
            logger.warning(
                f"Priority source '{source_name}' not in known_sources or has no URL."
            )

    all_scrape_targets_dict = {}
    for source, url in priority_scrape_targets:
        all_scrape_targets_dict[source] = url
    for source, url in scrape_targets_for_research:
        all_scrape_targets_dict[source] = url

    final_sources_to_scrape = list(all_scrape_targets_dict.keys())
    final_urls_to_scrape = list(all_scrape_targets_dict.values())

    if final_sources_to_scrape:
        logger.info(
            f"Consolidated scraping for stage '{stage}': {final_sources_to_scrape} with URLs: {final_urls_to_scrape}"
        )
        scrape_data(stage, final_sources_to_scrape, final_urls_to_scrape)

    logger.info(
        f"Attempting to process research data for stage '{stage}' using existing/newly scraped files."
    )
    model.process_scraped_research_data(stage)

    current_dataset_path = model.get_latest_dataset_path(stage)
    if not current_dataset_path:
        logger.error(
            f"Could not determine latest dataset path for stage '{stage}'. Training cannot proceed."
        )
        return

    logger.info(
        f"Training using dataset version: {current_dataset_path} for stage '{stage}'"
    )
    model.current_dataset_version = os.path.basename(current_dataset_path)

    manifest_path = os.path.join(current_dataset_path, "manifest.json")
    excluded_files_list = []
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                manifest_data = json.load(f)
            excluded_files_list = manifest_data.get("excluded_files", [])
            if excluded_files_list:
                logger.info(
                    f"Excluding files from training based on manifest: {excluded_files_list}"
                )
        except Exception as e:
            logger.warning(
                f"Could not read or parse manifest.json at {manifest_path}. Proceeding without exclusions. Error: {e}"
            )

    all_txt_files_in_dir = glob.glob(os.path.join(current_dataset_path, "*.txt"))
    all_txt_files_in_dir = [
        f for f in all_txt_files_in_dir if os.path.basename(f) != "manifest.json"
    ]
    data_files_for_load_dataset = []
    for f_path in all_txt_files_in_dir:
        if os.path.basename(f_path) not in excluded_files_list:
            data_files_for_load_dataset.append(f_path)
        else:
            logger.info(f"  Skipping excluded file: {os.path.basename(f_path)}")

    if not data_files_for_load_dataset:
        logger.error(
            f"No data files remaining for training in {current_dataset_path} after applying exclusions. Aborting training."
        )
        return

    try:
        train_dataset = load_dataset(
            "text", data_files=data_files_for_load_dataset, split="train"
        )
        if not train_dataset:
            logger.warning(
                f"The loaded dataset from files {data_files_for_load_dataset} is empty."
            )
        elif hasattr(train_dataset, "__len__"):
            # train_dataset has __len__ attribute, so it conforms to Sized.
            # Cast for Pylance to confirm this understanding for the len() call.
            if len(cast(Sized, train_dataset)) == 0:
                logger.warning(
                    f"The loaded dataset (map-style) from files {data_files_for_load_dataset} is empty."
                )
    except Exception as e:
        logger.error(
            f"Error loading dataset from specific files {data_files_for_load_dataset}: {e}",
            exc_info=True,
        )
        return

    current_loss_value = None
    total_steps = 0  # For TensorBoard global_step

    num_epochs = get_typed_config_value("training_defaults.num_epochs", 5, int)
    batch_size = get_typed_config_value("training_defaults.batch_size", 4, int)
    gradient_clipping_norm = get_typed_config_value("training_defaults.gradient_clipping_norm", 1.0, float)

    logger.info(
        f"Starting training for {num_epochs} epochs with batch size {batch_size}."
    )

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        for batch in DataLoader(
            cast(TorchDataset, train_dataset), batch_size=batch_size # type: ignore
        ):
            tokenized_inputs = tokenizer(
                batch["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=model.hidden_size,
            ) # type: ignore
            inputs = {k: v.to(model.device) for k, v in tokenized_inputs.items()}
            outputs = model(
                inputs["input_ids"], src_key_padding_mask=inputs.get("attention_mask")
            )
            loss = nn.CrossEntropyLoss()(
                outputs.view(-1, outputs.size(-1)), inputs["input_ids"].view(-1)
            )
            loss.backward()
            
            if gradient_clipping_norm > 0: # Apply gradient clipping if configured
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_norm)

            optimizer.step()

            loss_item = loss.item()
            model.log_performance(
                "loss", loss_item
            )  # This writes to performance_log.json
            current_loss_value = loss_item
            epoch_loss += loss_item
            num_batches += 1
            total_steps += 1

            if writer:  # Log batch loss to TensorBoard
                writer.add_scalar(
                    f"Training/Batch_Loss_Stage_{stage}", loss_item, total_steps
                )

            code = model.generate_code("write function")
            if run_unit_tests(code):
                model.log_task_progress("unit_test_accuracy", success=True)

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float("nan")
        logger.info(
            f"Epoch {epoch+1}/{num_epochs}, Average Epoch Loss: {avg_epoch_loss}, Last Batch Loss: {current_loss_value}"
        )
        if writer:  # Log epoch loss to TensorBoard
            writer.add_scalar(
                f"Training/Epoch_Avg_Loss_Stage_{stage}", avg_epoch_loss, epoch + 1
            )
            writer.add_scalar(
                f"Training/Learning_Rate_Stage_{stage}", scheduler.get_last_lr()[0], epoch + 1
            )
        scheduler.step() # Step the learning rate scheduler

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        ai_state_for_checkpoint = model.get_state_for_checkpoint()
        checkpoint_data = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": current_loss_value,
            "ai_state": ai_state_for_checkpoint,
        }
        epoch_checkpoint_filename = os.path.join(
            CHECKPOINT_DIR,
            f"model_stage_{model.stage}_config_{model.configuration_id}_epoch_{epoch+1}.pt",
        )
        latest_checkpoint_filename = os.path.join(
            CHECKPOINT_DIR,
            f"model_stage_{model.stage}_config_{model.configuration_id}_latest.pt",
        )
        torch.save(checkpoint_data, epoch_checkpoint_filename)
        torch.save(checkpoint_data, latest_checkpoint_filename)
        logger.info(
            f"Saved checkpoint for epoch {epoch+1} to {epoch_checkpoint_filename} and {latest_checkpoint_filename}"
        )

    if writer:
        writer.close()
        logger.info("TensorBoard writer closed.")


if __name__ == "__main__":
    # Setup logging as early as possible
    setup_logging()

    # Now that logging is set up, these initializations can log if needed
    logger.info(f"Initialized Adam optimizer with learning rate: {default_lr}")
    logger.info(f"Using checkpoint directory: {CHECKPOINT_DIR}")

    import argparse

    parser = argparse.ArgumentParser(description="Train the PythonMasterAI model.")
    parser.add_argument(
        "--stage",
        type=str,
        default=model.stage,
        choices=list(model.define_growth_tasks().keys()),
        help="The training stage.",
    )
    args = parser.parse_args()

    logger.info(f"Starting training script for stage: {args.stage}")
    train(stage=args.stage)
