# train.py
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset as TorchDataset # Added TorchDataset
import torch.nn as nn
from python_master_ai import PythonMasterAI
from datasets import load_dataset
from transformers import AutoTokenizer
from scrape_data import scrape_data
import pytest
from typing import cast # Added cast

model = PythonMasterAI()
tokenizer = AutoTokenizer.from_pretrained("gpt2")
optimizer = Adam(model.parameters(), lr=1e-4)


def run_unit_tests(code):
    with open("test_temp.py", "w") as f:
        f.write(code)
    result = pytest.main(["test_temp.py", "--quiet"])
    return result == 0


def train(stage="baby"):
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
        # After ALL scraping is done, process the data relevant to research
        model.process_scraped_research_data(stage)

    # Load the 'train' split directly to get a datasets.Dataset object.
    # This makes the type clearer for Pylance and avoids indexing a potentially
    # misinterpreted DatasetDict (which Pylance thought was an IterableDataset).
    # datasets.Dataset is map-style by default when streaming=False (default).
    train_dataset = load_dataset("text", data_dir=f"data/{stage}", split="train")

    for epoch in range(5):
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
            code = model.generate_code("write function")
            if run_unit_tests(code):
                model.log_task_progress("unit_test_accuracy", success=True)
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


if __name__ == "__main__":
    train()
