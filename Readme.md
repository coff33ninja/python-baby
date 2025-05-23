# PythonMasterAI Project

## Project Overview
PythonMasterAI is a self-improving AI designed to master Python programming through various stages of growth, from "baby" to "adult". It learns by scraping data, conducting research, and eventually training itself. The project features a Gradio-based GUI for interaction and control.

## Setup Instructions

1.  **Create and activate a Python virtual environment** (e.g., using `venv` or `conda`). It's recommended to use Python 3.9 or newer.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
2.  **Install dependencies from `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Install Playwright browsers** (needed by `scrapy-playwright` for web scraping):
    ```bash
    playwright install
    ```

## Running the System

### Dataset Initialization:

Run `scrape_data.py` to initialize the dataset for a specific stage (e.g., "baby"). This will create a versioned dataset.
```bash
python scrape_data.py baby # Scrapes default sources for baby stage
# Or specify sources:
# python scrape_data.py baby github_beginner https://api.github.com/search/repositories?q=language:python+stars:>100
```

### Master Key API:

The Master Key API (`master_key.py`) is used for approving critical actions like model growth.
For security, it's recommended to run this in a controlled environment.

1.  Ensure `master_key.py` is present (it's part of the repository).
2.  Run using Docker (example):
    ```bash
    docker run -d -p 8000:8000 -v $(pwd)/master_key.py:/app/master_key.py:ro python:3.11-slim bash -c "pip install fastapi uvicorn && uvicorn master_key:app --host 0.0.0.0 --port 8000"
    ```
    *(Note: Adjust `python:3.11-slim` if your `requirements.txt` implies a different Python version for FastAPI/Uvicorn compatibility, though they are generally flexible.)*

### Training:

Run `train.py` for a specific stage. The script will use the latest versioned dataset for that stage.
```bash
python train.py --stage baby
```
Training duration varies based on the stage and hardware (e.g., ~6 hours on CPU for the initial baby stage with limited data). The model will save checkpoints in the `checkpoints/` directory.

### Interaction via GUI:

Run `gui.py` to launch the Gradio interface for interacting with and managing the AI.
```bash
python gui.py
```
The GUI provides tabs for:
*   **Master Commands:** Send commands to the AI (e.g., "MASTER: Write a function").
*   **Research:** Trigger the AI's research process.
*   **Source Discovery:** Initiate discovery of new data sources.
*   **Status:** View the AI's current status (stage, parameters, tasks, etc.).
*   **Growth:** Approve model growth to the next stage (requires Master Key).
*   **Manual Data Upload:** Upload additional training data to a specific stage's dataset.
*   **Run Training/Scraper Scripts:** Trigger `train.py` or `scrape_data.py` from the GUI.
*   **Model Management:** Manually trigger loading of the latest checkpoint.
*   **Dataset Management:** Explore dataset versions, view manifest files, preview file contents, set the "latest" version for a stage, and exclude/include files from training.

### Model Growth:

Model growth can be initiated via the "Growth" tab in the Gradio GUI (requires Master Key) or by running `grow.py` directly after the AI has met its growth criteria for the current stage.
```bash
python grow.py
```
This script creates a new, more complex model instance, attempts to transfer weights from the previous model, and saves an initial checkpoint for the new configuration.

### Evaluation:

The `evaluate.py` script runs a formal evaluation suite to test the AI's capabilities on various tasks.
```bash
python evaluate.py --model_checkpoint_path <path_to_checkpoint.pt> --eval_dataset_path sample_evaluation_dataset.jsonl --output_dir eval_results
```
See `sample_evaluation_dataset.jsonl` for task examples. Evaluation results, including detailed logs and a summary, will be saved in the specified output directory (default: `eval_results`).


## Current Capabilities (Baby Stage - Example)

*   **Python Code Generation:** Generates simple Python functions (e.g., `def add(a, b): return a + b`).
*   **Explanations:** Provides basic explanations for Python concepts or code snippets.
*   **Self-Scraping & Data Versioning:** Collects data from sources like GitHub, study guides, Stack Overflow, and Reddit. Datasets are versioned with timestamps and manifests.
*   **Self-Research:** Identifies knowledge gaps, formulates queries, and processes scraped data to fill these gaps.
*   **Self-Source Discovery:** Simulates finding new data sources.
*   **Self-Growth & Checkpointing:** Tracks task completion for growth. Model checkpoints are saved during training and after growth. Grown models use a weight seeding strategy.
*   **Master Commands & Stop Code:** Responds to master commands and can be paused.
*   **Gradio GUI:** Provides comprehensive control and monitoring for all major functions.
*   **Evaluation Suite:** Includes `evaluate.py` for formal testing of model capabilities.

## Next Steps

*   **Advanced Generation:** Improve the `generate` method in `PythonMasterAI` for more complex and coherent outputs.
*   **Custom Tokenizer:** Train a ByteLevelBPETokenizer on a larger Python-specific corpus.
*   **Real Source Search Integration:** Replace simulated source discovery with actual Google API or GitHub API calls.
*   **Unit Test Expansion:** Develop more comprehensive unit tests for all AI tasks and capabilities.
*   **RAG Implementation:** Integrate a Retrieval Augmented Generation pipeline (e.g., using LangChain) for more informed responses.
*   **Refine Evaluation:** Enhance `evaluate.py` with more sophisticated metrics and test cases.
*   **Scaling:** Progress through toddler, teenager, and adult stages, implementing more complex tasks and architectural changes.

## Example Gradio Interaction

(This section can be updated as the AI's generation capabilities improve)

*   **Master Commands:**
    *   Input: `MASTER: Write a function to sum a list of numbers`, Key: `8f9b7f8f...`
    *   Output: (AI's generated code for the sum function)
*   **Dataset Management:**
    *   Select "baby" stage, view versions, select a version, view its manifest or files.

## Challenges and Solutions

*   **Generation Quality:** Initial generation is basic. Requires significant training and larger model sizes.
*   **Compute Resources:** CPU is feasible for early stages, but GPU will be necessary for larger models and extensive training.
*   **Security:** Master key provides basic protection. `RestrictedPython` in `evaluate.py` offers some safety for code execution, but true sandboxing for arbitrary code is complex.
