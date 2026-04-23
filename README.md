# README

## Project Overview
This repository contains the code for my TinyReproductions project, A Reproduction of Transformers without Normalization: Evaluating Dynamic Tanh as a LayerNorm Replacement. The project compares a small GPT-2 style baseline with standard LayerNorm against a modified version in which LayerNorm is replaced by Dynamic Tanh (DyT).

## Main File
- `main.py`: main experiment script

## What the Code Does
The script:
- loads the GPT-2 tokenizer
- automatically downloads the WikiText-2 raw dataset from Hugging Face
- uses a reduced subset of the dataset:
  - 2000 training samples
  - 400 validation samples
- builds a small GPT-2 style language model with:
  - 2 Transformer layers
  - hidden size 256
  - 4 attention heads
  - context length 128
- defines a DyT module with learnable `alpha`, `gamma`, and `beta`
- recursively replaces all LayerNorm modules with DyT for the DyT condition
- trains and evaluates both LayerNorm and DyT models
- repeats the experiment across 3 random seeds: 42, 123, and 456
- records training loss, validation loss, and DyT alpha values
- saves summary tables and plots to the `results/` folder

## Requirements
Install the following packages before running the script:
- `transformers`
- `datasets`
- `accelerate`
- `matplotlib`
- `pandas`
- `torch`
- `numpy`

Example installation command:

## How to Run

### Option 1: Run in Google Colab
1. Upload `main.py` to Google Colab, or paste the code into a Colab notebook.
2. Install the required packages if they are not already available.
3. Run the script from top to bottom.
4. The tokenizer and dataset will be downloaded automatically.
5. Output files will be saved in the `results/` directory.

### Option 2: Run on a Local Machine
1. Make sure Python and the required packages are installed.
2. pip install transformers datasets accelerate matplotlib pandas torch numpy
    
    ```
    pip install transformers datasets accelerate matplotlib pandas torch numpy
    ```
3. Remove any Colab-specific command such as:

   ```python
   !pip -q install transformers datasets accelerate matplotlib pandas
   ```
    if it is still present in the script.
4. Run the script from the command line:
    ```
    python main.py
    ```
5. The tokenizer and dataset will be downloaded automatically.
6. Output files will be saved in the results/ directory.

## Expected Output Files
The script saves the following files in ./results/:
- all_runs.json
- summary.csv
- epoch_logs.csv
- alpha_history.csv (for DyT runs)
- aggregate_summary.csv
- eval_loss_curve.png
- train_loss_curve_seed42.png
- dyt_alpha_curve_seed42.png

## Reproducibility Notes
- Random seeds are fixed for Python, NumPy, PyTorch, and Hugging Face.
- The experiment uses the same training setup for both LayerNorm and DyT models.
- No manual dataset download is required.

## Code Authorship and Adaptation

The DyT layer implementation in `main.py` (lines 63–71) was adapted from prior code associated with the original method. The remaining parts of the reproduction pipeline were written by me, including:
- lines 26–31: seed setup
- lines 40–61: dataset loading and tokenization
- lines 73–100: GPT-2 construction and LayerNorm-to-DyT replacement
- lines 102–160: logging helpers and custom Trainer
- lines 165–369: training, evaluation, multi-seed experiments, result export, and plotting