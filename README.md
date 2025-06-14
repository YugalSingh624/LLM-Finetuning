# Mistral-7B Fine-Tuning Project

## Overview
This project demonstrates the fine-tuning of the Mistral-7B language model using the Unsloth library for improved efficiency. The model is fine-tuned on the Alpaca dataset and evaluated on the ARC-Easy benchmark to measure performance improvements.

## Files
- `mistral-7b.ipynb`: Jupyter notebook containing the complete fine-tuning process
- `baseline.json`: Evaluation results of the base Mistral-7B model
- `finetuned.json`: Evaluation results of the fine-tuned Mistral-7B model
- `summary.pdf`: Summary report of the project findings
- `transformer_reflection.pdf`: Technical analysis of transformer-based models

## Model Details
- Base Model: Mistral-7B (4-bit quantized)
- Fine-tuning Method: LoRA (Low-Rank Adaptation)
- Training Dataset: Alpaca (subset of 1500 examples)
- LoRA Configuration:
  - Rank (r): 16
  - Alpha: 16
  - Target Modules: Query, Key, Value projections and MLP layers
  - Training Epochs: 2

## Results
Evaluation on ARC-Easy benchmark:

| Model | Accuracy | Normalized Accuracy |
|-------|----------|---------------------|
| Base Mistral-7B | 79.80% | 78.49% |
| Fine-tuned Mistral-7B | 81.19% | 80.47% |

The fine-tuned model shows a +1.39% improvement in accuracy and +1.98% in normalized accuracy.

## Environment
- Hardware: Tesla P100-PCIE-16GB GPU
- Framework: PyTorch 2.6.0
- Libraries: Unsloth, Transformers 4.52.4, bitsandbytes

## Usage
To reproduce the fine-tuning:
1. Open `mistral-7b.ipynb` in a GPU-enabled environment
2. Install the required dependencies
3. Run the notebook cells sequentially

## Evaluation
The model was evaluated using the lm-eval framework on the ARC-Easy benchmark, which consists of grade-school level, easy multiple-choice science questions.

## Acknowledgments
- Mistral AI for the base Mistral-7B model
- Unsloth for the efficient fine-tuning library
- Tatsu Lab for the Alpaca dataset
- Allen AI for the ARC dataset

**Tags:** LLM-Finetuning 