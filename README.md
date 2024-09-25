# Donut Fine-tuning on the Cord Dataset

This project focuses on fine-tuning the **Donut** model on the **Cord dataset**, aiming to enhance performance for document understanding tasks. The codebase provides scripts for training and testing the model.

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Configuration](#configuration)

## Project Overview

The **Donut** model is a transformer-based architecture designed for document understanding. In this project, we fine-tune the Donut model on the **Cord dataset**, a dataset for OCR and key information extraction from receipts.

The fine-tuning process involves training the model to better extract structured information from semi-structured document images, such as receipts, using a pre-trained Donut model as the base.

## Folder Structure

```
DONUT-CORD/
│
├── configs/                                # Configuration files for training
│   ├── train_config.json                   # Main training configuration file
│
├── model_logs/                             # Directory where model checkpoints and logs are saved
│
├── notebooks/                              # Jupyter notebooks for data exploration and predictions
│   ├── 01-look-at-data.ipynb               # Data exploration and visualization
│   ├── 02-look-at-prepared-data-and-model-output.ipynb  # Prepared data and model output analysis
│   ├── 03-look-at-predictions.ipynb        # Visualization of model predictions
│
├── scripts/                                # Shell scripts for training and testing
│   ├── test_donut.sh                       # Script for running model evaluation
│   ├── train_donut.sh                      # Script for starting model training
│
├── src/                                    # Source code for model training, testing, and data processing
│   ├── data_prep/                          # Data preparation and processing modules
│   │   ├── data_preparation.py             # Functions for preparing the dataset
│   │   ├── data_processing.py              # Data processing utilities
│   ├── arg_parse.py                        # Argument parsing for training/evaluation
│   ├── evaluation.py                       # Model evaluation utilities
│   ├── metrics.py                          # Custom metrics for performance evaluation
│   ├── ml_pipeline.py                      # Main pipeline for running model training
│   ├── prediction.py                       # Code for generating model predictions
│   ├── training.py                         # Model training loop and optimizer setup
│   ├── utils.py                            # Helper functions for miscellaneous tasks
│
├── requirements.txt                        # Python dependencies
├── test_donut.py                           # Python script for model evaluation
└── train_donut.py                          # Python script for model training
```

## Setup and Installation

### Requirements
- Python 3.8+
- PyTorch (with CUDA support)
- Hugging Face Transformers
- Accelerate for distributed training

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shahinntu/Donut-Cord.git
   cd Donut-Cord
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure that you have access to a GPU (NVIDIA recommended) and that `accelerate` is properly set up for distributed training.

## Usage

### Training

To train the Donut model on the Cord dataset, use the following command:
```bash
bash scripts/train_donut.sh
```

The model checkpoints and logs will be saved in the `model_logs/` directory.

### Evaluation

Before running the evaluation, ensure that you have the correct model checkpoint available.

To evaluate the trained model:
1. Update the `test_donut.sh` script to specify the correct restore version of the model.
2. Run the following command for evaluation:

   ```bash
   bash scripts/test_donut.sh
   ```

## Configuration

The training configuration can be modified by editing the `configs/train_config.json` file. Key parameters include:
- Batch size
- Learning rate
- Number of epochs
- Optimizer settings