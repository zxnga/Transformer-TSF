# Transformer Time-Series Forecasting

This project implements time series forecasting using a Vanilla Transformer model, using HF's TimeSeriesTransformer.

## Table of Contents

- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Files Description](#files-description)

## Dataset Structure

The model expects the dataset to be structured as follows:

- `target`: The values to be predicted.
- `start`: The start time of the series.
- `feat_static_cat`: Static categorical features associated with the time series.
- `feat_dynamic_real`: Dynamic real-valued features that change over time.
- `item_id`: Identifier for each time series item.

Ensure your dataset aligns with this structure for optimal performance.

## Installation

To set up the project environment, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/zxnga/Transformer-TSF.git
   cd Transformer-TSF
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows, use 'env\Scripts\activate'
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To train and evaluate the TimeSeriesTransformer model:

1. **Prepare your dataset:** Ensure it follows the [Dataset Structure](#dataset-structure) mentioned above.

2. **Define a configuration for the Transformer:** Update any hyperparameters as needed, https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/time_series_transformer#transformers.TimeSeriesTransformerConfig.
3. Follow the example in example.ipynb

## Files Description

- `ts_transformer.py`: Contains the implementation of the functions to train the TimeSeriesTransformer model.
- `plotting.py`: Includes functions for visualizing the results.
- `utils.py`: Utility functions for data preprocessing, loading, and other helper methods.