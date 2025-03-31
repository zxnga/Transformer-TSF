# Transformer Time-Series Forecasting

This project implements time series forecasting using a Vanilla Transformer model, using HF's TimeSeriesTransformer.

What we provide:
1. Inference helper classes to help managing the model in a production setting.
2. Monitoring of the forecasting loss of the model to detect shifts in the underlying data distribution.
3. Attention Classifier to map the distribution shift to a known different data profile using the latent space of the Transformer.

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
   - list with dimension equal to the number of features
- `feat_static_real`:  Dynamic real-valued features associated with the time series.
   - list with dimension equal to the number of features
- `feat_dynamic_real`: Dynamic real-valued features that change over time. (curretnly not supported)
   - array with shape equal to (number of features, target length)
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
3. Follow the example in example.ipynb for training 
4. Use inference.py functions for inference helper class in production settings

Note: TimeSeriesTransformerForPrediction doesn't allow feat_dynamic_real

## Files Description
```
Transformer-TSF
├── examples
│   ├── loss_monitoring.ipynb
│   ├── train_test.ipynb
├── src (core codes)
│   ├── inference (to simplify running the mode in production)
│   │   ├── data
│   │   |   ├── buffer.py (buffers to store context, true values and predictions)
│   │   |   ├── helper.py (helper to manage all the data needed by the model)
│   │   ├── monitor.py (loss monitoring via ensemble forecast uncertainty weighting)
│   │   ├── wrapper.py (model wrapper for inference)
│   ├── networks
│   │   ├── classifier.py (Attention Classifier to be used from the Transfomer's latent space)
│   │   ├── lstm.py (simple lstm model to use as comparison)
│   ├── plotting.py
│   ├── ts_transformer.py (functions to initilalize, train and test the model)
│   ├── utils.py
├── README.md
├── requirements.txt
├── train_example.ipynb
```