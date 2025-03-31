from typing import Optional, Iterable, Dict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
import pandas as pd
import os
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from pydantic import Field

from transformers import (
    PretrainedConfig,
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerForPrediction
)

from accelerate import Accelerator
from evaluate import load

from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from gluonts.transform import Transformation
from gluonts.transform.sampler import InstanceSampler
from datasets import Dataset
from functools import partial
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import Cached, Cyclic
from gluonts.dataset.loader import as_stacked_batches

from gluonts.time_feature import (
    get_lags_for_frequency, 
    time_features_from_frequency_str,
    TimeFeature,
    get_seasonality
)

from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
    RenameFields,
)

from .utils import transform_start_field

class SlidingWindowSampler(InstanceSampler):
    past_length: int = Field(..., description="Number of past time steps (context + lags)")
    future_length: int = Field(..., description="Number of future time steps")
    step: int = Field(1, description="Sliding window step size")

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        """
        Given a time series array (e.g. from the target field "values"),
        return all valid start indices for a window of length:
            past_length + future_length.
        """
        window_length = self.past_length + self.future_length
        T = ts.shape[0]
        if T < window_length:
            return np.array([], dtype=int)
        start_max = T - window_length + 1
        return np.arange(0, start_max, self.step)

def create_transformation(freq: str, config: PretrainedConfig) -> Transformation:
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)

    # a bit like torchvision.transforms.Compose
    return Chain(
        # step 1: remove static/dynamic fields if not specified
        [RemoveFields(field_names=remove_field_names)]
        # step 2: convert the data to NumPy (potentially not needed)
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                )
            ]
            if config.num_static_categorical_features > 0
            else []
        )
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                )
            ]
            if config.num_static_real_features > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.TARGET,
                # we expect an extra dim for the multivariate case:
                expected_ndim=1 if config.input_size == 1 else 2,
            ),
            # step 3: handle the NaN's by filling in the target with zero
            # and return the mask (which is in the observed values)
            # true for observed values, false for nan's
            # the decoder uses this mask (no loss is incurred for unobserved values)
            # see loss_weights inside the xxxForPrediction model
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # step 4: add temporal features based on freq of the dataset
            # month of year in the case when freq="M"
            # these serve as positional encodings
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(freq),
                pred_length=config.prediction_length,
            ),
            # step 5: add another temporal feature (just a single number)
            # tells the model where in its life the value of the time series is,
            # sort of a running counter
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
            # step 6: vertically stack all the temporal features into the key FEAT_TIME
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + (
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if config.num_dynamic_real_features > 0
                    else []
                ),
            ),
            # step 7: rename to match HuggingFace names
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask",
                }
            ),
        ]
    )

def create_train_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    num_batches_per_epoch: int,
    mode: str = 'train',
    step: int = 1,
    shuffle_buffer_length: Optional[int] = None,
    cache_data: bool = True,
    **kwargs,
) -> Iterable:
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")
    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")
    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_values",
        "future_observed_mask",
    ]

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=True)
    
    if cache_data:
        transformed_data = Cached(transformed_data)

    # we initialize a Training instance
    instance_splitter = create_instance_splitter(config, mode, step)

    # the instance splitter will sample a window of
    # context length + lags + prediction length (from all the possible transformed time series)
    # randomly from within the target time series and return an iterator.
    stream = Cyclic(transformed_data).stream()
    training_instances = instance_splitter.apply(stream)
    
    return as_stacked_batches(
        training_instances,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=TRAINING_INPUT_NAMES,
        output_type=torch.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )


def create_instance_splitter(
    config: PretrainedConfig,
    mode: str,
    step: int = 1,
    train_sampler: Optional[InstanceSampler] = None,
    validation_sampler: Optional[InstanceSampler] = None,
) -> Transformation:
    assert mode in ["train", "validation", "test", "systematic"]

    instance_sampler = {
        "train": train_sampler
        or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=config.prediction_length
        ),
        "validation": validation_sampler
        or ValidationSplitSampler(min_future=config.prediction_length),
        "test": TestSplitSampler(),
        "systematic": SlidingWindowSampler(
            past_length=config.context_length + max(config.lags_sequence),
            future_length=config.prediction_length,
            step=step,
        )
    }[mode]

    return InstanceSplitter(
        target_field="values",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config.context_length + max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=["time_features", "observed_mask"],
    )

def create_test_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    mode = 'test',
    **kwargs,
):
    # print(mode)
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=False)

    # we create a Test Instance splitter which will sample the very last
    # context window seen during training only for the encoder.
    instance_sampler = create_instance_splitter(config, mode)

    # we apply the transformations in test mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=False)

    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )

def create_backtest_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    past_length=None,
    **kwargs,
):
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data)

    # we create a Validation Instance splitter which will sample the very last
    # context window seen during training only for the encoder.
    instance_sampler = create_instance_splitter(config, "validation")

    # we apply the transformations in train mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=True)
    
    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )

def setup_training(
    train_df: pd.DataFrame,
    freq: str,
    batch_size: int = 32,
    num_batches_per_epoch: int = 16,
    max_lags: Optional[int] = None,
    transformer_config: Optional[Dict] = {},
):
    train_data = Dataset.from_pandas(train_df, preserve_index=False)
    train_data.set_transform(partial(transform_start_field, freq=freq))

    lags_sequence = get_lags_for_frequency(freq)
    if max_lags:
        lags_sequence = [i for i in lags_sequence if i<=max_lags]
    time_features = time_features_from_frequency_str(freq)
    
    transformer_config['num_time_features'] = len(time_features) + 1
    transformer_config['lags_sequence'] = lags_sequence
    config = TimeSeriesTransformerConfig(**transformer_config)
    transformer = TimeSeriesTransformerForPrediction(config)

    train_dataloader = create_train_dataloader(
                            config=config,
                            freq=freq,
                            data=train_data,
                            batch_size=32,
                            num_batches_per_epoch=16)

    return transformer, train_dataloader

#TODO: modify constructor add config + frq
def setup_testing_data(test_df: pd.DataFrame):
    test_data = Dataset.from_pandas(test_df, preserve_index=True)
    test_data.set_transform(partial(transform_start_field, freq=freq))

    test_dataloader = create_backtest_dataloader(
    config=config,
    freq=freq,
    data=test_data,
    batch_size=16)

    return test_dataloader, test_data

def train(
    transformer,
    train_dataloader,
    epochs: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    accelerator = Accelerator()
    device = accelerator.device
    transformer.to(device)

    if not optimizer:
        print('Using default AdamW parametres.')
        optimizer = AdamW(transformer.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=1e-1)

    transformer, optimizer, train_dataloader = accelerator.prepare(
        transformer,
        optimizer,
        train_dataloader,
    )

    transformer.train()
    list_loss = []
    for epoch in range(epochs):
        total_loss = 0
        for idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = transformer(
                static_categorical_features=batch["static_categorical_features"].to(device)
                if transformer.config.num_static_categorical_features > 0
                else None,
                static_real_features=batch["static_real_features"].to(device)
                if transformer.config.num_static_real_features > 0
                else None,
                past_time_features=batch["past_time_features"].to(device),
                past_values=batch["past_values"].to(device),
                future_time_features=batch["future_time_features"].to(device),
                future_values=batch["future_values"].to(device),
                past_observed_mask=batch["past_observed_mask"].to(device),
                future_observed_mask=batch["future_observed_mask"].to(device),
            )
            # print(outputs)
            loss = outputs.loss
            total_loss += loss.item()


            # Backpropagation
            accelerator.backward(loss)
            optimizer.step()
        list_loss.append(total_loss)

    return transformer, list_loss

def test(
    transformer,
    test_dataloader,
):
    accelerator = Accelerator()
    device = accelerator.device
    transformer.to(device)
    transformer.eval()
    forecasts = []

    for batch in test_dataloader:
        outputs = transformer.generate(
            static_categorical_features=batch["static_categorical_features"].to(device)
            if transformer.config.num_static_categorical_features > 0
            else None,
            static_real_features=batch["static_real_features"].to(device)
            if transformer.config.num_static_real_features > 0
            else None,
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
        )
        forecasts.append(outputs.sequences.cpu().numpy())
    
    return np.vstack(forecasts)


def evaluate(test_data, forecasts, prediction_length, freq):
    mase_metric = load("evaluate-metric/mase")
    smape_metric = load("evaluate-metric/smape")

    forecast_median = np.median(forecasts, 1)

    mase_metrics_transformer = []
    smape_metrics_transformer = []
    for item_id, ts in enumerate(test_data):
        training_data = ts["target"][:-prediction_length]
        ground_truth = ts["target"][-prediction_length:]
        mase = mase_metric.compute(
            predictions=forecast_median[item_id],
            references=np.array(ground_truth),
            training=np.array(training_data),
            periodicity=get_seasonality(freq))
        mase_metrics_transformer.append(mase["mase"])
        
        smape = smape_metric.compute(
            predictions=forecast_median[item_id], 
            references=np.array(ground_truth), 
        )
        smape_metrics_transformer.append(smape["smape"])
    
    return mase_metrics_transformer, smape_metrics_transformer