from typing import Optional, Dict

import pandas as pd
import numpy as np
import datetime
import torch.nn
from datasets import Dataset
from transformers import PretrainedConfig
from gluonts.transform import Transformation
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.time_feature import time_features_from_frequency_str
from functools import partial

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
    RenameFields
)

from .buffer import TSBuffer, TSLossBuffer, CircularHorizonPredictionBuffer
from ...utils import parse_frequency, transform_start_field

COLS_INFER_DF = [
    'target', 'start', 'feat_static_cat', 'feat_static_real',
    'feat_dynamic_real', 'item_id']

class TFDataHandler:
    """
        Transformer Data Helper to handle the data needed by the model:
            - Context needed to make the inference
            - Opionnaly true values to be able to monitor the loss of the model
    """
    def __init__(
        self,
        config: PretrainedConfig,
        freq: str,
        transformation: Transformation = None,
        loss_window: int = 0,
    ):
        self.config = config
        self.freq = freq
        self.transformation = transformation
        self.context_length = config.context_length
        # actual context pass during inference is context_length + max(lags_sequence)
        self.full_context_length = config.context_length + max(config.lags_sequence)
        if transformation is None:
            self.transformation = self._create_transformation()
        
        context_buffer_cls = TSBuffer
        self.pred_buffer = None
        if loss_window > 0:
            context_buffer_cls = TSLossBuffer
            self.pred_buffer = CircularHorizonPredictionBuffer(
                loss_window, self.config.prediction_length)

        self.context_buffer = context_buffer_cls(
            self.full_context_length,
            config.num_static_categorical_features,
            config.num_static_real_features,
            config.num_dynamic_real_features,
            parse_frequency(freq),
            loss_window)

    #TODO: remove transformation form class
    def _create_transformation(self) -> Transformation:
        remove_field_names = []
        if self.config.num_static_real_features == 0:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if self.config.num_dynamic_real_features == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
        if self.config.num_static_categorical_features == 0:
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
                if self.config.num_static_categorical_features > 0
                else []
            )
            + (
                [
                    AsNumpyArray(
                        field=FieldName.FEAT_STATIC_REAL,
                        expected_ndim=1,
                    )
                ]
                if self.config.num_static_real_features > 0
                else []
            )
            + [
                AsNumpyArray(
                    field=FieldName.TARGET,
                    # we expect an extra dim for the multivariate case:
                    expected_ndim=1 if self.config.input_size == 1 else 2,
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
                    time_features=time_features_from_frequency_str(self.freq),
                    pred_length=self.config.prediction_length,
                ),
                # step 5: add another temporal feature (just a single number)
                # tells the model where in its life the value of the time series is,
                # sort of a running counter
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.config.prediction_length,
                    log_scale=True,
                ),
                # step 6: vertically stack all the temporal features into the key FEAT_TIME
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if self.config.num_dynamic_real_features > 0
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

    def initialize_buffer(
        self,
        context: np.ndarray,
        start: datetime.datetime,
        static_cat_features: Optional[np.ndarray]=None,
        static_real_features: Optional[np.ndarray]=None,
        dynamic_real_features: Optional[np.ndarray]=None,
    ):
        self.context_buffer.initialize(
            context.astype(np.float32),
            start,
            static_cat_features,
            static_real_features,
            dynamic_real_features)

    def update_context_buffer(self, value: float, dynamic_real_features: Optional[np.ndarray]=None):
        self.context_buffer.update(value, dynamic_real_features)

    def _get_inference_df(self, item_id: str) -> pd.DataFrame:
        infer_df = pd.DataFrame(columns=COLS_INFER_DF)
        values = self.context_buffer.get_values()
        infer_df.loc[0] = [*values, item_id]

        infer_df =  Dataset.from_pandas(infer_df, preserve_index=True)
        infer_df.set_transform(partial(transform_start_field, freq=self.freq))
        return infer_df

    def _create_inference_dataloader(
        self,
        data: pd.DataFrame,
        batch_size: int,
        **kwargs,
    ):
        PREDICTION_INPUT_NAMES = [
            "past_time_features",
            "past_values",
            "past_observed_mask",
            "future_time_features",
        ]
        if self.config.num_static_categorical_features > 0:
            PREDICTION_INPUT_NAMES.append("static_categorical_features")

        if self.config.num_static_real_features > 0:
            PREDICTION_INPUT_NAMES.append("static_real_features")

        transformed_data = self.transformation.apply(data, is_train=False)
        instance_sampler = InstanceSplitter(
                target_field="values",
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=TestSplitSampler(),
                past_length=self.full_context_length,
                future_length=self.config.prediction_length,
                time_series_fields=["time_features", "observed_mask"])

        testing_instances = instance_sampler.apply(transformed_data, is_train=False)

        return as_stacked_batches(
            testing_instances,
            batch_size=batch_size,
            output_type=torch.tensor,
            field_names=PREDICTION_INPUT_NAMES,
        )

    def get_infer_dataloader(
        self,
        batch_size: int = 1, # at inference only one row
        item_id: str = 'T0'
    ):
        infer_loader = self._create_inference_dataloader(
            data=self._get_inference_df(item_id),
            batch_size=batch_size)

        # return next(iter(infer_loader)) #only 1 batch
        return infer_loader