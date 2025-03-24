from typing import Optional, Dict

import pandas as pd
import numpy as np
import datetime
from datasets import Dataset
from transformers import PretrainedConfig
from gluonts.transform import Transformation
from gluonts.dataset.field_names import FieldName

from gluonts.time_feature import time_features_from_frequency_str

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

from .buffer import TSBuffer, TSLossBuffer
from ..utils import parse_frequency, transform_start_field
from ..ts_transformer import create_test_dataloader

COLS_INFER_DF = [
    'target', 'start', 'feat_static_cat', 'feat_static_real',
    'feat_dynamic_real', 'item_id']

class InferenceHelper:
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
        if transformation is None:
            self.transformation = self._create_transformation()
        
        buffer_cls = TSLossBuffer if loss_window > 0 else TSBuffer
        self.buffer = buffer_cls(
            config.context_length,
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
        self.buffer.initialize_buffer(
            context.astype(np.float32),
            start,
            static_cat_features,
            static_real_features,
            dynamic_real_features)

    def update_buffer(self, value: float, dynamic_real_features: Optional[np.ndarray]=None):
        self.buffer.update_buffer(value, dynamic_real_features)

    def _get_inference_df(self, item_id: str):
        infer_df = pd.DataFrame(columns=COLS_INFER_DF)
        values = self.buffer.get_values()
        infer_df.loc[0] = [*values, item_id]

        infer_df =  Dataset.from_pandas(infer_df, preserve_index=True)
        infer_df.set_transform(partial(transform_start_field, freq=self.freq))
        return infer_df

    def get_infer_data(
        self,
        batch_size: int = 1, # at inference only one row
        item_id: str = 'T0'
    ):
        infer_loader = create_test_dataloader(
            config=self.config,
            freq=self.freq,
            data=self._get_inference_df(item_id),
            batch_size=batch_size,
            mode='infer')

        return next(iter(infer_loader)) #only 1 batch