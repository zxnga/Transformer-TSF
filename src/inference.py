from typing import NamedTuple, Optional, List, Dict

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from datasets import Dataset
from transformers import PretrainedConfig
from gluonts.transform import Transformation
from gluonts.dataset.field_names import FieldName

from gluonts.time_feature import (
    time_features_from_frequency_str
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
    RenameFields
)

from .utils import parse_frequency, transform_start_field
from .ts_transformer import create_test_dataloader

COLS_INFER_DF = [
    'target', 'start', 'feat_static_cat', 'feat_static_real',
    'feat_dynamic_real', 'item_id']


class BufferValues(NamedTuple):
    context: np.ndarray
    start: datetime.datetime
    feat_static_cat: List[int]
    feat_static_real: List[float]
    feat_dynamic_real: np.ndarray


class TSBuffer:
    def __init__(
        self,
        context_length: int,
        num_static_cat_features: int,
        num_static_real_features: int,
        num_dynamic_real_features: int,
        timedelta: Dict[str,int],
        
    ):
        self.context_length = context_length
        self.num_static_cat_features = num_static_cat_features
        self.num_static_real_features = num_static_real_features
        self.num_dynamic_real_features = num_dynamic_real_features
        self.timedelta = timedelta

        self.context = np.zeros((context_length), dtype=np.float32)
        self.start = None
        self.static_cat_features = np.zeros(num_static_cat_features)
        self.static_real_features = np.zeros((num_static_real_features), dtype=np.float32)
        self.dynamic_real_features = np.zeros(
            (num_dynamic_real_features,context_length), dtype=np.float32)

    def update_buffer(self, value: float, dynamic_real_features: Optional[np.ndarray]=None):
        # dynamic_real_features is of size (num_dynamic_real_features,)
        self.context = np.roll(self.context, -1)
        self.context[-1] = value
        if self.num_dynamic_real_features > 0:
            self.dynamic_real_features = np.roll(self.dynamic_real_features, -1)
            self.dynamic_real_features[-1] = dynamic_real_features
        self.start = self.start + datetime.timedelta(**self.timedelta)

    def get_values(self):
        data = (
            self.context,
            self.start,
            self.static_cat_features if self.static_cat_features.size!=0 else None,
            self.static_real_features if self.static_real_features.size!=0 else None,
            self.dynamic_real_features if self.dynamic_real_features.size!=0 else None,
        )
        return BufferValues(*tuple(data))

    def initialize_buffer(
        self,
        context: np.ndarray,
        start: datetime.datetime,
        static_cat_features: Optional[np.ndarray]=None,
        static_real_features: Optional[np.ndarray]=None,
        dynamic_real_features: Optional[np.ndarray]=None,
    ):
        # TODO: use seter instead
        assert context.shape == self.context.shape, (
            f"Expected context shape: {self.context.shape}, got {context.shape}")

        if self.num_static_cat_features > 0:
            assert static_cat_features.shape == self.static_cat_features.shape,(
                f"Expected static_cat shape: {self.static_cat_features.shape}, got {static_cat_features.shape}")
            self.static_cat_features = static_cat_features

        if self.num_static_real_features > 0:
            assert static_real_features.shape == self.static_real_features.shape,(
                f"Expected static_real shape: {self.static_real_features.shape}, got {static_real_features.shape}")
            self.static_real_features = static_real_features

        if self.num_dynamic_real_features > 0:
            assert dynamic_real_features.shape == self.dynamic_real_features.shape,(
                f"Expected dynamic_real shape: {self.dynamic_real_features.shape}, got {dynamic_real_features.shape}")
            self.dynamic_real_features = dynamic_real_features
        
        self.context = context
        self.start = start

class InferenceHelper:
    def __init__(self, config: PretrainedConfig, freq: str, transformation: Transformation = None):
        self.config = config
        self.freq = freq
        self.transformation = transformation
        if transformation is None:
            self.transformation = self._create_transformation()
        
        self.buffer = TSBuffer(
            config.context_length,
            config.num_static_categorical_features,
            config.num_static_real_features,
            config.num_dynamic_real_features,
            parse_frequency(freq))

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
        static_cat_features: Optional[List[int]]=None,
        static_real_features: Optional[List[float]]=None,
        dynamic_real_features: Optional[np.ndarray]=None,
    ):
        # cast to avoid loosing future decimal part if initial values come from list of int
        if static_real_features is not None:
            static_real_features = static_real_features.astype(np.float32)
        if dynamic_real_features is not None:
            dynamic_real_features = dynamic_real_features.astype(np.float32)

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
        infer_loader = create_test_dataloader(
            config=self.config,
            freq=self.freq,
            data=self._get_inference_df(item_id),
            batch_size=batch_size,
            mode='infer')

        return next(iter(infer_loader)) #only 1 batch