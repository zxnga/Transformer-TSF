from typing import Optional, Dict, Any, Tuple, Union, List

import numpy as np
import datetime
from accelerate import Accelerator
from gluonts.itertools import IterableSlice
from gluonts.transform import Transformation

from .data.helper import TFDataHandler

class TFInferenceHelper:
    """
        Transformer Wrapper just to handle the inference of the model in production
    """
    def __init__(
        self,
        trained_model,
        use_accelerator=True,
        *args,
        **kwargs
    ):
        self.model = trained_model
        self.model.eval()

        if use_accelerator:
            accelerator = Accelerator()
            self.device = accelerator.device
            self.model.to(self.device)
        else:
            self.device = model.device
        print(self.model.device)

    def _prediction_from_dataloader(self, data_loader: IterableSlice):
        forecasts = []
        for batch in data_loader:
            outputs = self.model.generate(
                static_categorical_features=batch["static_categorical_features"].to(self.device)
                if self.model.config.num_static_categorical_features > 0
                else None,
                static_real_features=batch["static_real_features"].to(self.device)
                if self.model.config.num_static_real_features > 0
                else None,
                past_time_features=batch["past_time_features"].to(self.device),
                past_values=batch["past_values"].to(self.device),
                future_time_features=batch["future_time_features"].to(self.device),
                past_observed_mask=batch["past_observed_mask"].to(self.device),
            )
            forecasts.append(outputs.sequences.cpu().numpy())
        
        return np.vstack(forecasts) #np.median(pred, 0)

    def predict(self, data_loader: IterableSlice, form_return: str = 'sample') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        values = self._prediction_from_dataloader(data_loader)
        return self.modify_output(values, form_return)
    
    @staticmethod
    def modify_output(values: np.ndarray, form_return: str = 'sample'):
        if form_return == 'sample':
            return values, None
        elif form_return == 'single':
            return np.median(values, 0), np.std(values, 0) 
        else:
            raise NotImplementedError

class TFWrapper:
    """
    """
    def __init__(
        self,
        trained_model,
        freq: str,
        data_transformation: Transformation=None,
        inference_config: Dict[str, Any] = {},
        loss_window=0
    ):
        self.transformer = trained_model
        self.model_config = trained_model.config
        self.freq = freq
        self.inference_config = inference_config
        self.loss_window = loss_window
        self.data_transformation = data_transformation

        self.reset_buffers()

    @property
    def context(self):
        return self.data_handler.context_buffer.context

    @property
    def prediction_and_uncertainty(self):
        pred_buf = getattr(self.data_handler, "pred_buffer", None)
        if pred_buf:
            return (getattr(pred_buf, "pred_buffer", None),
                    getattr(pred_buf, "uncert_buffer", None))
        return None

    def reset_buffers(self):
        self.data_handler = TFDataHandler(
            self.transformer.config,self.freq,
            self.data_transformation, self.loss_window
        )
        self.inference_helper = TFInferenceHelper(
            self.transformer, **self.inference_config)

    def initialize_buffer(
        self,
        context: Union[List[float], np.ndarray],
        start: datetime.datetime,
        static_cat_features: Optional[Union[List[float], np.ndarray]]=None,
        static_real_features: Optional[Union[List[float], np.ndarray]]=None,
        dynamic_real_features: Optional[Union[List[float], np.ndarray]]=None,
    ) -> None:

        self.reset_buffers()
        self.data_handler.initialize_buffer(
            self._maybe_cast_type(context),
            start,
            self._maybe_cast_type(static_cat_features),
            self._maybe_cast_type(static_real_features),
            self._maybe_cast_type(dynamic_real_features)
        )

    def ingest(self, value: float, dynamic_real_features: Optional[np.ndarray]=None):
        "ingest incomming time-series data"
        self.data_handler.update_context_buffer(value, dynamic_real_features)

    def predict(self,
        batch_size: int = 1, # at inference only one row
        item_id: str = 'T0',
        # form_return = 'sample',
    ):
        #TODO: check how to handle update of pred_buffer and return type of user
        "forecast values"
        data_loader = self.data_handler.get_infer_dataloader(batch_size, item_id)
        values = self.inference_helper.predict(data_loader, 'sample')
        if self.loss_window > 0:
            values_ = self.inference_helper.modify_output(values, 'single')
            self.data_handler.update(*values_)
        return values

    @staticmethod
    def _maybe_cast_type(array: Optional[Union[List[float], np.ndarray]]) -> np.ndarray:
        if array is None:
            return array
        if isinstance(array, list):
            return np.array(array, dtype=np.float32)
        elif isinstance(array, np.ndarray):
            if array.dtype != np.float32:
                return array.astype(np.float32)
            return array
        else:
            raise TypeError("Input must be either a list of floats or a numpy.ndarray")



        




    