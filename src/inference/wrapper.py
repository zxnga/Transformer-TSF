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

    def _get_hidden_representations_from_dataloader(self, data_loader):
        latent_representations = []
        encoder = self.model.get_encoder()
        
        for batch in data_loader:
            past_values = batch["past_values"].to(self.device)
            past_time_features = batch["past_time_features"].to(self.device)
            past_observed_mask = batch["past_observed_mask"].to(self.device)
            
            static_cat = (
                batch["static_categorical_features"].to(self.device)
                if self.model.config.num_static_categorical_features > 0
                else None
            )
            static_real = (
                batch["static_real_features"].to(self.device)
                if self.model.config.num_static_real_features > 0
                else None
            )
            
            # Create the unified transformer inputs
            transformer_inputs, loc, scale, static_feat = self.model.create_network_inputs(
                past_values=past_values,
                past_time_features=past_time_features,
                past_observed_mask=past_observed_mask,
                static_categorical_features=static_cat,
                static_real_features=static_real,
                future_values=None,      # We're not forecasting here so we
                future_time_features=None, # don't catre about furure vals
            )
            
            # The encoder takes only the first context_length steps.
            enc_input = transformer_inputs[:, : self.model.config.context_length, ...]
            
            encoder_outputs = encoder(
                inputs_embeds=enc_input,
                output_hidden_states=True,
                return_dict=True,
            )
            
            # Extract the final hidden state (latent representation).
            latent_rep = encoder_outputs.last_hidden_state
            latent_representations.append(latent_rep.cpu().detach().numpy())
            
        # Stack or process latent_representations as needed
        return np.vstack(latent_representations)

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
        return np.vstack(forecasts) #np.median(pred, 1)

    def predict(self, data_loader: IterableSlice, form_return: str = 'sample') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        values = self._prediction_from_dataloader(data_loader)
        return self.modify_output(values, form_return)
    
    @staticmethod
    def modify_output(values: np.ndarray, form_return: str = 'sample'):
        if form_return == 'sample':
            return values, None
        elif form_return == 'single':
            return np.median(values, 1), np.std(values, 1) 
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

        self.data_handler = None
        self.inference_helper = None
        self.reset_buffers()

        self.last_action_ingest = None

    @property
    def context_length(self):
        return self.data_handler.context_length

    @property
    def full_context_length(self):
        return self.data_handler.full_context_length

    @property
    def context(self):
        return self.data_handler.context_buffer.context[-self.context_length:]

    @property
    def full_context(self):
        return self.data_handler.context_buffer.context

    @property
    def prediction_and_uncertainty(self):
        pred_buf = getattr(self.data_handler, "pred_buffer", None)
        if pred_buf:
            return (getattr(pred_buf, "pred_buffer", None),
                    getattr(pred_buf, "uncert_buffer", None))
        return None

    @property
    def nb_prediction(self):
        return self.data_handler.pred_buffer.current_time + 1

    def reset_buffers(self):
        self.data_handler = TFDataHandler(
            self.transformer.config, self.freq,
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
        self.last_action_ingest = True

    def predict(self,
        batch_size: int = 1, # at inference only one row
        item_id: str = 'T0',
        # form_return = 'sample',
    ):
        #TODO: check how to handle update of pred_buffer and return type of user
        "forecast values"
        data_loader = self.data_handler.get_infer_dataloader(batch_size, item_id)
        values, _ = self.inference_helper.predict(data_loader, 'sample')

        if self.loss_window > 0:
            values_, uncertainty = self.inference_helper.modify_output(values, 'single')
            #TODO: better handle dimension + check for non-single batch
            self.data_handler.update_prediction_buffer(values_.squeeze(), uncertainty.squeeze())

        self.last_action_ingest = False
        return values

    def get_last_points_predictions(self):
        """
        Carefull ! As soon as a prediction is made for a timestep, this timestep is considered valid
        and predictions made in the past for that timestep are returned. We may not have access at that point
        at the true value for that timestep. -> need to ingest the true value first
        -> we make prediction before ingesting as the first prediction uses the context
        """
        #TODO: verify alignement if start with ingest + maybe crop the list here
        if not self.last_action_ingest:
            print("Lest prediction included, no given True value for it. Ingest the next value for alignement!")
        return self.data_handler.get_past_points_predictions()

    def get_last_true_points(self):
        return self.data_handler.get_past_true_points()

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



        




    