from typing import NamedTuple, Optional, Dict, List

import numpy as np
import datetime

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
        *args,
        **kwargs,
    ):
        self.context_length = context_length
        self.current_context_size = 0 # if initial context < context_length
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

    def initialize(
        self,
        context: np.ndarray,
        start: datetime.datetime,
        static_cat_features: Optional[np.ndarray]=None,
        static_real_features: Optional[np.ndarray]=None,
        dynamic_real_features: Optional[np.ndarray]=None,
    ):
        if self.num_static_cat_features > 0:
            assert static_cat_features.shape == self.static_cat_features.shape,(
                f"Expected static_cat shape: {self.static_cat_features.shape}, got {static_cat_features.shape}")
            self.static_cat_features = static_cat_features

        if self.num_static_real_features > 0:
            assert static_real_features.shape == self.static_real_features.shape,(
                f"Expected static_real shape: {self.static_real_features.shape}, got {static_real_features.shape}")
            # cast to avoid loosing future decimal part if initial values come from list of int
            self.static_real_features = static_real_features.astype(np.float32)

        # TODO modify when context_size < context_length add padding ect
        if self.num_dynamic_real_features > 0:
            assert dynamic_real_features.shape == self.dynamic_real_features.shape,(
                f"Expected dynamic_real shape: {self.dynamic_real_features.shape}, got {dynamic_real_features.shape}")
            self.dynamic_real_features = dynamic_real_features.astype(np.float32)
        
        # keep only necessary context
        if context.size < self.context_length:
            self.current_context_size = context.size
            context = np.pad(
                context,
                (self.context_length - context.size, 0),
                mode='constant').astype(np.float32)
        else:
            context = context[-self.context_length:]

        self.context = context
        self.start = start

    def update(self, value: float, dynamic_real_features: Optional[np.ndarray]=None):
        # dynamic_real_features is of size (num_dynamic_real_features,)
        self.context = np.roll(self.context, -1)
        self.context[-1] = value
        if self.num_dynamic_real_features > 0:
            self.dynamic_real_features = np.roll(self.dynamic_real_features, -1)
            self.dynamic_real_features[-1] = dynamic_real_features
        self.start = self.start + datetime.timedelta(**self.timedelta)
        if self.current_context_size < self.context_length:
            self.current_context_size += 1

    def get_values(self):
        data = (
            self.context[-self.current_context_size:],
            self.start,
            self.static_cat_features if self.static_cat_features.size!=0 else None,
            self.static_real_features if self.static_real_features.size!=0 else None,
            self.dynamic_real_features[-self.current_context_size:] if self.dynamic_real_features.size!=0 else None,
        )
        return BufferValues(*tuple(data))

class TSLossBuffer(TSBuffer):
    def __init__(
        self,
        context_length: int,
        num_static_cat_features: int,
        num_static_real_features: int,
        num_dynamic_real_features: int,
        timedelta: Dict[str,int],
        loss_window: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            context_length,
            num_static_cat_features,
            num_static_real_features,
            num_dynamic_real_features,
            timedelta,
            *args,
            **kwargs)
        
        self.loss_window = loss_window
        self.new_vals = 0
        # store only 1 array
        if loss_window > self.context_length:
            self.context = np.zeros((loss_window), dtype=np.float32)

    def initialize(self,
        context: np.ndarray,
        start: datetime.datetime,
        static_cat_features: Optional[np.ndarray]=None,
        static_real_features: Optional[np.ndarray]=None,
        dynamic_real_features: Optional[np.ndarray]=None,
    ):
        super().initialize(
            context,
            start,
            static_cat_features,
            static_real_features,
            dynamic_real_features)

        if self.loss_window > self.context_length: # pad to get full window size
            self.context = np.pad(
                self.context,
                (self.loss_window - self.context_length, 0),
                mode='constant')       
        
    def get_values(self):
        assert self.context_length >= self.current_context_size
        data = (
            self.context[-min(self.context_length,self.current_context_size):],  # Sliced context array if window > context_length
            self.start,
            self.static_cat_features if self.static_cat_features.size != 0 else None,
            self.static_real_features if self.static_real_features.size != 0 else None,
            self.dynamic_real_features[-min(self.context_length,self.current_context_size):] if self.dynamic_real_features.size != 0 else None,
        )
        return BufferValues(*data)

    def update(self, value: float, dynamic_real_features: Optional[np.ndarray]=None):
        self.new_vals += 1
        super().update(value, dynamic_real_features)

    def get_true_window(self):
        # to get true window we have to keep a counter to avoid returning initial context which are not part of the values we predicted
        if self.new_vals == 0:
            return np.empty(0)
        if self.new_vals >= self.loss_window:
            return self.context[-self.loss_window:]
        else:
            return self.context[-self.new_vals:]

class CircularHorizonPredictionBuffer:
    def __init__(self, loss_window, forecast_length):
        """
        Initializes the circular buffer for storing predictions by forecast horizon.

        Parameters:
            loss_window (int): The number of recent true time steps used for loss monitoring.
            forecast_length (int): The number of forecasted values produced at each update.

            #TODO: chnage name of buffer (predbuffer in upper class) + change property name in upper class /!
        """
        # The buffer size is set to loss_window + forecast_length to cover all future predictions that may be needed.
        self.loss_window = loss_window
        self.forecast_length = forecast_length
        self.buffer_size = loss_window + forecast_length

        # Create buffers for predictions and uncertainties.
        # Each index in the buffer corresponds to a forecast horizon relative to some true time.
        self.pred_buffer = [ [] for _ in range(self.buffer_size) ]
        self.uncert_buffer = [ [] for _ in range(self.buffer_size) ]
        # For each slot, keep track of the cycle number (computed as predicted_time // buffer_size).
        self.cycles = [-1] * self.buffer_size

        # Keep track of the current true time (absolute time step).
        self.current_time = -1

    def update(self, forecast, uncertainties):
        """
        Updates the buffer with a new forecast array.
        
        The forecast and uncertainties are assumed to be 0-indexed:
          - forecast[0] is the prediction for the current true time.
          - forecast[1] is for current time + 1, etc.
        
        For each forecast horizon h, the prediction is stored in the buffer slot corresponding to (current_time + h) % buffer_size.
        Before appending, if that slot belongs to an older cycle, it is reset (cleared).

        Parameters:
            forecast (np.array): Array of predicted values with shape (forecast_length,).
            uncertainties (np.array): Array of uncertainties with shape (forecast_length,).

        Returns:
            tuple: (current_predictions, current_uncertainties) for the current true time step.
                   Note that these are not cleared immediately, but will be reset once their slot is overwritten in a future cycle.
        """
        # Advance the true time.
        self.current_time += 1
        t = self.current_time

        # Process each forecast horizon.
        for h, (pred, uncert) in enumerate(zip(forecast, uncertainties)):
            # Compute the absolute time for which this prediction is intended.
            predicted_time = t + h
            # Determine the circular buffer index.
            idx = predicted_time % self.buffer_size
            # Compute the cycle number for this predicted time.
            new_cycle = predicted_time // self.buffer_size
            # If the slot's cycle is not the same as the new cycle, reset the slot.
            if self.cycles[idx] != new_cycle:
                self.pred_buffer[idx] = []
                self.uncert_buffer[idx] = []
                self.cycles[idx] = new_cycle
            # Append the new prediction and its uncertainty.
            self.pred_buffer[idx].append(pred)
            self.uncert_buffer[idx].append(uncert)

        # Retrieve the predictions for the current true time (without clearing them).
        current_idx = t % self.buffer_size
        current_predictions = self.pred_buffer[current_idx]
        current_uncertainties = self.uncert_buffer[current_idx]
        return current_predictions, current_uncertainties

    def get_buffer_state(self):
        """
        Returns the current state of the buffer for inspection.
        Each element is a tuple: (predictions, uncertainties, cycle).
        """
        return list(zip(self.pred_buffer, self.uncert_buffer, self.cycles))
