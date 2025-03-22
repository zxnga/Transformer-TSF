from typing import NamedTuple, Optional, Dict, List

import numpy as np

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

    def update_buffer(self, value: float, dynamic_real_features: Optional[np.ndarray]=None):
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

    def initialize_buffer(
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

class TSLossBuffer(TSBuffer):
    def __init__(
        self,
        loss_window: int,
        context_length: int,
        num_static_cat_features: int,
        num_static_real_features: int,
        num_dynamic_real_features: int,
        timedelta: Dict[str,int],
    ):
        super().__init__(
            context_length,
            num_static_cat_features,
            num_static_real_features,
            num_dynamic_real_features,
            timedelta)
        
        self.loss_window = loss_window
        self.monitor_counter = 0
        self.monitor_loss_ready = False
        self.window_full = False
        # store only 1 array
        if loss_window > self.context_length:
            self.context = np.zeros((loss_window), dtype=np.float32)

    def initialize_buffer():
        # TODO context can now be of size window
        pass
        
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

    def update_buffer(self, value: float, dynamic_real_features: Optional[np.ndarray]=None):
        self.monitor_counter += 1
        if self.monitor_counter == self.context_length:
            self.monitor_loss_ready = True
        if self.monitor_counter == self.loss_window:
            self.window_full =  True
        super().update_buffer(value, dynamic_real_features)

    #to get true window we have to keep a counter to avoid returning initial context which are not part of the values we predicted
    # The context after the initial one are values we predicted so are the ones we return to monitor the loss
    # once we added n=context_legth new values in the buffer we can just return the whole window
    def get_true_window(self):
        if self.monitor_loss_ready:
            return self.context

    