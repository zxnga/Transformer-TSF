import torch
import torch.nn as nn
from typing import List, Optional, Type, TypeVar


T = TypeVar('T', bound='Projector')


class Projector(nn.Module):
    """
    Base class for projection-style models:
    - Automatically sets device (GPU if available).
    - Provides save/load methods for model weights.
    """
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Move the model to the appropriate device
        self.to(self.device)

    def save(self, file_path: str) -> None:
        """
        Save model state_dict to the given file path.
        """
        torch.save(self.state_dict(), file_path)

    @classmethod
    def load(
        cls: Type[T],
        file_path: str,
        map_location: Optional[torch.device] = None,
        **model_kwargs
    ) -> T:
        """
        Instantiate the model with `model_kwargs`, load weights from file,
        and move to the correct device.
        """
        model = cls(**model_kwargs)
        # Determine where to load the weights
        loc = map_location or model.device
        state = torch.load(file_path, map_location=loc)
        model.load_state_dict(state)
        model.to(model.device)
        return model


class LSTMModel(Projector):
    """
    LSTM-based sequence-to-vector model.
    Predicts output_size values from the last timestep.
    """
    def __init__(
        self,
        input_size: int = 1,
        hidden_layer_size: int = 50,
        output_size: int = 24,
        num_layers: int = 1
    ) -> None:
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_layer_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def _init_hidden(self, batch_size: int) -> torch.Tensor:
        """
        Create initial hidden and cell states on the correct device.
        """
        h0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_layer_size,
            device=self.device
        )
        c0 = torch.zeros_like(h0)
        return (h0, c0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        hidden = self._init_hidden(batch_size)

        lstm_out, _ = self.lstm(x.to(self.device), hidden)
        # Only use the last timestep
        last_output = lstm_out[:, -1, :]
        return self.linear(last_output)


class AttentionPooling(Projector):
    """
    Attention-based pooling over time:
    Encodes (batch, time, dim) → (batch, dim) by learned weighted sum.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: nn.Module = nn.Tanh,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [input_dim]

        layers: List[nn.Module] = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(activation())
            if dropout:
                layers.append(nn.Dropout(dropout))
            curr_dim = h
        # Final scalar score per timestep
        layers.append(nn.Linear(curr_dim, 1))

        self.attention = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        scores = self.attention(x)                  # (B, T, 1)
        weights = torch.softmax(scores, dim=1)      # normalize over time
        pooled = (x * weights).sum(dim=1)           # (B, dim)
        return pooled


class CNNPooling(Projector):
    """
    1D-CNN sequence encoder:
    Applies stacked Conv1d + activation (+ dropout), global pool, then linear.
    """
    def __init__(
        self,
        input_channels: int,
        conv_channels: Optional[List[int]] = None,
        kernel_sizes: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        paddings: Optional[List[int]] = None,
        pooling: str = 'avg',
        output_size: int = 1,
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        if conv_channels is None:
            conv_channels = [input_channels]
        n_layers = len(conv_channels)
        kernel_sizes = kernel_sizes or [3] * n_layers
        strides = strides or [1] * n_layers
        paddings = paddings or [k // 2 for k in kernel_sizes]

        conv_blocks: List[nn.Module] = []
        in_ch = input_channels
        for out_ch, k, s, p in zip(conv_channels, kernel_sizes, strides, paddings):
            conv_blocks.append(nn.Conv1d(in_ch, out_ch, k, stride=s, padding=p))
            conv_blocks.append(activation())
            if dropout:
                conv_blocks.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.conv_net = nn.Sequential(*conv_blocks)

        if pooling == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pooling == 'max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError("pooling must be 'avg' or 'max'")

        self.fc = nn.Linear(conv_channels[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device).transpose(1, 2)  # (B, C, T)
        feats = self.conv_net(x)
        pooled = self.pool(feats).squeeze(-1)  # (B, C)
        return self.fc(pooled)
