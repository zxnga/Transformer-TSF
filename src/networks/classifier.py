import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        activation=nn.Tanh,
        dropout: float = 0.0,
    ):
        """
        Args:
            input_dim (int): Dimensionality of each time step (typically the encoder's hidden_size).
            hidden_dims (list): List of integers specifying the hidden layer sizes for the attention MLP.
                                If None, defaults to a single hidden layer with size equal to input_dim.
            activation: Activation function class to use (e.g., nn.Tanh, nn.ReLU, nn.GELU).
            dropout (float): Dropout rate applied after each hidden layer (default is 0.0).
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [input_dim]  # Default: one hidden layer of size input_dim

        layers = []
        current_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(current_dim, h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = h
        # Final layer maps to 1 (score per time step)
        layers.append(nn.Linear(current_dim, 1))
        self.attention = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, context_length, input_dim)
        Returns:
            pooled: Tensor of shape (batch_size, input_dim) after weighted pooling over time.
        """
        attn_scores = self.attention(x)             # (batch_size, context_length, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # Normalize over time steps
        pooled = (x * attn_weights).sum(dim=1)        # (batch_size, input_dim)
        return pooled

class Classifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        num_classes: int = 1,
        activation=nn.ReLU,
        dropout: float = 0.0,
    ):
        """
        Args:
            input_dim (int): Input feature dimension.
            hidden_dims (list): List of hidden layer sizes for the classifier MLP.
                                If None, defaults to one hidden layer of size equal to input_dim.
            num_classes (int): Number of output classes.
            activation: Activation function class to use.
            dropout (float): Dropout rate after each hidden layer.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [input_dim]

        layers = []
        current_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(current_dim, h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = h
        layers.append(nn.Linear(current_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class RepresentationClassifier(nn.Module):
    def __init__(
        self,
        encoder_hidden_size: int,
        attn_hidden_dims: list = None,
        classifier_hidden_dims: list = None,
        num_classes: int = 3,
        attn_activation=nn.Tanh,
        classifier_activation=nn.ReLU,
        attn_dropout: float = 0.0,
        classifier_dropout: float = 0.0,
    ):
        """
        Chains an attention pooling module with a classifier.
        
        Args:
            encoder_hidden_size (int): Dimensionality of the encoder's hidden representation.
            attn_hidden_dims (list): Hidden layer sizes for the attention pooling network.
            classifier_hidden_dims (list): Hidden layer sizes for the classifier.
            num_classes (int): Number of target classes.
            attn_activation: Activation function for the attention module.
            classifier_activation: Activation function for the classifier.
            attn_dropout (float): Dropout rate for the attention module.
            classifier_dropout (float): Dropout rate for the classifier.
        """
        super().__init__()
        self.attention_pooling = AttentionPooling(
            input_dim=encoder_hidden_size,
            hidden_dims=attn_hidden_dims,
            activation=attn_activation,
            dropout=attn_dropout,
        )
        self.classifier = Classifier(
            input_dim=encoder_hidden_size,
            hidden_dims=classifier_hidden_dims,
            num_classes=num_classes,
            activation=classifier_activation,
            dropout=classifier_dropout,
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, context_length, encoder_hidden_size)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        pooled = self.attention_pooling(x)  # Aggregated representation (batch_size, encoder_hidden_size)
        logits = self.classifier(pooled)      # Classification logits
        return logits