from typing import Optional, List

import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
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
        hidden_dims: Optional[List[int]] = None,
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
        attn_hidden_dims: Optional[List[int]] = None,
        classifier_hidden_dims: Optional[List[int]] = None,
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
        logits = self.classifier(pooled)    # logits
        return logits

def train_classifier(model, classifier, train_loader, optimizer, criterion, device, num_epochs, valid_loader=None):
    """
    Trains the classifier using latent representations extracted from the given training dataloader
    and evaluates on a validation dataloader.

    Args:
        model: The full transformer model (e.g., TimeSeriesTransformerForPrediction) used to extract latents.
        classifier: The classifier network that maps latent representations to class logits.
        train_loader: DataLoader for training data. Each batch is a dict with keys needed by get_latents_and_labels_from_batch.
        valid_loader: DataLoader for validation data with the same structure as train_loader.
        optimizer: Optimizer for the classifier (e.g., Adam).
        criterion: Loss function (e.g., CrossEntropyLoss).
        device: Torch device (e.g., 'cuda' or 'cpu').
        num_epochs (int): Number of training epochs.

    Returns:
        classifier: The trained classifier.
        train_losses: List of average training losses per epoch.
        valid_losses: List of average validation losses per epoch.
    """
    classifier.to(device)
    model.to(device)

    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        classifier.train()
        total_loss = 0.0
        total_samples = 0
        
        # trainig loop
        for batch in train_loader:
            latent_rep, labels = get_latents_and_labels_from_batch(batch, model, device)
            labels = labels[:, 0]  # using the first static feature as the label

            latent_tensor = torch.tensor(latent_rep, device=device)
            labels_tensor = torch.tensor(labels, device=device).long()

            optimizer.zero_grad()
            outputs = classifier(latent_tensor)  # outputs shape: (batch_size, num_classes)
            loss = criterion(outputs, labels_tensor)
            loss.backward()
            optimizer.step()

            batch_size = latent_tensor.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        epoch_train_loss = total_loss / total_samples
        train_losses.append(epoch_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_train_loss:.4f}")

        # ^validation loop
        if valid_loader is not None:
            classifier.eval()
            valid_loss_total = 0.0
            valid_samples = 0
            with torch.no_grad():
                for batch in valid_loader:
                    latent_rep, labels = get_latents_and_labels_from_batch(batch, model, device)
                    labels = labels[:, 0]  # using the first static feature as the label

                    latent_tensor = torch.tensor(latent_rep, device=device)
                    labels_tensor = torch.tensor(labels, device=device).long()

                    outputs = classifier(latent_tensor)
                    loss = criterion(outputs, labels_tensor)

                    valid_loss_total += loss.item() * latent_tensor.size(0)
                    valid_samples += latent_tensor.size(0)

            epoch_valid_loss = valid_loss_total / valid_samples
            valid_losses.append(epoch_valid_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {epoch_valid_loss:.4f}")

    return classifier, train_losses, valid_losses

def get_latents_and_labels_from_dataloader(model, data_loader, device):
    """
    Iterates over the dataloader and extracts latent representations (from the encoder)
    as well as the true labels (assumed to be in the "static_categorical_features" key).

    Args:
        model: Your TimeSeriesTransformerForPrediction model.
        data_loader: The training dataloader.
        device: The torch device to run inference on.
    
    Returns:
        latent_representations: List of numpy arrays, one per batch,
                                each of shape (batch_size, context_length, hidden_size).
        labels: List of numpy arrays, one per batch, containing the corresponding true labels.
    """
    latent_representations = []
    labels = []
    # Get the encoder from the underlying model.
    encoder = model.get_encoder()
    
    for batch in data_loader:
        # Move the necessary fields to the device.
        past_values = batch["past_values"].to(device)
        past_time_features = batch["past_time_features"].to(device)
        past_observed_mask = batch["past_observed_mask"].to(device)
        
        # Extract the true labels from static_categorical_features
        static_cat = None
        if model.config.num_static_categorical_features > 0 and "static_categorical_features" in batch:
            static_cat = batch["static_categorical_features"].to(device)
        
        static_real = None
        if model.config.num_static_real_features > 0 and "static_real_features" in batch:
            static_real = batch["static_real_features"].to(device)
        
        # Create unified transformer inputs
        transformer_inputs, _, _, _ = model.model.create_network_inputs(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            static_categorical_features=static_cat,
            static_real_features=static_real,
            future_values=None,       # We're not forecasting here.
            future_time_features=None
        )
        
        # The encoder only processes the first context_length time steps.
        enc_input = transformer_inputs[:, : model.config.context_length, ...]
        
        # Run the encoder to obtain hidden states.
        encoder_outputs = encoder(
            inputs_embeds=enc_input,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Extract the final hidden latent representation from the encoder.
        latent_rep = encoder_outputs.last_hidden_state  # shape: (batch_size, context_length, hidden_size)
        latent_representations.append(latent_rep.cpu().detach().numpy())
        
        # Also collect the true labels from static_categorical_features
        if static_cat is not None:
            labels.append(static_cat.cpu().detach().numpy())
    
    latent_representations = np.vstack(latent_representations)
    labels = np.vstack(labels)
    
    return latent_representations, labels

def get_latents_and_labels_from_batch(batch, model, device):
    """
    Process a single batch to extract the latent representations from the encoder and
    the true labels from the 'static_categorical_features' key.

    Args:
        batch (dict): A dictionary containing the batch data.
        model: Your TimeSeriesTransformerForPrediction model.
        device: The torch device (e.g., 'cuda' or 'cpu').

    Returns:
        latent_rep (np.array): Numpy array of shape (batch_size, context_length, hidden_size)
                               representing the latent encoder outputs.
        labels (np.array or None): Numpy array of true labels if available, else None.
    """
    past_values = batch["past_values"].to(device)
    past_time_features = batch["past_time_features"].to(device)
    past_observed_mask = batch["past_observed_mask"].to(device)

    static_cat = None
    if model.config.num_static_categorical_features > 0 and "static_categorical_features" in batch:
        static_cat = batch["static_categorical_features"].to(device)
    
    static_real = None
    if model.config.num_static_real_features > 0 and "static_real_features" in batch:
        static_real = batch["static_real_features"].to(device)
    
    transformer_inputs, _, _, _ = model.model.create_network_inputs(
        past_values=past_values,
        past_time_features=past_time_features,
        past_observed_mask=past_observed_mask,
        static_categorical_features=static_cat,
        static_real_features=static_real,
        future_values=None,       # We're not forecasting here
        future_time_features=None
    )
    
    enc_input = transformer_inputs[:, :model.config.context_length, ...]
    encoder = model.get_encoder()
    encoder_outputs = encoder(
        inputs_embeds=enc_input,
        output_hidden_states=True,
        return_dict=True,
    )
    
    latent_rep = encoder_outputs.last_hidden_state.cpu().detach().numpy()
    labels = None
    if static_cat is not None:
        labels = static_cat.cpu().detach().numpy()
    
    return latent_rep, labels
