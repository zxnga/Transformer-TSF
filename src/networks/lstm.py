import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=24, num_layers=1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(self.device)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(self.device)
        return hidden_state, cell_state
    
    def forward(self, input_seq):
        batch_size = input_seq.size(0)
        hidden_cell = self.init_hidden(batch_size)
    
        lstm_out, _ = self.lstm(input_seq, hidden_cell)
        

        predictions = self.linear(lstm_out[:, -1, :])  # Use only the last LSTM output for prediction
        return predictions
