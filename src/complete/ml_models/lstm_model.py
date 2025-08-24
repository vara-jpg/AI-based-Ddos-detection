#!/usr/bin/env python3
"""
LSTM Model for Low-Rate DDoS Detection
PyTorch implementation of bidirectional LSTM for network traffic analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class LSTMDDoSDetector(nn.Module):
    """
    Bidirectional LSTM model for DDoS attack detection in network traffic
    """
    
    def __init__(self, 
                 input_size: int = 20,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = True):
        """
        Initialize LSTM model
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
            output_size: Number of output classes (2 for binary classification)
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMDDoSDetector, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention mechanism
        self.attention = AttentionLayer(lstm_output_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, output_size)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 4)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            lengths: Actual sequence lengths for packed sequences
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0, c0 = self._init_hidden(batch_size, x.device)
        
        # LSTM forward pass
        if lengths is not None:
            # Pack padded sequences for variable length sequences
            x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            lstm_out_packed, (hn, cn) = self.lstm(x_packed, (h0, c0))
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out_packed, batch_first=True)
        else:
            lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Apply attention mechanism
        attended_output = self.attention(lstm_out, lengths)
        
        # Fully connected layers with regularization
        out = F.relu(self.bn1(self.fc1(attended_output)))
        out = self.dropout1(out)
        
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.dropout2(out)
        
        out = self.fc3(out)
        
        return out
    
    def _init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state"""
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(device)
        return h0, c0
    
    def predict_proba(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get prediction probabilities"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, lengths)
            probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def predict(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get predictions"""
        probabilities = self.predict_proba(x, lengths)
        return torch.argmax(probabilities, dim=1)


class AttentionLayer(nn.Module):
    """
    Attention mechanism for LSTM outputs
    """
    
    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention_weights = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, lstm_output: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply attention mechanism
        
        Args:
            lstm_output: LSTM output tensor (batch_size, seq_len, hidden_size)
            lengths: Actual sequence lengths
            
        Returns:
            Attention-weighted output (batch_size, hidden_size)
        """
        # Calculate attention scores
        attention_scores = self.attention_weights(lstm_output).squeeze(-1)  # (batch_size, seq_len)
        
        # Apply mask for variable length sequences
        if lengths is not None:
            mask = self._create_mask(lstm_output.size(0), lstm_output.size(1), lengths, lstm_output.device)
            attention_scores.masked_fill_(mask, -float('inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)
        
        # Apply attention weights to LSTM output
        attended_output = torch.sum(lstm_output * attention_weights.unsqueeze(-1), dim=1)  # (batch_size, hidden_size)
        
        return attended_output
    
    def _create_mask(self, batch_size: int, seq_len: int, lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Create mask for variable length sequences"""
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        for i, length in enumerate(lengths):
            if length < seq_len:
                mask[i, length:] = True
        return mask


class CNNLSTMModel(nn.Module):
    """
    Hybrid CNN-LSTM model for DDoS detection
    """
    
    def __init__(self, 
                 input_size: int = 20,
                 cnn_channels: int = 64,
                 kernel_size: int = 3,
                 lstm_hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 2,
                 dropout: float = 0.2):
        super(CNNLSTMModel, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(input_size, cnn_channels, kernel_size, padding=1)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout_cnn = nn.Dropout(dropout)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=cnn_channels * 2,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc = nn.Linear(lstm_hidden_size * 2, output_size)
        self.dropout_fc = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CNN-LSTM model
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor (batch_size, output_size)
        """
        # Transpose for CNN (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        
        # CNN layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout_cnn(x)
        
        # Transpose back for LSTM (batch_size, seq_len, channels)
        x = x.transpose(1, 2)
        
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        x = lstm_out[:, -1, :]
        
        # Fully connected layer
        x = self.dropout_fc(x)
        x = self.fc(x)
        
        return x


class AutoEncoder(nn.Module):
    """
    Autoencoder for anomaly detection in network traffic
    """
    
    def __init__(self, 
                 input_size: int = 20,
                 hidden_sizes: list = [128, 64, 32],
                 dropout: float = 0.2):
        super(AutoEncoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            encoder_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        hidden_sizes_reversed = list(reversed(hidden_sizes[:-1])) + [input_size]
        
        for hidden_size in hidden_sizes_reversed:
            decoder_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU() if hidden_size != input_size else nn.Identity(),
                nn.Dropout(dropout) if hidden_size != input_size else nn.Identity()
            ])
            prev_size = hidden_size
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input"""
        return self.encoder(x)
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode input"""
        return self.decoder(x)


if __name__ == "__main__":
    # Test the models
    batch_size, seq_len, input_size = 32, 10, 20
    
    # Test LSTM model
    model = LSTMDDoSDetector(input_size=input_size)
    x = torch.randn(batch_size, seq_len, input_size)
    output = model(x)
    print(f"LSTM Model Output Shape: {output.shape}")
    
    # Test CNN-LSTM model
    cnn_lstm_model = CNNLSTMModel(input_size=input_size)
    output = cnn_lstm_model(x)
    print(f"CNN-LSTM Model Output Shape: {output.shape}")
    
    # Test Autoencoder
    autoencoder = AutoEncoder(input_size=input_size)
    x_flat = torch.randn(batch_size, input_size)
    output = autoencoder(x_flat)
    print(f"Autoencoder Output Shape: {output.shape}")
