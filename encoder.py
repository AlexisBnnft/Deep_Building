

import torch
import torch.nn as nn
from typing import Tuple


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode the input sequence.
        
        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, sequence_length, input_size)
        
        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - encoder_outputs (torch.Tensor): Encoded sequence of shape (batch_size, sequence_length, hidden_size)
                - (h_n, c_n) (Tuple[torch.Tensor, torch.Tensor]): Final hidden and cell states of shape (num_layers, batch_size, hidden_size)
        """
        return self.lstm(x)

class Decoder(nn.Module):
    def __init__(self, output_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, encoder_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Decode the output sequence.
        
        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, 1, output_size)
            encoder_state (Tuple[torch.Tensor, torch.Tensor]): Final hidden and cell states from the encoder
        
        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - output (torch.Tensor): Decoded output of shape (batch_size, 1, output_size)
                - (h_n, c_n) (Tuple[torch.Tensor, torch.Tensor]): Updated hidden and cell states of shape (num_layers, batch_size, hidden_size)
        """
        output, state = self.lstm(x, encoder_state)
        output = self.fc(output)
        return output, state