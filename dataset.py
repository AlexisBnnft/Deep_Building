import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

INPUT_WINDOW = 7 * 24 # 1 week of hourly data
OUTPUT_WINDOW = 2 * 24 # 2 day of hourly data
STRIDE = 24  # Move forward by 1 day

class TerminalLoadDataset(Dataset):
    def __init__(self, terminal_loads_df, weather_df, 
                 input_window=INPUT_WINDOW,  
                 output_window=OUTPUT_WINDOW,  
                 stride=STRIDE):        
        
        # Ensure continuous hourly timestamps
        self.terminal_loads = terminal_loads_df.copy()
        self.weather = weather_df.copy()
        
        # Get the date range
        start_date = max(self.terminal_loads.index.min(), self.weather.index.min())
        end_date = min(self.terminal_loads.index.max(), self.weather.index.max())
        
        # Create continuous hourly index
        full_index = pd.date_range(start=start_date, end=end_date, freq='h')
        
        # Reindex both dataframes to ensure all timestamps exist
        self.terminal_loads = self.terminal_loads.reindex(full_index)
        self.weather = self.weather.reindex(full_index)
        
        # Drop any rows where either dataframe has missing data
        valid_idx = ~(self.terminal_loads.isna().any(axis=1) | self.weather.isna().any(axis=1))
        self.terminal_loads = self.terminal_loads[valid_idx]
        self.weather = self.weather[valid_idx]
        
        # Standardize the data
        self.load_scaler = StandardScaler()
        self.weather_scaler = StandardScaler()
        
        self.normalized_loads = pd.DataFrame(
            self.load_scaler.fit_transform(self.terminal_loads),
            columns=self.terminal_loads.columns,
            index=self.terminal_loads.index
        )
        
        self.normalized_weather = pd.DataFrame(
            self.weather_scaler.fit_transform(self.weather),
            columns=self.weather.columns,
            index=self.weather.index
        )
        
        # Create sequences
        self.sequences = []
        total_length = input_window + output_window
        
        # Only create sequences if we have enough continuous data
        if len(self.normalized_loads) >= total_length:
            for i in range(0, len(self.normalized_loads) - total_length + 1, stride):
                # Get sequences
                input_loads = self.normalized_loads.iloc[i:i+input_window].values
                input_weather = self.normalized_weather.iloc[i:i+input_window].values
                target_loads = self.normalized_loads.iloc[i+input_window:i+total_length].values
                target_weather = self.normalized_weather.iloc[i+input_window:i+total_length].values
                
                # Only add if sequences are complete
                if (input_loads.shape[0] == input_window and 
                    target_loads.shape[0] == output_window):
                    self.sequences.append((
                        (input_loads, input_weather),
                        (target_loads, target_weather)
                    ))
        
        print(f"Created {len(self.sequences)} valid sequences")
        if len(self.sequences) > 0:
            print(f"First sequence shapes:")
            print(f"Input loads: {self.sequences[0][0][0].shape}")
            print(f"Input weather: {self.sequences[0][0][1].shape}")
            print(f"Target loads: {self.sequences[0][1][0].shape}")
            print(f"Target weather: {self.sequences[0][1][1].shape}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        (input_loads, input_weather), (target_loads, target_weather) = self.sequences[idx]
        return (
            (torch.FloatTensor(input_loads), torch.FloatTensor(input_weather)),
            (torch.FloatTensor(target_loads), torch.FloatTensor(target_weather))
        )