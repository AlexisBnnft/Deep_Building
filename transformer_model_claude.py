import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, List
import random
import math

# Constants
WINDOW_SIZE = 168  # 1 week of hourly data
FORECAST_HORIZON = 24  # 1 day ahead forecast
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
DEVICE = torch.device('mps' if torch.cuda.is_available() else 'cpu')

class TimeSeriesDataset(Dataset):
    def __init__(self, df_a: pd.DataFrame, df_b: pd.DataFrame, window_size: int, 
                 forecast_horizon: int, train: bool = True, scaler_a: StandardScaler = None, 
                 scaler_b: StandardScaler = None):
        """
        Parameters:
        -----------
        df_a : DataFrame
            DataFrame containing type A time series
        df_b : DataFrame
            DataFrame containing type B time series
        window_size : int
            Size of the historical window
        forecast_horizon : int
            Number of future time steps to predict
        train : bool
            Whether this is the training dataset
        scaler_a : StandardScaler, optional
            Pre-fitted scaler for A series (required if train=False)
        scaler_b : StandardScaler, optional
            Pre-fitted scaler for B series (required if train=False)
        """
        if len(df_a) != len(df_b):
            raise ValueError(f"Length mismatch: df_a has {len(df_a)} samples, df_b has {len(df_b)} samples")
            
        if len(df_a) < window_size + forecast_horizon:
            raise ValueError(f"Not enough samples. Need at least {window_size + forecast_horizon}, but got {len(df_a)}")
            
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        
        if train:
            self.scaler_a = StandardScaler()
            self.scaler_b = StandardScaler()
            self.scaled_a = self.scaler_a.fit_transform(df_a)
            self.scaled_b = self.scaler_b.fit_transform(df_b)
        else:
            if scaler_a is None or scaler_b is None:
                raise ValueError("Pre-fitted scalers must be provided when train=False")
            self.scaler_a = scaler_a
            self.scaler_b = scaler_b
            self.scaled_a = self.scaler_a.transform(df_a)
            self.scaled_b = self.scaler_b.transform(df_b)
            
        self.scaled_a = torch.FloatTensor(self.scaled_a)
        self.scaled_b = torch.FloatTensor(self.scaled_b)
        
        # Create sequences
        self.sequences = []
        for i in range(len(df_a) - window_size - forecast_horizon + 1):
            a_hist = self.scaled_a[i:i+window_size]
            b_hist = self.scaled_b[i:i+window_size]
            b_future = self.scaled_b[i+window_size:i+window_size+forecast_horizon]
            target = self.scaled_a[i+window_size:i+window_size+forecast_horizon]
            self.sequences.append((a_hist, b_hist, b_future, target))
            
        if len(self.sequences) == 0:
            raise ValueError(f"No sequences could be created from data of length {len(df_a)} "
                           f"with window_size={window_size} and forecast_horizon={forecast_horizon}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]
    
def create_datasets(df_a: pd.DataFrame, df_b: pd.DataFrame, window_size: int, 
                   forecast_horizon: int, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """
    Create train, validation, and test datasets with proper scaling and data split validation.
    """
    # First, check if we have enough data for the window size and forecast horizon
    min_required_length = window_size + forecast_horizon
    if len(df_a) < min_required_length:
        raise ValueError(f"Not enough data points. Need at least {min_required_length} points, but got {len(df_a)}")
    
    # Calculate split indices
    total_usable_length = len(df_a) - window_size - forecast_horizon + 1
    
    train_end = int(len(df_a) * train_ratio)
    val_end = int(len(df_a) * (train_ratio + val_ratio))
    
    # Ensure each split has enough data for at least one sequence
    min_split_size = window_size + forecast_horizon
    
    print(f"Data split sizes:")
    print(f"Training data: {train_end} samples")
    print(f"Validation data: {val_end - train_end} samples")
    print(f"Test data: {len(df_a) - val_end} samples")
    
    if train_end < min_split_size:
        raise ValueError(f"Training split too small. Need at least {min_split_size} samples, but got {train_end}")
    if (val_end - train_end) < min_split_size:
        raise ValueError(f"Validation split too small. Need at least {min_split_size} samples, but got {val_end - train_end}")
    if (len(df_a) - val_end) < min_split_size:
        raise ValueError(f"Test split too small. Need at least {min_split_size} samples, but got {len(df_a) - val_end}")
    
    # Create training dataset first to get fitted scalers
    train_dataset = TimeSeriesDataset(
        df_a[:train_end], 
        df_b[:train_end],
        window_size, 
        forecast_horizon, 
        train=True
    )
    
    # Create validation dataset using scalers from training
    val_dataset = TimeSeriesDataset(
        df_a[train_end:val_end],
        df_b[train_end:val_end],
        window_size,
        forecast_horizon,
        train=False,
        scaler_a=train_dataset.scaler_a,
        scaler_b=train_dataset.scaler_b
    )
    
    # Create test dataset using scalers from training
    test_dataset = TimeSeriesDataset(
        df_a[val_end:],
        df_b[val_end:],
        window_size,
        forecast_horizon,
        train=False,
        scaler_a=train_dataset.scaler_a,
        scaler_b=train_dataset.scaler_b
    )
    
    # Print sequence counts
    print(f"\nSequence counts:")
    print(f"Training sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")
    print(f"Test sequences: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim_a: int, input_dim_b: int, d_model: int = 256, 
                 nhead: int = 8, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Embedding layers for both types of series
        self.embed_a = nn.Linear(input_dim_a, d_model)
        self.embed_b = nn.Linear(input_dim_b, d_model)
        
        # Embedding for future B values
        self.embed_b_future = nn.Linear(input_dim_b, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder for historical data
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                  dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Additional transformer layer for combining historical and future data
        self.future_combination_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                                 nhead=nhead,
                                                                 dropout=dropout,
                                                                 batch_first=True)
        
        # Output layer
        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # Combine historical and future features
            nn.ReLU(),
            nn.Linear(d_model, input_dim_a)
        )
        
    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor, x_b_future: torch.Tensor) -> torch.Tensor:
        # Process historical data
        # x_a shape: [batch_size, window_size, input_dim_a]
        # x_b shape: [batch_size, window_size, input_dim_b]
        x_a_embedded = self.embed_a(x_a)
        x_b_embedded = self.embed_b(x_b)
        
        # Combine historical embeddings
        x_hist_combined = x_a_embedded + x_b_embedded
        x_hist_combined = self.pos_encoder(x_hist_combined)
        
        # Process historical data through transformer
        hist_features = self.transformer_encoder(x_hist_combined)
        hist_features = hist_features[:, -1, :]  # Take last time step features
        
        # Process future B data
        # x_b_future shape: [batch_size, forecast_horizon, input_dim_b]
        x_b_future_embedded = self.embed_b_future(x_b_future)
        x_b_future_encoded = self.pos_encoder(x_b_future_embedded)
        future_features = self.future_combination_layer(x_b_future_encoded)
        future_features = future_features.mean(dim=1)  # Pool future features
        
        # Combine historical and future features
        combined_features = torch.cat([hist_features, future_features], dim=-1)
        
        # Generate predictions
        predictions = self.decoder(combined_features)
        predictions = predictions.unsqueeze(1)  # Add time dimension back
        
        return predictions

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                num_epochs: int, device: torch.device) -> Tuple[List[float], List[float]]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            x_a, x_b, x_b_future, y = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            output = model(x_a, x_b, x_b_future)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x_a, x_b, x_b_future, y = [b.to(device) for b in batch]
                output = model(x_a, x_b, x_b_future)
                val_loss += criterion(output, y).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
                
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
    
    return train_losses, val_losses

def evaluate_model(model: nn.Module, test_loader: DataLoader, scaler_a: StandardScaler, 
                  device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_a, x_b, x_b_future, y = [b.to(device) for b in batch]
            output = model(x_a, x_b, x_b_future)
            
            # Convert to numpy and inverse transform
            pred = scaler_a.inverse_transform(output.cpu().numpy())
            actual = scaler_a.inverse_transform(y.cpu().numpy())
            
            predictions.extend(pred)
            actuals.extend(actual)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals)**2))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    return mae, rmse, mape

def plot_predictions(model: nn.Module, test_loader: DataLoader, scaler_a: StandardScaler, 
                    device: torch.device, num_samples: int = 3):
    model.eval()
    
    # Get random samples
    samples = random.sample(range(len(test_loader.dataset)), num_samples)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for idx, ax in zip(samples, axes):
        x_a, x_b, x_b_future, y = test_loader.dataset[idx]
        x_a, x_b, x_b_future, y = [t.unsqueeze(0).to(device) for t in [x_a, x_b, x_b_future, y]]
        
        with torch.no_grad():
            pred = model(x_a, x_b, x_b_future)
        
        # Inverse transform
        pred = scaler_a.inverse_transform(pred.cpu().numpy().squeeze())
        actual = scaler_a.inverse_transform(y.cpu().numpy().squeeze())
        
        # Plot
        ax.plot(actual, label='Actual', marker='o')
        ax.plot(pred, label='Predicted', marker='x')
        ax.set_title(f'Sample {idx}')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

# Example usage:
def main():
    # Load your data (df_tload and df_weather should be loaded here)
    # df_tload = pd.read_csv('path_to_tload_data.csv')
    # df_weather = pd.read_csv('path_to_weather_data.csv')
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        df_tload, 
        df_weather,
        WINDOW_SIZE, 
        FORECAST_HORIZON
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = TimeSeriesTransformer(
        input_dim_a=df_tload.shape[1],
        input_dim_b=df_weather.shape[1]
    ).to(DEVICE)
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, NUM_EPOCHS, DEVICE)
    
    # Evaluate model
    mae, rmse, mape = evaluate_model(model, test_loader, train_dataset.scaler_a, DEVICE)
    print(f'Test MAE: {mae:.2f}')
    print(f'Test RMSE: {rmse:.2f}')
    print(f'Test MAPE: {mape:.2f}%')
    
    # Plot some predictions
    plot_predictions(model, test_loader, train_dataset.scaler_a, DEVICE)

if __name__ == "__main__":
    main()