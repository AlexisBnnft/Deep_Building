
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from dataset import TerminalLoadDataset, INPUT_WINDOW, OUTPUT_WINDOW
from encoder import Encoder, Decoder

class TerminalLoadPredictor(nn.Module):
    def __init__(self, n_zones, n_weather_features, hidden_size=64, num_layers=2, p_dropout=0):
        super().__init__()
        
        self.n_zones = n_zones
        self.n_weather_features = n_weather_features
        
        # LSTM for processing terminal loads
        self.loads_lstm = nn.LSTM(
            input_size=n_zones,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # LSTM for processing weather data
        self.weather_lstm = nn.LSTM(
            input_size=n_weather_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(p_dropout),  # Add dropout for regularization
            nn.Linear(hidden_size, n_zones * OUTPUT_WINDOW)  # Full week prediction
        )
        
    def forward(self, loads, weather):
        # Process loads
        loads_out, _ = self.loads_lstm(loads)
        
        # Process weather
        print(loads.shape)
        weather_out, _ = self.weather_lstm(weather)
        print(weather.shape)
        # Combine features
        combined = torch.cat([loads_out[:, -1, :], weather_out[:, -1, :]], dim=1)
        
        # Generate predictions
        predictions = self.fc(combined)
        predictions = predictions.view(-1, OUTPUT_WINDOW, self.n_zones)
        
        return predictions


def train_model(model, train_loader, val_loader, epochs=50, learning_rate=0.001):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    print(f"Training on device: {device}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        batch_count = 0
        
        for (input_loads, input_weather), (target_loads, _) in train_loader:
            # Move data to device
            input_loads = input_loads.to(device)
            input_weather = input_weather.to(device)
            target_loads = target_loads.to(device)

            # Forward pass
            predictions = model(input_loads, input_weather)
            loss = criterion(predictions, target_loads)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = total_train_loss / batch_count if batch_count > 0 else float('inf')
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for (input_loads, input_weather), (target_loads, _) in val_loader:
                input_loads = input_loads.to(device)
                input_weather = input_weather.to(device)
                target_loads = target_loads.to(device)
                
                predictions = model(input_loads, input_weather)
                loss = criterion(predictions, target_loads)
                
                total_val_loss += loss.item()
                val_batch_count += 1
        
        avg_val_loss = total_val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses



def get_trained_model(terminal_loads_df, weather_df, n_epochs=50, p_dropout=0):
    # First, check if we have enough data
    print("Checking data availability...")
    print(f"Terminal loads date range: {terminal_loads_df.index.min()} to {terminal_loads_df.index.max()}")
    print(f"Weather data date range: {weather_df.index.min()} to {weather_df.index.max()}")
    
    # Calculate the split date dynamically to ensure we have enough data for both sets
    total_days = (terminal_loads_df.index.max() - terminal_loads_df.index.min()).days
    split_days = int(total_days * 0.8)  # 80% for training
    split_date = terminal_loads_df.index.min() + pd.Timedelta(days=split_days)
    
    print(f"\nUsing split date: {split_date}")
    
    # Split data into train and validation sets
    train_loads = terminal_loads_df[terminal_loads_df.index < split_date]
    train_weather = weather_df[weather_df.index < split_date]
    
    val_loads = terminal_loads_df[terminal_loads_df.index >= split_date]
    val_weather = weather_df[weather_df.index >= split_date]
    
    # Check if we have enough data in each set
    min_required_hours = INPUT_WINDOW + OUTPUT_WINDOW  # Input window + output window
    
    print("\nChecking data sizes:")
    print(f"Training set hours: {len(train_loads)}")
    print(f"Validation set hours: {len(val_loads)}")
    print(f"Minimum required hours: {min_required_hours}")
    
    if len(train_loads) < min_required_hours or len(val_loads) < min_required_hours:
        raise ValueError(
            f"Insufficient data for training/validation. Need at least {min_required_hours} "
            f"continuous hours in each set. Got {len(train_loads)} training hours and "
            f"{len(val_loads)} validation hours."
        )
    
    print("\nCreating datasets...")
    train_dataset = TerminalLoadDataset(train_loads, train_weather)
    val_dataset = TerminalLoadDataset(val_loads, val_weather)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(
            f"No valid sequences created. Got {len(train_dataset)} training sequences "
            f"and {len(val_dataset)} validation sequences. Check for data gaps or "
            "missing values in your input data."
        )
    
    print("\nCreating dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print("\nInitializing model...")
    model = TerminalLoadPredictor(
        n_zones=len(terminal_loads_df.columns),
        n_weather_features=len(weather_df.columns),
        p_dropout=p_dropout
    )
    
    print("\nStarting training...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=n_epochs)
    
    return model, train_losses, val_losses, train_dataset, val_dataset, train_loader, val_loader
