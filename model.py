
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from dataset import TerminalLoadDataset, INPUT_WINDOW, OUTPUT_WINDOW
from encoder import Encoder, Decoder
import json

class TerminalLoadPredictor(nn.Module):
    def __init__(self, n_zones, n_weather_features, hidden_size=64, num_layers=2, p_dropout=0, output_window=OUTPUT_WINDOW):
        super().__init__()
        
        self.n_zones = n_zones
        self.n_weather_features = n_weather_features
        
        # LSTM for processing terminal loads
        self.past_lstm = nn.LSTM(
            input_size=n_zones + n_weather_features,  # combined input size for terminal load and weather
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # LSTM (or FC layers) for future weather data
        self.future_weather_lstm = nn.LSTM(
            input_size=n_weather_features,
            hidden_size=hidden_size,
            num_layers=num_layers, 
            batch_first=True
        )
                
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # combine past LSTM and future weather LSTM
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_size, n_zones)  # predict terminal load for each zone at each future timestep
        )
        
        self.output_window = output_window
        
    def forward(self, past_tload_weather, future_weather):
        # Process loads
        
        _, (past_hidden, _) = self.past_lstm(past_tload_weather)  # (num_layers, batch_size, hidden_size)
        past_hidden = past_hidden[-1]  # take the hidden state of the last layer

        # Process weather

        future_out, _ = self.future_weather_lstm(future_weather)  # (batch_size, future_seq_len, hidden_size)
        
        # Initialize output list for the full prediction window
        predictions = []

        for t in range(self.output_window):
            # Concatenate hidden state from past data with future weather at current time step
            combined = torch.cat([past_hidden, future_out[:, t, :]], dim=1)  # (batch_size, hidden_size * 2)
            
            # Generate prediction for this time step
            prediction = self.fc(combined)  # (batch_size, n_zones)
            predictions.append(prediction)
        
        # Stack predictions to form the final sequence
        predictions = torch.stack(predictions, dim=1)  # (batch_size, output_window, n_zones)
        
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
        
        for (input_loads_weather, input_weather), target_loads in train_loader:
            # Move data to device
            input_loads_weather = input_loads_weather.to(device)
            input_weather = input_weather.to(device)
            target_loads = target_loads.to(device)

            # Forward pass
            predictions = model(input_loads_weather, input_weather)
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
            for (input_loads_weather, input_weather), target_loads in val_loader:
                input_loads_weather = input_loads_weather.to(device)
                input_weather = input_weather.to(device)
                target_loads = target_loads.to(device)
                
                predictions = model(input_loads_weather, input_weather)
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



def calculate_errors(model, dataset, load_scaler):
    """
    Calculate the L2 error for each prediction in the dataset.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    all_predictions = []
    all_targets = []
    all_errors = []
    
    with torch.no_grad():
        for (input_loads, input_weather), target_loads in dataset:
            input_loads = input_loads.unsqueeze(0).to(device)
            input_weather = input_weather.unsqueeze(0).to(device)
            target_loads = target_loads.unsqueeze(0).to(device)
            
            predictions = model(input_loads, input_weather)
            #print(predictions)
            #print(predictions.shape)    
            
            predictions = predictions.squeeze(0).cpu().numpy()
            target_loads = target_loads.squeeze(0).cpu().numpy()

            #print(predictions)
            #print(predictions.shape)    
            
            predictions = load_scaler.inverse_transform(predictions)
            target_loads = load_scaler.inverse_transform(target_loads)
            errors = np.sum(np.power(predictions - target_loads, 2), axis=0)

            all_predictions.append(predictions)
            all_targets.append(target_loads) 
            all_errors.append(errors)
            
    
    #errors = all_predictions - all_targets
    return np.array(all_errors), np.array(all_predictions), np.array(all_targets)