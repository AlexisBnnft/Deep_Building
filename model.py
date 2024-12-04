import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from dataset import TerminalLoadDataset, INPUT_WINDOW, OUTPUT_WINDOW
from encoder import Encoder, Decoder
import json

        
class TerminalLoadPredictor(nn.Module):
    def __init__(self, n_zones, n_weather_features, hidden_size=256, num_layers=2, p_dropout=0.2, output_window=OUTPUT_WINDOW, l2_lambda=0.0001):
        super().__init__()
        
        self.n_zones = n_zones
        self.n_weather_features = n_weather_features
        self.l2_lambda = l2_lambda
        
        # Add Layer Normalization
        self.layer_norm_past = nn.LayerNorm(n_zones + n_weather_features)
        self.layer_norm_weather = nn.LayerNorm(n_weather_features)
        
        # LSTM for processing terminal loads with increased dropout
        self.past_lstm = nn.LSTM(
            input_size=n_zones + n_weather_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=p_dropout if num_layers > 1 else 0
        )
        
        # LSTM for future weather data
        self.future_weather_lstm = nn.LSTM(
            input_size=n_weather_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=p_dropout if num_layers > 1 else 0
        )
        
        # Add Batch Normalization before FC layers
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        
        # Modified FC layers with dropout between them
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_size, n_zones)
        )
        
        self.output_window = output_window
        
    def forward(self, past_tload_weather, future_weather):
        # Apply Layer Normalization
        past_tload_weather = self.layer_norm_past(past_tload_weather)
        future_weather = self.layer_norm_weather(future_weather)
        
        _, (past_hidden, _) = self.past_lstm(past_tload_weather)
        past_hidden = past_hidden[-1]

        # Get the last terminal load
        last_tload = past_tload_weather[:, -1, :self.n_zones] # Get the last terminal load, size: (batch_size, n_zones)
        
        future_out, _ = self.future_weather_lstm(future_weather)
        
        predictions = []
        
        for t in range(self.output_window):
            if t < future_out.size(1):
                combined = torch.cat([past_hidden, future_out[:, t, :]], dim=1)
                # Apply batch normalization to combined features
                prediction = self.fc(combined)    
                predictions.append(prediction)
        
        predictions = torch.stack(predictions, dim=1)
        return predictions


def train_model(model, train_loader, val_loader, epochs=50, learning_rate=0.0001):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    print(f"Training on device: {device}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    def compute_l2_loss(model):
        """Calculate L2 regularization loss"""
        l2_loss = 0
        for param in model.parameters():
            l2_loss += torch.norm(param, p=2) ** 2
        return l2_loss
    
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
            
            # Combine losses
            l2_loss = compute_l2_loss(model)
            loss = criterion(predictions, target_loads) + model.l2_lambda * l2_loss
            
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


def load_all(model_path, data_path, model_class = TerminalLoadPredictor):
    # Load the other components
    data = joblib.load(data_path)
    
    train_losses = data['train_losses']
    val_losses = data['val_losses']
    train_dataset = data['train_dataset']
    val_dataset = data['val_dataset']
    train_loader = data['train_loader']
    val_loader = data['val_loader']

    # Instantiate the model with the correct arguments
    model = model_class(len(train_dataset.terminal_loads.columns), len(train_dataset.weather.columns))
    
    # Load the model state dictionary
    model.load_state_dict(torch.load(model_path))
    
    return model, train_losses, val_losses, train_dataset, val_dataset, train_loader, val_loader

def save_all(model, train_losses, val_losses, train_dataset, val_dataset, train_loader, val_loader, model_path, data_path):
    # Save the model state dictionary
    torch.save(model.state_dict(), model_path)
    
    # Save the other components
    data = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'train_loader': train_loader,
        'val_loader': val_loader
    }
    
    joblib.dump(data, data_path)

