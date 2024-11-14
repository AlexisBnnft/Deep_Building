import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd
from dataset import TerminalLoadDataset, INPUT_WINDOW, OUTPUT_WINDOW    
from torch.utils.data import DataLoader


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]

class TransformerLoadPredictor(nn.Module):
    def __init__(self, n_zones, n_weather_features, d_model=64, nhead=4, 
                 num_layers=1, dim_feedforward=256, dropout=0.1):
        super().__init__()
        
        self.n_zones = n_zones
        self.n_weather_features = n_weather_features
        
        # Input processing
        self.past_proj = nn.Sequential(
            nn.Linear(n_zones + n_weather_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.weather_proj = nn.Sequential(
            nn.Linear(n_weather_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Simple positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model) * 0.02)
        
        # Simplified Transformer with only encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, n_zones)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, past_tload_weather, future_weather):
        # Project inputs
        past_embedded = self.past_proj(past_tload_weather)
        future_embedded = self.weather_proj(future_weather)
        
        # Combine sequences
        seq_len_past = past_embedded.size(1)
        seq_len_future = future_embedded.size(1)
        
        # Add positional encoding
        past_embedded = past_embedded + self.pos_encoder[:, :seq_len_past, :]
        future_embedded = future_embedded + self.pos_encoder[:, :seq_len_future, :]
        
        # Concatenate past and future
        combined = torch.cat([past_embedded, future_embedded], dim=1)
        
        # Transform
        output = self.transformer(combined)
        
        # Only take the future timesteps
        future_output = output[:, -seq_len_future:, :]
        
        # Project to predictions
        predictions = self.output_proj(future_output)
        
        return predictions

def train_transformer_model(model, train_loader, val_loader, epochs=50, learning_rate=0.001, 
                          warmup_steps=4000, ):
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    # Use AdaMod optimizer (more stable than Adam)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0005,
        betas=(0.9, 0.999),
        weight_decay=0.05
    )
    
    # Cyclical learning rate
    def cyclic_lr(step, step_size=500, min_lr=1e-6, max_lr=0.001):
        cycle = math.floor(1 + step / (2 * step_size))
        x = abs(step / step_size - 2 * cycle + 1)
        return min_lr + (max_lr - min_lr) * max(0, (1 - x))
    
    # Robust loss function
    criterion = nn.SmoothL1Loss(beta=0.1)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    total_steps = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        batch_count = 0
        
        for (input_loads_weather, input_weather), target_loads in train_loader:
            input_loads_weather = input_loads_weather.to(device)
            input_weather = input_weather.to(device)
            target_loads = target_loads.to(device)
            
            # Update learning rate
            lr = cyclic_lr(total_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            optimizer.zero_grad()
            
            predictions = model(input_loads_weather, input_weather)
            loss = criterion(predictions, target_loads)
            #loss = criterion(predictions, target_loads) * ((target_loads).std()**2+1)*100
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            batch_count += 1
            total_steps += 1
        
        avg_train_loss = total_train_loss / batch_count
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
        
        avg_val_loss = total_val_loss / val_batch_count
        val_losses.append(avg_val_loss)
        
        
        ## Early stopping check
        #if avg_val_loss < best_val_loss:
        #    best_val_loss = avg_val_loss
        #    patience_counter = 0
        #    # Save best model
        #    torch.save(model.state_dict(), 'best_model.pth')
        #else:
        #    patience_counter += 1
        #
        #if patience_counter >= patience:
        #    print(f'Early stopping triggered after epoch {epoch+1}')
        #    # Load best model
        #    model.load_state_dict(torch.load('best_model.pth'))
        #    break
        #
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, LR: {lr:.6f}')
    
    return train_losses, val_losses


def get_trained_model(terminal_loads_df, weather_df, n_epochs=50, p_dropout=0.1):
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
    model = TransformerLoadPredictor(
        n_zones=len(terminal_loads_df.columns),
        n_weather_features=len(weather_df.columns),
        d_model=64,
        nhead=4,
        num_layers=1
    )
    print("\nStarting training...")
    train_losses, val_losses = train_transformer_model(
                            model,
                            train_loader,
                            val_loader,
                            epochs=100,
                            learning_rate=0.0001,
                            warmup_steps=1000
                        )
    
    return model, train_losses, val_losses, train_dataset, val_dataset, train_loader, val_loader
