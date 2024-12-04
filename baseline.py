import torch
import torch.nn as nn
import joblib
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from dataset import TerminalLoadDataset, INPUT_WINDOW, OUTPUT_WINDOW  # Ensure these are correctly defined in your project

# BaselinePredictor Class
class BaselinePredictor(nn.Module):
    """
    A simple baseline predictor that outputs the mean terminal loads from the first day of the input window
    as predictions for the output window. Assumes that future patterns are similar to those of the first day.
    """
    def __init__(self, n_zones, output_window=OUTPUT_WINDOW):
        super().__init__()
        self.n_zones = n_zones
        self.output_window = output_window

    def forward(self, past_tload_weather, future_weather=None):
        """
        Forward pass for the BaselinePredictor.

        Args:
            past_tload_weather (Tensor): Tensor of shape (batch_size, input_seq_len, n_zones + n_weather_features)
            future_weather (Tensor, optional): Tensor of shape (batch_size, output_seq_len, n_weather_features)
                                                Not used in baseline predictions.

        Returns:
            Tensor: Predictions of shape (batch_size, output_window, n_zones)
        """
        # Determine the number of time steps that make up a day
        # For example, if data is hourly and a day has 24 hours
        time_steps_per_day = 24  # Adjust based on your data frequency

        # Extract the first day's terminal loads
        first_day_loads = past_tload_weather[:, :time_steps_per_day, :self.n_zones]  # Shape: (batch_size, time_steps_per_day, n_zones)

        # Compute the mean load of the first day as the prediction
        day_mean_load = first_day_loads.mean(dim=1)  # Shape: (batch_size, n_zones)

        # Repeat the day_mean_load for the output window
        predictions = day_mean_load.unsqueeze(1).repeat(1, self.output_window, 1)  # Shape: (batch_size, output_window, n_zones)

        return predictions

# Function to Train TerminalLoadPredictor
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

# Function to Train BaselinePredictor (No Training Needed)
def get_baseline_model(terminal_loads_df, weather_df, n_epochs=50, p_dropout=0.2):
    """
    Prepares the BaselinePredictor model by handling data splitting and dataset creation.
    Although the BaselinePredictor does not require training, this function maintains consistency
    with the TerminalLoadPredictor's get_trained_model function.

    Args:
        terminal_loads_df (pd.DataFrame): DataFrame containing terminal load data with datetime index.
        weather_df (pd.DataFrame): DataFrame containing weather data with datetime index.
        n_epochs (int, optional): Number of epochs for training. Not used in BaselinePredictor.
        p_dropout (float, optional): Dropout probability. Not used in BaselinePredictor.

    Returns:
        Tuple:
            - model (BaselinePredictor): The instantiated baseline model.
            - train_losses (List[float]): Empty list as baseline does not train.
            - val_losses (List[float]): Empty list as baseline does not train.
            - train_dataset (TerminalLoadDataset): Training dataset.
            - val_dataset (TerminalLoadDataset): Validation dataset.
            - train_loader (DataLoader): DataLoader for training data.
            - val_loader (DataLoader): DataLoader for validation data.
    """
    print("Preparing BaselinePredictor model...")

    # Check data availability
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

    print("\nInitializing BaselinePredictor model...")
    model = BaselinePredictor(
        n_zones=len(terminal_loads_df.columns),
        output_window=OUTPUT_WINDOW
    )

    print("BaselinePredictor model is ready.")
    # Since BaselinePredictor does not require training, we return empty loss lists
    return model, [], [], train_dataset, val_dataset, train_loader, val_loader


def calculate_baseline_errors(model, dataset, load_scaler):
    """
    Calculate the L2 error for each prediction in the dataset using the BaselinePredictor.

    Args:
        model (BaselinePredictor): The baseline model.
        dataset (TerminalLoadDataset): The dataset to evaluate.
        load_scaler (StandardScaler): Scaler used to inverse transform the load data.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of errors, predictions, and targets.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    all_predictions = []
    all_targets = []
    all_errors = []
    
    with torch.no_grad():
        for (input_loads, input_weather), target_loads in dataset:
            input_loads = input_loads.unsqueeze(0).to(device)  # Shape: (1, input_seq_len, n_zones + n_weather_features)
            # input_weather is not used in baseline
            target_loads = target_loads.unsqueeze(0).to(device)  # Shape: (1, output_window, n_zones)
            
            predictions = model(input_loads)
            
            predictions = predictions.squeeze(0).cpu().numpy()
            target_loads = target_loads.squeeze(0).cpu().numpy()
            
            # Inverse transform the scaled data
            predictions = load_scaler.inverse_transform(predictions)
            target_loads = load_scaler.inverse_transform(target_loads)
            
            # Calculate L2 error for each zone
            errors = np.sum(np.power(predictions - target_loads, 2), axis=0)
            
            all_predictions.append(predictions)
            all_targets.append(target_loads)
            all_errors.append(errors)
    
    return np.array(all_errors), np.array(all_predictions), np.array(all_targets)


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



def load_all(model_path, data_path, model_class = BaselinePredictor):
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
