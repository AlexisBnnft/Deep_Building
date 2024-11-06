import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from typing import Tuple, List

class PredictionVisualizer:
    def __init__(self, model, dataset, load_scaler, zones_names=None):
        """
        Initialize the visualizer with a trained model and dataset.
        
        Args:
            model: Trained TerminalLoadPredictor model
            dataset: TerminalLoadDataset instance
            load_scaler: StandardScaler used to normalize the load data
            zones_names: List or Index of zone names (optional)
        """
        self.model = model
        self.dataset = dataset
        self.load_scaler = load_scaler
        # Convert zones_names to list if it's a pandas Index
        self.zones_names = list(zones_names) if zones_names is not None else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def plot_training_history(self, train_losses: List[float], val_losses: List[float]) -> None:
        """Plot training and validation loss history."""
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
        plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def get_predictions(self, sample_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions for a specific sample from the dataset.
        
        Returns:
            Tuple of (actual_values, predicted_values)
        """
        self.model.eval()
        with torch.no_grad():
            # Get sample data
            (input_loads, input_weather), (target_loads, _) = self.dataset[sample_idx]
            
            # Add batch dimension and move to device
            input_loads = input_loads.unsqueeze(0).to(self.device)
            input_weather = input_weather.unsqueeze(0).to(self.device)
            
            # Get predictions
            predictions = self.model(input_loads, input_weather)
            
            # Convert to numpy and inverse transform
            predictions = predictions.cpu().numpy().squeeze()
            actual = target_loads.numpy()
            
            # Inverse transform the data
            predictions = self.load_scaler.inverse_transform(predictions)
            actual = self.load_scaler.inverse_transform(actual)
            
            return actual, predictions

    def plot_predictions(self, sample_idx: int, zones_to_plot: List[int] = None) -> None:
        """
        Plot actual vs predicted values for specified zones.
        
        Args:
            sample_idx: Index of the sample to visualize
            zones_to_plot: List of zone indices to plot (default: first 3 zones)
        """
        actual, predictions = self.get_predictions(sample_idx)
        
        if zones_to_plot is None:
            zones_to_plot = list(range(min(3, actual.shape[1])))
        
        hours = np.arange(actual.shape[0])
        
        plt.figure(figsize=(15, 5 * len(zones_to_plot)))
        
        for i, zone_idx in enumerate(zones_to_plot):
            plt.subplot(len(zones_to_plot), 1, i + 1)
            
            zone_name = self.zones_names[zone_idx] if self.zones_names is not None else f"Zone {zone_idx}"
            
            # Plot actual values
            plt.plot(hours, actual[:, zone_idx], 
                    label='Actual', color='blue', alpha=0.7)
            
            # Plot predictions
            plt.plot(hours, predictions[:, zone_idx], 
                    label='Predicted', color='red', alpha=0.7, linestyle='--')
            
            plt.title(f'Load Predictions for {zone_name}')
            plt.xlabel('Hours')
            plt.ylabel('Load')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_error_distribution(self, sample_idx: int, zones_to_plot: List[int] = None) -> None:
        """
        Plot error distribution for predictions.
        
        Args:
            sample_idx: Index of the sample to visualize
            zones_to_plot: List of zone indices to plot (default: first 3 zones)
        """
        actual, predictions = self.get_predictions(sample_idx)
        
        if zones_to_plot is None:
            zones_to_plot = list(range(min(3, actual.shape[1])))
        
        plt.figure(figsize=(15, 5))
        
        errors = predictions - actual
        
        for zone_idx in zones_to_plot:
            zone_name = self.zones_names[zone_idx] if self.zones_names is not None else f"Zone {zone_idx}"
            sns.kdeplot(errors[:, zone_idx], label=zone_name)
        
        plt.title('Prediction Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def print_metrics(self, sample_idx: int, zones_to_plot: List[int] = None) -> None:
        """
        Print error metrics for predictions.
        
        Args:
            sample_idx: Index of the sample to visualize
            zones_to_plot: List of zone indices to plot (default: first 3 zones)
        """
        actual, predictions = self.get_predictions(sample_idx)
        
        if zones_to_plot is None:
            zones_to_plot = list(range(min(3, actual.shape[1])))
        
        print("\nPrediction Metrics:")
        print("-" * 50)
        
        for zone_idx in zones_to_plot:
            zone_name = self.zones_names[zone_idx] if self.zones_names is not None else f"Zone {zone_idx}"
            
            mape = mean_absolute_percentage_error(
                actual[:, zone_idx], 
                predictions[:, zone_idx]
            ) * 100
            
            rmse = np.sqrt(mean_squared_error(
                actual[:, zone_idx], 
                predictions[:, zone_idx]
            ))
            
            print(f"\n{zone_name}:")
            print(f"MAPE: {mape:.2f}%")
            print(f"RMSE: {rmse:.2f}")

def visualize_results(model, dataset, load_scaler, train_losses, val_losses, 
                     zones_names=None, sample_idx=0, zones_to_plot=None):
    """
    Wrapper function to create visualizer and generate all plots.
    
    Args:
        model: Trained TerminalLoadPredictor model
        dataset: TerminalLoadDataset instance
        load_scaler: StandardScaler used to normalize the load data
        train_losses: List of training losses
        val_losses: List of validation losses
        zones_names: List or Index of zone names (optional)
        sample_idx: Index of the sample to visualize
        zones_to_plot: List of zone indices to plot
    """
    visualizer = PredictionVisualizer(model, dataset, load_scaler, zones_names)
    
    # Plot training history
    visualizer.plot_training_history(train_losses, val_losses)
    
    # Plot predictions
    visualizer.plot_predictions(sample_idx, zones_to_plot)
    
    # Plot error distribution
    visualizer.plot_error_distribution(sample_idx, zones_to_plot)
    
    # Print metrics
    visualizer.print_metrics(sample_idx, zones_to_plot)