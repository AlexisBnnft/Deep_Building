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
            (input_loads, input_weather), (target_loads) = self.dataset[sample_idx]
            
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
            plt.ylim(-100,100)
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
                     zones_names=None, sample_idx=0, zones_to_plot=None, n_samples=None):
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
    if n_samples is not None:
        zones_to_plot = np.random.choice(len(dataset.terminal_loads.columns), n_samples, replace=False)
    visualizer = PredictionVisualizer(model, dataset, load_scaler, zones_names)
    
    # Plot training history
    visualizer.plot_training_history(train_losses, val_losses)
    
    # Plot predictions
    visualizer.plot_predictions(sample_idx, zones_to_plot)
    
    # Plot error distribution
    visualizer.plot_error_distribution(sample_idx, zones_to_plot)
    
    # Print metrics
    visualizer.print_metrics(sample_idx, zones_to_plot)


def compare_results(models, train_losses, val_losses, model_names=None):
    """
    Compare results for multiple models.
    
    Args:
        models: List of trained models
        train_losses: List of training losses for each model
        val_losses: List of validation losses for each model
    """
    for i, model in enumerate(models):
        print(f"Model {i + 1}")
        print("-" * 50)
        print(model)
        print("\n")
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    for i, model in enumerate(models):
        plt.plot(train_losses[i], label=f'{model_names[i]} Training Loss', alpha=0.7)
        plt.plot(val_losses[i], label=f'{model_names[i]} Validation Loss', alpha=0.7)
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_bad_samples_predictions(all_errors, all_predictions, all_targets, N_samples=5, N_zones=3, good_zones = False):
    mean_absolute_errors = np.mean(np.abs(all_errors), axis=0)

    # Get the sorted indices based on the mean absolute error
    sorted_indices = np.argsort(mean_absolute_errors)

    # Sort the columns of all_errors based on the sorted indices
    sorted_all_errors = all_errors[:, sorted_indices]

    if good_zones:
        most_error_indices = sorted_indices[:N_zones] # Get the indices of the top N_zones zones with the least errors
    else:
        most_error_indices = sorted_indices[-N_zones:]  # Get the indices of the top N_zones zones with the most errors

    # Plot some samples for the zones with the most errors in nested subplots
    num_samples_to_plot = N_samples
    fig, axs = plt.subplots(len(most_error_indices), 1, figsize=(15, 3 * len(most_error_indices)), sharex=True)

    sample_ids = np.random.randint(0, all_predictions.shape[0], num_samples_to_plot)  # Randomly select some samples to plot
    print(sample_ids)
    for i, idx in enumerate(most_error_indices):
        # Create nested subplots for each sample within the current zone subplot
        subfig = axs[i].inset_axes([0, 0, 1, 1])
        subfig.set_title(f'Predictions vs True Values for Zone {idx}')
        subfig.set_xlabel('Samples')
        subfig.set_ylabel('Terminal Load')
        subfig.set_ylim(-100, 100)
    

        for j, sample_id in enumerate(sample_ids):
            ax = subfig.inset_axes([j / num_samples_to_plot, 0, 1 / num_samples_to_plot, 1])
            ax.plot(all_predictions[sample_id, :, idx], label='Predicted Value', color='red', alpha=0.5)
            ax.plot(all_targets[sample_id, :, idx], label='True Value', color='blue', alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim(-100, 100)
            if j == 0:
                ax.legend()
            else:
                ax.set_yticklabels([])

    # Remove y-tick labels from nested subplots
    for ax in axs:
        ax.set_yticklabels([])


    plt.tight_layout()
    plt.show()


def plot_dataset_error(all_errors):
    # Calculate the mean absolute error for each column
    mean_absolute_errors = np.mean(np.abs(all_errors), axis=0)

    # Get the sorted indices based on the mean absolute error
    sorted_indices = np.argsort(mean_absolute_errors)

    # Sort the columns of all_errors based on the sorted indices
    sorted_all_errors = all_errors[:, sorted_indices]

    # Plot the sorted all_errors
    plt.figure(figsize=(10, 5))
    plt.imshow(np.sqrt(sorted_all_errors), aspect='auto', cmap='viridis')
    plt.colorbar(label='sqrt of MSE Error')
    plt.xlabel('Zones (sorted by error)')
    plt.ylabel('Samples')
    plt.title('Errors Sorted by Zone')
    plt.show()