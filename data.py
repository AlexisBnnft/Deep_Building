import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def load_tload(data_path):
    df = pd.read_csv(data_path)
    df.index = pd.to_datetime(df['Date'])
    df = df.drop(columns=['Date'])
    df.resample('h').mean()
    df.index = pd.to_datetime(df.index)
    return df

def load_weather(data_path):
    weather = pd.read_csv(data_path)
    weather.index = pd.to_datetime(weather['Date']) 
    weather = weather.drop(columns=['Date'])
    weather.resample('h').mean()
    weather.index = pd.to_datetime(weather.index)
    weather.drop(columns=['daily_rain'], inplace=True)
    return weather


def nan_viz(df, wide = False):
    # Visualize NaN locations
    nan_locations = df.isna().astype(int)
    if wide:
        plt.figure(figsize=(10, 5))
    else:
        plt.figure(figsize=(3, 2))
    plt.imshow(nan_locations, aspect='auto', cmap='gray', interpolation='nearest')
    plt.colorbar(label='NaN Indicator (1 = NaN, 0 = Non-NaN)')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.title('NaN Locations in Terminal Load DataFrame')
    plt.show()
    
    # Find days where all VAV columns have NaN values
    all_nan_days = df.index[df.isna().all(axis=1)]
    return all_nan_days


def tload_viz(tload_df, small = False, title = "Building Wide Terminal Load Profile"):    
    # Assuming tload_df is your DataFrame
    # Create a copy of the DataFrame to avoid modifying the original
    tload_df_copy = tload_df.copy()

    # Define the custom colormap
    cmap = plt.cm.RdBu  # Red to Blue colormap
    cmap.set_bad(color='black')  # Set color for NaN values

    # Replace NaN values with a masked array
    masked_array = np.ma.masked_invalid(tload_df_copy)

    # Plot the DataFrame with the custom colormap
    if small:
        plt.figure(figsize=(25, 5))
    else:
        plt.figure(figsize=(25, 14))
    plt.imshow(masked_array, aspect='auto', cmap=cmap, interpolation='nearest', vmin=-100, vmax=100)
    plt.colorbar(label='Value')
    plt.xlabel('Vav')
    plt.ylabel('Time')
    plt.title(title)
    plt.show()

def error_viz(error_df):    
    # Assuming tload_df is your DataFrame
    # Create a copy of the DataFrame to avoid modifying the original
    error_df = np.abs(error_df.copy())

    # Define the custom colormap
    cmap = plt.cm.viridis  # Red to Blue colormap
    cmap.set_bad(color='black')  # Set color for NaN values

    # Replace NaN values with a masked array
    masked_array = np.ma.masked_invalid(error_df)

    # Plot the DataFrame with the custom colormap
    plt.figure(figsize=(25, 14))
    plt.imshow(masked_array, aspect='auto', cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Error')
    plt.xlabel('Vav')
    plt.ylabel('Time')
    plt.title('Absolute Error on the prediction on the validation set')
    plt.show()
