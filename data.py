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
    return weather


def nan_viz(df):
    # Visualize NaN locations
    nan_locations = df.isna().astype(int)
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


def tload_viz(tload_df):    
    # Assuming tload_df is your DataFrame
    # Create a copy of the DataFrame to avoid modifying the original
    tload_df_copy = tload_df.copy()

    # Define the custom colormap
    cmap = plt.cm.RdBu  # Red to Blue colormap
    cmap.set_bad(color='black')  # Set color for NaN values

    # Replace NaN values with a masked array
    masked_array = np.ma.masked_invalid(tload_df_copy)

    # Plot the DataFrame with the custom colormap
    plt.figure(figsize=(25, 14))
    plt.imshow(masked_array, aspect='auto', cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Value')
    plt.xlabel('Vav')
    plt.ylabel('Time')
    plt.title('Building Wide Terminal Load Profile')
    plt.show()
