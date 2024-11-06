# Deep_Building
CS230 Project: Using a Long Short Term Memory Network to predicting Terminal Load (a zonal metric of cooling or heating need) time series of a multi-zone commercial building with weather data. 

# Terminal Load Forecasting

This project is a deep learning-based solution for forecasting terminal load (energy demand) using historical load and weather data.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training and Evaluation](#training-and-evaluation)
5. [Usage](#usage)
6. [Visualization](#visualization)
7. [Dependencies](#dependencies)
8. [Contributing](#contributing)
9. [License](#license)

## Introduction
Terminal load forecasting can be a crucial task in building operation, as it represents the main indicator of the various energetic need of zones in a building. This accurate predictions of energy demand can help forecasting cooling load, optimize power generation, and even understand better the main trends within a building. This project aims to develop a robust deep learning model to predict terminal load based on historical load and weather data.

## Dataset
The dataset used in this project consists of:
- Terminal load data: Hourly energy demand for multiple zones, which goes from -100 to 100. The calculation of such a metric is building specific, but is mainly related to its setpoint, and measurement of temperature and airflow within this szone
- Weather data: Relevant weather features (e.g., temperature, humidity, wind speed) for the same time period

The dataset is then preprocessed and split into training and validation sets, which for now correspond for 4 month  of training data and 1 month of validating. Data may be available upon request. 

In the code the weather dataframe is formatted like so:
| Date                | temperature | RH  | Tdew | wind | sun_rad | daily_rain |
|---------------------|-------------|-----|------|------|---------|------------|
| 2023-05-01 00:00:00 | 52.6        | 75.0| 44.8 | 9.2  | 0.0     | 0.00       |
| 2023-05-01 01:00:00 | 52.4        | 75.0| 44.7 | 7.4  | 0.0     | 0.00       |

The features don't have to be all present, and you can even have more weather features if you have some. 

The terminal load dataframe is formated as so:
| Date                | VAV2-33 | VAV2-17   | VAV2-29 | ... | VAV2-11 | VAV2-18 | VAV3-15 |
|---------------------|---------|-----------|---------|-----|---------|---------|---------|
| 2023-05-01 00:00:00 | 0.0     | 0.000000  | -100.0  | ... | 0.0     | 0.0     | 0.000000|
| 2023-05-01 01:00:00 | 0.0     | -0.355917 | -100.0  | ... | 0.0     | 0.0     | 0.000000|



## Model Architecture
The model architecture is based on a hybrid approach, combining Long Short-Term Memory (LSTM) networks to process the temporal load and weather data, and fully connected layers to generate the final load predictions.

## Training and Evaluation
The model is trained using PyTorch and optimized using the Adam optimizer. The training process minimizes the Mean Squared Error (MSE) loss between the predicted and actual load values.

During training, the model's performance is evaluated on the validation set, and the training and validation losses are tracked.

## Usage
To use the terminal load forecasting model, follow these steps:
1. Ensure you have the required dependencies installed (see [Dependencies](#dependencies) section).
2. Prepare your terminal load and weather data in the appropriate format.
3. Instantiate the `TerminalLoadDataset` and `TerminalLoadPredictor` classes.
4. Train the model using the `train_model` function.
5. Use the trained model to generate predictions on new data.

## Visualization
The project includes a `PredictionVisualizer` class that provides the following visualization capabilities:
- Plot the training and validation loss history
- Display actual vs. predicted load values for selected zones
- Plot the error distribution for the predictions
- Print key performance metrics (MAPE, RMSE) for each zone

## Dependencies
The project relies on the following main dependencies:
- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

See the `requirements.txt` file for the complete list of dependencies.

## Contributing
Contributions to this project are welcome. If you find any issues or have ideas for improvements, please feel free to open a new issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).
