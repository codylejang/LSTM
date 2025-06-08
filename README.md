
# LSTM Stock Price Predictor

This project implements a Long Short-Term Memory (LSTM) neural network to predict the stock price of Microsoft (MSFT) using historical daily closing prices. The model is trained on past price data and forecasts future values using a time-series approach.

## Overview

The model is built using TensorFlow/Keras and follows a standard pipeline of:
- Fetching historical stock price data from the ORATS API
- Normalizing and preparing time-series data
- Training a deep LSTM network
- Predicting future stock prices and visualizing results

## Features

- Data sourced directly from ORATS using HTTP requests
- Preprocessing using `MinMaxScaler`
- LSTM network with multiple layers and dropout for regularization
- Early stopping to prevent overfitting
- Forecasts and visualizations for historical, validation, and future stock price predictions

## Data

Data is pulled from the [ORATS API](https://api.orats.io/) using custom HTTP GET requests. The model uses:
- Training set: 2000–2016
- Validation set: 2017–2023
- Testing set: 2023–2024
- Prediction: 4 days ahead from the most recent available data

## Model Architecture

- 4 LSTM layers with 50 units each
- Dropout regularization (0.2 and 0.5)
- Fully connected Dense layer with 1 output
- Optimizer: Adam
- Loss function: Mean Squared Error

## Visualization

The notebook visualizes:
- Training and validation loss
- Predicted vs. actual stock prices
- Forecasted stock prices for the next 4 days

## Usage

To run the project:

1. Make sure you have an ORATS API token and replace `'mytoken'` in the code.
2. Install required packages:
3. Run the notebook or Python script.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn
- tensorflow
- requests

## Author

**Cody Lejang**
B.S. in Cognitive Science, Specialization in Computing, minor in Data Science Engineering – UCLA
Interested in the intersection of machine learning, psychology, and data analytics.

Exploring deep learning applications in time series forecasting, specifically in financial markets.
