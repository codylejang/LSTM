import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
import requests

def generate_data(ticker, start_date, end_date):
    
    url = 'https://api.orats.io/datav2/hist/dailies'
    payload = {
        'token': 'mytoken',
        'ticker': ticker,
        'tradeDate': start_date + ',' + end_date,
        'fields': 'tradeDate,ticker,clsPx'
    }

    response = requests.get(url, params=payload)
    response_dict = response.json()

    # Extracting 'data' from JSON
    data_list = response_dict['data']

    # Creating DataFrame
    ticker_df = pd.DataFrame(data_list)

    # Reformat
    ticker_df = ticker_df[['tradeDate', 'clsPx']]
    # Convert 'date' column to datetime
    ticker_df['tradeDate'] = pd.to_datetime(ticker_df['tradeDate'])
    ticker_df.set_index('tradeDate', inplace=True)
    data = ticker_df.values
    
    return data


#training and validation
training_data = generate_data('MSFT','2000-08-01','2016-12-31')
validation_data = generate_data('MSFT','2017-01-01','2023-12-31')

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(training_data)
val_scaled = scaler.fit_transform(validation_data)

# Define the number of time steps
time_steps = 3

X_train = []
y_train = []

# Loop through the data to create partitions
for i in range(time_steps, train_scaled.shape[0]):
    # Create a partition of the previous 3 days' data
    X_train.append(train_scaled[i - time_steps:i])

    # Append the next day's Close price to the label array
    y_train.append(train_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_val = []
y_val = []
for i in range(time_steps, val_scaled.shape[0]):
    X_val.append(val_scaled[i - time_steps:i])
    y_val.append(val_scaled[i, 0])

X_val, y_val = np.array(X_val), np.array(y_val)
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

# Building the LSTM Model
model = keras.Sequential()
model.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(units=50, return_sequences=True))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(units=50, return_sequences=True))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.LSTM(units=50))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(units=1))

# Compiling the LSTM Model
model.compile(optimizer='adam', loss='mean_squared_error')

#early stopping callback, stop training if model does not improve for 3 consecutive epochs
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)

# Training the Model and store history
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[callback])

# Plot loss and accuracy during training
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Predictions
test_data= generate_data('MSFT','2023-06-01','2024-06-01')

test_scaled = scaler.fit_transform(test_data)

# Prepare test data
X_test = []
y_test = []
for i in range(time_steps, test_scaled.shape[0]):
    X_test.append(test_scaled[i - time_steps:i])
    y_test.append(test_scaled[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Making predictions
predicted_scaled = model.predict(X_test)
predicted = scaler.inverse_transform(predicted_scaled)
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

# Graph
plt.figure(figsize=(14, 7))
plt.plot(y_test_original, color='blue', label='True Values')
plt.plot(predicted, color='red', label='Predicted Values')
plt.title('LSTM Model Predictions vs True Values')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

rmse = np.sqrt(mean_squared_error(y_test_original, predicted))

print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')

#predict the next 4 days

# generate the latest test data up to the current date
current_date = pd.Timestamp.now().strftime('%Y-%m-%d')
recent_data = generate_data('MSFT', '2024-01-01', current_date)

test_scaled = scaler.transform(recent_data)

#Prepare input sequence for the next 4 days prediction
input_sequence = test_scaled[-time_steps:].reshape(1, time_steps, 1)

# Predict the next 4 days
predicted_prices = []
for _ in range(4):  # Predicting the next 4 days
    predicted_price = model.predict(input_sequence)
    predicted_prices.append(predicted_price[0, 0])
    
    # Update input_sequence by removing the first element and adding the predicted price
    input_sequence = np.append(input_sequence[:, 1:, :], np.reshape(predicted_price, (1, 1, 1)), axis=1)

# Inverse transform the predicted prices
predicted_prices = np.array(predicted_prices).reshape(-1, 1)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the predictions
plt.figure(figsize=(14, 7))
plt.plot(range(len(test_data)), test_data[:, 0], color='blue', label='Historical Prices')
plt.plot(range(len(test_data), len(test_data) + len(predicted_prices)), predicted_prices, color='red', label='Predicted Prices')
plt.title('LSTM Model Predictions for the Next 4 Days')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()





    



