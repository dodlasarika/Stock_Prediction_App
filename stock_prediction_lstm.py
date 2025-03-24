# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

# Select the stock symbol (Change this to test different stocks)
stock_symbol = "AAPL"  # Example: AAPL (Apple), TSLA (Tesla), GOOGL (Google), AMZN (Amazon)

# Download Stock Data
data = yf.download(stock_symbol, start="2020-01-01", end="2025-01-01")

# Show first 5 rows
print(data.head())

# Plot Closing Price
plt.figure(figsize=(12,6))
plt.plot(data['Close'], label="Closing Price")
plt.title(f"{stock_symbol} Stock Closing Price")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

# Keep only the 'Close' price
df = data[['Close']].dropna()  # Remove missing values

# Normalize the data (Scale between 0 and 1 for LSTM)
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)

# Split into training (80%) and testing (20%)
train_size = int(len(df_scaled) * 0.8)
train_data, test_data = df_scaled[:train_size], df_scaled[train_size:]

# Function to create sequences for LSTM input
def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 50  # Number of past days used for prediction
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

print(f"Training Data Shape: {X_train.shape}, Labels Shape: {y_train.shape}")
print(f"Testing Data Shape: {X_test.shape}, Labels Shape: {y_test.shape}")

# Build the LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),  # First LSTM layer
    Dropout(0.2),  # Dropout for regularization
    LSTM(50, return_sequences=False),  # Second LSTM layer
    Dropout(0.2),
    Dense(25),  # Fully connected layer
    Dense(1)  # Output layer (predicting stock price)
])

# Compile the Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the Model
history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))

# Plot Training Loss
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Training Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Make Predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Reverse scaling to get actual price values
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot Actual vs Predicted Prices
plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label="Actual Prices", color='blue')
plt.plot(test_predictions, label="Predicted Prices", color='red')
plt.title(f"{stock_symbol} Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.show()

# Predict Future Stock Prices (Next 30 Days)
future_days = 365 # Number of days to predict

# Get the last sequence from the test dataset to start predicting
last_sequence = test_data[-seq_length:]
future_predictions = []

for _ in range(future_days):
    prediction = model.predict(last_sequence.reshape(1, seq_length, 1))  # Predict 1 day
    future_predictions.append(prediction[0][0])  # Save the predicted value
    last_sequence = np.append(last_sequence[1:], prediction, axis=0)  # Update the sequence

# Reverse scale predictions back to original stock price range
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Create future dates for plotting
last_date = df.index[-1]  # Get last date in dataset
future_dates = pd.date_range(start=last_date, periods=future_days + 1)[1:]  # Generate future dates

# Plot Future Predictions
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], label="Actual Prices", color='blue')  # Past actual prices
plt.plot(future_dates, future_predictions, label="Future Predictions", color='green')  # Future predicted prices
plt.title(f"{stock_symbol} Stock Price Prediction for Next {future_days} Days")
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.show()
model.save("stock_price_lstm_model.h5")
model.save("stock_price_lstm_model.h5")
print("✅ Model saved successfully!")
