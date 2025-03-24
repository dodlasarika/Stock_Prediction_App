import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = load_model("stock_price_lstm_model.h5")  # Make sure this model is saved

# Streamlit UI
st.title("📈 AI-Powered Stock Price Predictor")

# Select stock
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, GOOGL):", "AAPL")
future_days = st.slider("Select Future Days for Prediction:", min_value=7, max_value=365, value=30)

# Download stock data
st.write(f"Fetching stock data for **{stock_symbol}**...")
data = yf.download(stock_symbol, start="2020-01-01", end="2025-01-01")
df = data[['Close']].dropna()  # Keep only closing prices

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)

# Prepare last sequence for prediction
seq_length = 50
last_sequence = df_scaled[-seq_length:]
future_predictions = []

for _ in range(future_days):
    prediction = model.predict(last_sequence.reshape(1, seq_length, 1))
    future_predictions.append(prediction[0][0])
    last_sequence = np.append(last_sequence[1:], prediction, axis=0)

# Reverse scaling
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Create future dates
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date, periods=future_days + 1)[1:]

# Plot results
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df.index, df['Close'], label="Actual Prices", color='blue')
ax.plot(future_dates, future_predictions, label="Predicted Prices", color='green')
ax.set_title(f"{stock_symbol} Stock Price Prediction")
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price (USD)")
ax.legend()
st.pyplot(fig)

st.success("✅ Prediction Complete! Adjust settings above to see different results.")
