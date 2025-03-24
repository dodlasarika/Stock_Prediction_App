import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = load_model("stock_price_lstm_model.h5")  # Ensure this model is saved

# Streamlit UI - Sidebar
st.sidebar.title("📊 AI Stock Price Predictor")
st.sidebar.write("Choose a stock and prediction settings below:")

# Select stock
stock_list = ["AAPL", "TSLA", "GOOGL", "AMZN", "MSFT"]
stock_symbol = st.sidebar.selectbox("Select Stock Symbol", stock_list)

# Select date range
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))

# Select future prediction days
future_days = st.sidebar.slider("Days to Predict", min_value=7, max_value=365, value=30)

# Fetch stock data
st.sidebar.write(f"📥 Fetching stock data for **{stock_symbol}**...")
data = yf.download(stock_symbol, start=start_date, end=end_date)
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

# Plot Actual Prices with Moving Averages
fig = go.Figure()

fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name="Actual Prices", line=dict(color='blue')))
fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), mode='lines', name="Future Predictions", line=dict(color='green')))

# Calculate Moving Averages
df['SMA50'] = df['Close'].rolling(window=50).mean()
df['SMA200'] = df['Close'].rolling(window=200).mean()

# Add Moving Averages to Chart
fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name="50-Day SMA", line=dict(color='orange')))
fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], mode='lines', name="200-Day SMA", line=dict(color='red')))

fig.update_layout(title=f"{stock_symbol} Stock Price Prediction",
                  xaxis_title="Date",
                  yaxis_title="Stock Price (USD)",
                  legend=dict(x=0, y=1),
                  template="plotly_dark")

st.plotly_chart(fig)

st.success("✅ Prediction Complete! Adjust settings above to see different results.")
