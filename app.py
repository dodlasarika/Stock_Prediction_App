import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from newsapi import NewsApiClient  # ✅ Added NewsAPI for stock news
from textblob import TextBlob  # ✅ Added TextBlob for sentiment analysis
import datetime

# Load the trained model
model = load_model("stock_price_lstm_model.h5")  # Ensure this model is saved

# Initialize NewsAPI (Replace 'YOUR_NEWSAPI_KEY' with your actual key)
newsapi = NewsApiClient(api_key="058d6155c198466ea75d5ce4efdacb4b")

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
with st.spinner("Downloading stock data..."):
    try:
        data = yf.download(stock_symbol, start=start_date, end=end_date)
        df = data[['Close']].dropna()  # Keep only closing prices
    except Exception as e:
        st.error(f"❌ Error fetching stock data: {e}")
        st.stop()

if df.empty:
    st.warning("⚠️ No data available for the selected stock and date range.")
    st.stop()

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
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

# Plot Actual Prices & Predictions using Candlestick Chart
fig = go.Figure()

# Actual Stock Prices as a Candlestick Chart
fig.add_trace(go.Candlestick(
    x=df.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=df["Close"],
    name="Actual Prices",
    increasing_line_color='blue',
    decreasing_line_color='red'
))

# Future Predicted Prices
fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_predictions.flatten(),
    mode='lines',
    name="Future Predictions",
    line=dict(color='green', width=2, dash="dash")  # Dashed line for future predictions
))

# Add Moving Averages
df["SMA50"] = df["Close"].rolling(window=50).mean()
df["SMA200"] = df["Close"].rolling(window=200).mean()

fig.add_trace(go.Scatter(
    x=df.index, 
    y=df["SMA50"], 
    mode="lines", 
    name="50-Day SMA", 
    line=dict(color="orange")
))

fig.add_trace(go.Scatter(
    x=df.index, 
    y=df["SMA200"], 
    mode="lines", 
    name="200-Day SMA", 
    line=dict(color="red")
))

# Update Layout to Ensure Date Formatting
fig.update_layout(
    title=f"{stock_symbol} Stock Price Prediction",
    xaxis=dict(title="Date", type="date"),
    yaxis_title="Stock Price (USD)",
    xaxis_rangeslider_visible=False,
    template="plotly_dark"
)

# Display the Graph in Streamlit
st.plotly_chart(fig, use_container_width=True)

### 🔹 Add News & Sentiment Analysis Section ###
st.subheader(f"📰 Latest {stock_symbol} News & Sentiment Analysis")

def fetch_stock_news(stock_symbol):
    """Fetch latest news articles related to the selected stock."""
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    articles = newsapi.get_everything(q=stock_symbol, from_param=today, language='en', sort_by='relevancy')
    
    news_list = []
    for article in articles['articles'][:5]:  # Limit to 5 latest articles
        title = article['title']
        url = article['url']
        description = article['description'] or "No description available"
        sentiment = analyze_sentiment(description)
        news_list.append({"title": title, "url": url, "sentiment": sentiment})
    
    return news_list

def analyze_sentiment(text):
    """Analyze sentiment of the news article (Positive, Negative, Neutral)."""
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "🟢 Positive"
    elif polarity < 0:
        return "🔴 Negative"
    else:
        return "🟡 Neutral"

# Fetch & Display News
news_articles = fetch_stock_news(stock_symbol)

if news_articles:
    for news in news_articles:
        st.write(f"**[{news['title']}]({news['url']})** - {news['sentiment']}")
else:
    st.write("No recent news found for this stock.")
