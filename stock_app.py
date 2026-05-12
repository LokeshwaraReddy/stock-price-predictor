import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("📈 Stock Price Prediction with Sentiment Analysis")
st.markdown("Predict stock prices using historical data + news sentiment")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("merged_stock_sentiment_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# Sidebar
st.sidebar.header("Select Stock")
company = st.sidebar.selectbox("Company", df['Company'].unique())

# Filter data
company_df = df[df['Company'] == company].copy()

# Preprocess (same as your notebook)
company_df = company_df.sort_values('Date')
company_df.ffill(inplace=True)

# Show raw data
st.subheader(f"📊 {company} Stock Data")
st.dataframe(company_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment']].tail(10))

# Prepare features
company_df['Next_Close'] = company_df['Close'].shift(-1)
company_df.dropna(inplace=True)

features = ['Open', 'High', 'Low', 'Close', 'Volume']
if 'Sentiment_Score' in company_df.columns:
    features.append('Sentiment_Score')

X = company_df[features]
y = company_df['Next_Close']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

col1, col2 = st.columns(2)
col1.metric("R² Score", f"{r2:.3f}")
col2.metric("MSE", f"{mse:.2f}")

# Plot
st.subheader("Actual vs Predicted Prices")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y_test.values, label='Actual', color='blue')
ax.plot(y_pred, label='Predicted', color='red', alpha=0.7)
ax.set_title(f"{company} - Actual vs Predicted")
ax.legend()
st.pyplot(fig)

# Feature importance
st.subheader("Feature Importance")
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})
st.bar_chart(coef_df.set_index('Feature'))

# Prediction for tomorrow
st.subheader("🔮 Tomorrow's Prediction")
last_row = X.iloc[-1:]
future_price = model.predict(last_row)[0]
current_price = company_df['Close'].iloc[-1]

st.metric("Current Price", f"${current_price:.2f}")
st.metric("Predicted Price", f"${future_price:.2f}", 
          delta=f"{((future_price/current_price)-1)*100:.2f}%")

if future_price > current_price:
    st.success("📈 Market Direction: UP")
else:
    st.error("📉 Market Direction: DOWN")

st.caption("Built with Linear Regression | Includes sentiment analysis")