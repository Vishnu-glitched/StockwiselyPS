from django.shortcuts import redirect, render, HttpResponse
from sklearn.model_selection import train_test_split
from django.contrib.auth.models import User
from django.contrib.auth  import authenticate,login

# Create your views here.
def index(request):
    # return HttpResponse("This is the Home Page")
    return render(request,'index.html')

def about(request):
    # return HttpResponse("This is the Home Page")
    return render(request,'about.html')


#MODEL


import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from django.shortcuts import render

# Helper function to generate a plot
def generate_plot(prices, days, company):
    plt.figure(figsize=(10, 6))
    plt.plot(days, prices, label='Predicted Stock Prices', color='blue')
    plt.title(f'Stock Price Prediction for {company} - Next 10 Days')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return plot_url


# Linear Regression with Gradient Descent
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None

    def fit(self, X, y):
        m = len(X)
        self.theta = np.zeros((X.shape[1], 1))
        for _ in range(self.epochs):
            predictions = X.dot(self.theta)
            error = predictions - y
            gradient = X.T.dot(error) / m
            self.theta -= self.learning_rate * gradient

    def predict(self, X):
        return X.dot(self.theta)


# RSI Calculation
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# EMA Calculation
def calculate_ema(prices, period=14):
    return prices.ewm(span=period, adjust=False).mean()


# Prediction function for external A
def fetch_gemini_prediction_google(prompt):
    gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    api_key = "AIzaSyDYCVX1Bg9y2lbNnxPGLjZ366tnYT5SfcQ"  # Get the API key from environment variables
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        return None

    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    params = {"key": api_key}  # Pass the API key in the query parameters, as required by the Google API.

    try:
        response = requests.post(gemini_url, json=data, headers=headers, params=params)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx).
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching prediction from Gemini API: {e}")
        return None


# Prediction function using local model
def predict_with_local_model(company, future_date):
    try:
        df = pd.read_excel("C:/StockWisely/stockwiselyps/home/yahoo_data.xlsx")
        df["RSI"] = calculate_rsi(df["Close*"])
        df["EMA"] = calculate_ema(df["Close*"])
        df.dropna(inplace=True)

        features = ["RSI", "EMA", "Open", "High", "Low", "Volume"]
        X = df[features].values
        y = df["Close*"].values.reshape(-1, 1)

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X = scaler_X.fit_transform(X)
        y = scaler_y.fit_transform(y)

        train_size = int(0.7 * len(X))
        X_train, y_train = X[:train_size], y[:train_size]
        X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

        lr_gd = LinearRegressionGD(learning_rate=0.009, epochs=10000)
        lr_gd.fit(X_train, y_train)

        last_row = df.iloc[-1]
        rsi_diff = df["RSI"].iloc[-1] - df["RSI"].iloc[-2]
        ema_diff = df["EMA"].iloc[-1] - df["EMA"].iloc[-2]
        days_into_future = (future_date - datetime.now().date()).days
        future_rsi = min(max(last_row["RSI"] + rsi_diff * days_into_future, 0), 100)
        future_ema = last_row["EMA"] + ema_diff * days_into_future
        future_features = [future_rsi, future_ema, last_row["Open"], last_row["High"], last_row["Low"],
                           last_row["Volume"]]
        future_features_scaled = scaler_X.transform([future_features])
        future_features_scaled = np.hstack((np.ones((1, 1)), future_features_scaled))
        future_price_scaled = lr_gd.predict(future_features_scaled)
        future_price = scaler_y.inverse_transform(future_price_scaled)

        return future_price[0][0]

    except Exception as e:
        print(f"Error in local model prediction: {e}")
        return None


# Main Prediction Function
def Prediction(request):
    prediction_result = None
    error_message = None
    plot_url = None

    if request.method == "POST":
        company = request.POST.get("company")
        user_date = request.POST.get("DATE")

        try:
            future_date = datetime.strptime(user_date, "%Y-%m-%d").date()

            # Check if the selected company is "Yahoo"
            if company == "Yahoo Finance(Provided DataSet)":
                # Use the local model to predict the stock price for Yahoo
                predicted_price = predict_with_local_model(company, future_date)
                if predicted_price is not None:
                    prediction_result = f"Predicted stock price for Yahoo on {future_date}: ${predicted_price:.2f}"
                    days = [i for i in range(1, 11)]
                    predicted_prices = [predicted_price] * 10
                    plot_url = generate_plot(predicted_prices, days, company)
                else:
                    error_message = "Failed to predict using local model."

            else:
                # Construct a prompt for the Gemini API, asking for a guess.
                prompt = f"Give me a *speculative guess* for the stock price of {company} on {user_date}. If it's not possible provide a generic reply with 'Not Available'. Just provide a single number with '$' sign for the predicted price."

                response = fetch_gemini_prediction_google(prompt)

                if response:
                    # Check for the response from Gemini API and update accordingly.
                    try:
                        response_text = response['candidates'][0]['content']['parts'][0]['text'].strip()

                        # Check for "Not Available" response
                        if response_text.lower() == "not available":
                            prediction_result = f"Gemini cannot provide a speculative guess for {company} on {user_date}."
                        else:
                            # Attempt to extract the numerical value for further processing.
                            try:
                                predicted_price = float(response_text.replace('$', ''))
                                prediction_result = f"Predicted stock price for {company} on {future_date}: ${predicted_price:.2f}"
                                days = [i for i in range(1, 11)]
                                predicted_prices = [predicted_price] * 10
                                plot_url = generate_plot(predicted_prices, days, company)
                            except Exception as e:
                                error_message = f"Failed to parse a numeric price from Gemini response: {e}, Response: {response_text}"

                    except Exception as e:
                        error_message = f"Failed to process response: {e}, Raw Response: {response}"

                else:
                    error_message = "Failed to fetch prediction from Gemini API."

        except Exception as e:
            error_message = f"Error: {str(e)}"

    return render(request, "Prediction.html", {
        "prediction_result": prediction_result,
        "error_message": error_message,
        "plot_url": plot_url
    })
