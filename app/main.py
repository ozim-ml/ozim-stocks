from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from io import BytesIO
import base64
import os

import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from statsmodels.tsa.stattools import acf

sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

yf.pdr_override()

app = FastAPI()

current_directory = os.path.dirname(os.path.realpath(__file__))
static_folder = os.path.join(current_directory, "static")
templates_folder = os.path.join(static_folder, "templates")

app.mount("/static", StaticFiles(directory=static_folder), name="static")
templates = Jinja2Templates(directory=templates_folder)

def perform_analysis(ticker: str, start_date: str, end_date: str):
    # Convert start_date and end_date to datetime objects
    start_date = dt.datetime.strptime(start_date, "%Y-%m-%d")
    end_date = dt.datetime.strptime(end_date, "%Y-%m-%d")

    # Load data from yfinance for selected tickers within the specified date range
    df = yf.download(ticker, start=start_date, end=end_date)

    # Plot the Adjusted Closing Price
    plt.plot(df.index, df['Adj Close'], label='Adj Close', linewidth=1.5)
    plt.title(f'Adj Close of {ticker}')
    plt.xlabel(None)
    plt.ylabel('Adj Close Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # Create a BytesIO object to store the plot
    plot_bytes_adj_close = BytesIO()

    # Save the plot to BytesIO and encode as base64
    plot_bytes_adj_close.seek(0)
    plt.savefig(plot_bytes_adj_close, format='png')
    plot_base64_adj_close = base64.b64encode(plot_bytes_adj_close.getvalue()).decode('utf-8')

    plt.close()

    # Plot the Daily Returns and ACF
    plot_base64_daily_returns, plot_base64_acf = plot_daily_returns(df, ticker)

    return plot_base64_adj_close, plot_base64_daily_returns, plot_base64_acf

def plot_daily_returns(df, ticker):
    # Calculate daily returns
    returns = 100 * df['Adj Close'].pct_change().dropna()

    # Plot the daily returns
    plt.plot(returns.index, returns, label='Daily return', linewidth=1.5)
    plt.title(f'Daily return of {ticker}')
    plt.xlabel(None)
    plt.ylabel('PCT value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # Create a BytesIO object to store the plot
    plot_bytes_daily_returns = BytesIO()

    # Save the plot to BytesIO and encode as base64
    plot_bytes_daily_returns.seek(0)
    plt.savefig(plot_bytes_daily_returns, format='png')
    plot_base64_daily_returns = base64.b64encode(plot_bytes_daily_returns.getvalue()).decode('utf-8')

    plt.close()

    # Plot ACF of Squared Daily Returns
    plot_base64_acf = plot_acf(returns, ticker)

    return plot_base64_daily_returns, plot_base64_acf

def plot_acf(returns, ticker):
    # Calculate squared returns
    squared_returns = returns ** 2

    lags = range(1, 50)
    acf_values = acf(squared_returns, nlags=len(lags))

    # Plot the autocorrelation function for squared returns
    plt.bar(lags, acf_values[1:])  # Omit lag 0
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(f'ACF of squared daily returns of {ticker}')

    # Create a BytesIO object to store the plot
    plot_bytes_acf = BytesIO()

    # Save the plot to BytesIO and encode as base64
    plot_bytes_acf.seek(0)
    plt.savefig(plot_bytes_acf, format='png')
    plot_base64_acf = base64.b64encode(plot_bytes_acf.getvalue()).decode('utf-8')

    plt.close()

    return plot_base64_acf

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/analyze", response_class=HTMLResponse)
async def analyze_ticker(
    request: Request,
    ticker: str = Query(..., min_length=1, description="Stock ticker"),
    start_date: str = Query(..., description="Start date for analysis"),
    end_date: str = Query(..., description="End date for analysis")
):
    plot_base64_adj_close, plot_base64_daily_returns, plot_base64_acf = perform_analysis(ticker, start_date, end_date)
    return templates.TemplateResponse("analysis.html", {
        "request": request,
        "ticker": ticker,
        "plot_adj_close": plot_base64_adj_close,
        "plot_daily_returns": plot_base64_daily_returns,
        "plot_acf": plot_base64_acf,
    })
