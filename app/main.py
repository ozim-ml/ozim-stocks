from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from io import BytesIO
import base64
import os

import yfinance as yf
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from arch import arch_model
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

    return ticker, df, plot_base64_adj_close

def plot_daily_returns(df, ticker):

    global returns

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

    return returns, plot_base64_daily_returns

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

def simple_model(ticker, returns, sym_in: int, asym_in: int, lag_vol: int, hor: int):

    am = arch_model(returns, vol="GARCH", p=sym_in, o=asym_in, q=lag_vol, dist="normal")
    res = am.fit(update_freq=5)

    horizon = hor
    forecasts = res.forecast(horizon=horizon, method="simulation", reindex=False)
    sims = forecasts.simulations

    x = np.arange(1, horizon + 1)
    lines = plt.plot(x, sims.residual_variances[-1, ::5].T, color="#9cb2d6", alpha=0.5)
    lines[0].set_label("Simulated path")
    line = plt.plot(x, forecasts.variance.iloc[-1].values, color="#002868")
    line[0].set_label("Expected variance")
    plt.title(f'Simulation forecast of {ticker}')
    plt.gca().set_xticks(x)
    plt.gca().set_xlim(1, horizon)
    plt.legend()

    # Create a BytesIO object to store the plot
    plot_bytes_arch = BytesIO()

    # Save the plot to BytesIO and encode as base64
    plot_bytes_arch.seek(0)
    plt.savefig(plot_bytes_arch, format='png')
    plot_base64_arch = base64.b64encode(plot_bytes_arch.getvalue()).decode('utf-8')

    plt.close()

    return plot_base64_arch

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/analyze", response_class=HTMLResponse)
async def analyze_ticker(
    request: Request,
    ticker_query: str = Query(..., min_length=1, description="Stock ticker"),
    start_date: str = Query(..., description="Start date for analysis"),
    end_date: str = Query(..., description="End date for analysis"),
):
    global ticker
    ticker = ticker_query
    ticker, df, plot_base64_adj_close = perform_analysis(ticker, start_date, end_date)
    returns, plot_base64_daily_returns = plot_daily_returns(df, ticker)
    plot_base64_acf = plot_acf(returns, ticker)

    return templates.TemplateResponse("analysis.html", {
        "request": request,
        "ticker": ticker,
        "plot_adj_close": plot_base64_adj_close,
        "plot_daily_returns": plot_base64_daily_returns,
        "plot_acf": plot_base64_acf
    })

@app.get("/create_model", response_class=HTMLResponse)
async def create_arch(
        request: Request,
        sym_in: int = Query(..., description="Lag order of the symmetric innovation"),
        asym_in: int = Query(..., description="Lag order of the asymmetric innovation"),
        lag_vol: int = Query(..., description="Lag order of lagged volatility"),
        hor: int = Query(..., description="Horizon of forecast")
):
    plot_base64_arch = simple_model(ticker, returns, sym_in, asym_in, lag_vol, hor)
    return templates.TemplateResponse("model.html", {
        "request": request,
        "plot_arch": plot_base64_arch,
        "ticker": ticker,
        "sym_in": sym_in,
        "asym_in": asym_in,
        "lag_vol": lag_vol,
        "hor": hor
    })
