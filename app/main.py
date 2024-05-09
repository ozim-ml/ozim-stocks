import subprocess
subprocess.Popen("mlflow server --host 127.0.0.1 --port 5000")

import mlflow
import mlflow.sklearn
import mlflow.tensorflow

remote_server_uri = "http://127.0.0.1:5000/"
mlflow.set_tracking_uri(remote_server_uri)

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

yf.pdr_override()

app = FastAPI()

current_directory = os.path.dirname(os.path.realpath(__file__))
static_folder = os.path.join(current_directory, "static")
templates_folder = os.path.join(static_folder, "templates")

app.mount("/static", StaticFiles(directory=static_folder), name="static")
templates = Jinja2Templates(directory=templates_folder)

from app.basic_plots import *
from app.arch_forecast import *
from app.rnn import *

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/basic_plots", response_class=HTMLResponse)
async def basic_plots(
    request: Request,
    ticker_query: str = Query(..., min_length=1, description="Stock ticker"),
    start_date: str = Query(..., description="Start date for analysis"),
    end_date: str = Query(..., description="End date for analysis"),
):
    global ticker, stock_df, returns
    ticker = ticker_query
    ticker, stock_df, plot_base64_adj_close = perform_analysis(ticker, start_date, end_date)
    returns, plot_base64_daily_returns = plot_daily_returns(stock_df, ticker)
    plot_base64_risk_return = plot_risk_ret(returns, ticker)

    return templates.TemplateResponse("analysis.html", {
        "request": request,
        "ticker": ticker,
        "plot_adj_close": plot_base64_adj_close,
        "plot_daily_returns": plot_base64_daily_returns,
        "plot_risk_ret": plot_base64_risk_return
    })

@app.get("/input_arch", response_class=HTMLResponse)
async def input_arch(request: Request):
    plot_base64_acf = plot_acf(returns, ticker)
    return templates.TemplateResponse("input_arch.html", {
        "request": request,
        "plot_acf": plot_base64_acf
    })

@app.get("/vis_arch", response_class=HTMLResponse)
async def vis_arch(
    request: Request,
    sym_in: int = Query(..., description="Lag order of the symmetric innovation"),
    asym_in: int = Query(..., description="Lag order of the asymmetric innovation"),
    lag_vol: int = Query(..., description="Lag order of lagged volatility"),
    hor: int = Query(..., description="Horizon of forecast")
):
    plot_base64_arch, plot_base64_simsvar = eval_arch(
        ticker, returns, sym_in, asym_in, lag_vol, hor)
    return templates.TemplateResponse("vis_arch.html", {
        "request": request,
        "plot_arch": plot_base64_arch,
        "plot_simsvar": plot_base64_simsvar,
        "ticker": ticker,
        "sym_in": sym_in,
        "asym_in": asym_in,
        "lag_vol": lag_vol,
        "hor": hor
    })

@app.get("/input_lstm", response_class=HTMLResponse)
async def input_lstm(
    request: Request
    ):
    return templates.TemplateResponse("input_lstm.html", {
        "request": request
    })

@app.get("/vis_lstm", response_class=HTMLResponse)
async def vis_lstm(
    request: Request,
    t_steps: int = Query(..., description="Time steps"),
    fcst_steps: int = Query(..., description="Forecast steps"),               
):
    plot_base64_lstm = perform_lstm(stock_df, ticker, t_steps, fcst_steps)
    return templates.TemplateResponse("vis_rnn.html", {
        "request": request,
        "plot_lstm": plot_base64_lstm,
        "ticker": ticker,
        "t_steps": t_steps,
        "fcst_steps": fcst_steps
    })
