# Ozim-stocks - web app for stock price analysis and forecasting
## Introduction 
This is a simple web app made with FastAPI. It uses yfinance library which allows to download stock market data from Yahoo! Finance API.
Web app creates plots of historical stock prices, evaluates the GARCH model and forecasts the volatility of price.
Currently, this web app performs plotting of:
1. Adjusted closing price
2. Daily returns
3. Autocorrelation of squared daily returns
4. Forecasted volatility of daily returns.
