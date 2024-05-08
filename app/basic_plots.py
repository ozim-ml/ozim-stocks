from app.main import *

def perform_analysis(ticker: str, start_date: str, end_date: str):
    # Convert start_date and end_date to datetime objects

    start_date = dt.datetime.strptime(start_date, "%Y-%m-%d")
    end_date = dt.datetime.strptime(end_date, "%Y-%m-%d")

    # Load data from yfinance for selected tickers within the specified date range
    stock_df = yf.download(ticker, start=start_date, end=end_date)

    # Plot the Adjusted Closing Price
    plt.plot(stock_df.index, stock_df['Adj Close'], label='Adj Close', linewidth=1.5)
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

    return ticker, stock_df, plot_base64_adj_close

def plot_daily_returns(stock_df, ticker):

    # Calculate daily returns
    returns = 100 * stock_df['Adj Close'].pct_change().dropna()

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
    plt.tight_layout()

    # Create a BytesIO object to store the plot
    plot_bytes_acf = BytesIO()

    # Save the plot to BytesIO and encode as base64
    plot_bytes_acf.seek(0)
    plt.savefig(plot_bytes_acf, format='png')
    plot_base64_acf = base64.b64encode(plot_bytes_acf.getvalue()).decode('utf-8')

    plt.close()

    return plot_base64_acf

def plot_risk_ret(returns, ticker):

    area = np.pi * 20

    plt.scatter(returns.mean(), returns.std(), s=area)
    plt.xlabel('Expected return')
    plt.ylabel('Risk')
    plt.title(f'Risk-return of {ticker}')
    plt.tight_layout()

    for _, value in returns.items():
        x = returns.mean()
        y = returns.std()
        plt.annotate(
            ticker, xy=(x, y), xytext=(50, 50), textcoords='offset points',
            ha='right', va='bottom',
            arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3')
        )

    # Create a BytesIO object to store the plot
    plot_bytes_risk_ret = BytesIO()

    # Save the plot to BytesIO and encode as base64
    plot_bytes_risk_ret.seek(0)
    plt.savefig(plot_bytes_risk_ret, format='png')
    plot_base64_risk_ret = base64.b64encode(plot_bytes_risk_ret.getvalue()).decode('utf-8')

    plt.close()

    return plot_base64_risk_ret


