from app.main import *

def eval_arch(ticker, returns, sym_in: int, asym_in: int, lag_vol: int, hor: int):
    with mlflow.start_run():

        # Model evaluation and fitting
        am = arch_model(returns, vol="GARCH", p=sym_in, o=asym_in, q=lag_vol, dist="normal")
        res = am.fit(update_freq=5)

        mlflow.log_param("p", sym_in)
        mlflow.log_metric("AIC", res.aic)

        # Provisional model logging (in order to fully log the arch model, customized function is needed)
        mlflow.sklearn.log_model(res, "arch_models", signature=None)

    # Creating simulation forecast
    horizon = hor
    forecasts = res.forecast(horizon=horizon, method="simulation", reindex=False)
    sims = forecasts.simulations

    # Plotting simulation forecast
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

    # Plotting simulations variances
    sns.boxplot(data=sims.variances[-1])

    # Create a BytesIO object to store the plot
    plot_bytes_simsvar = BytesIO()

    # Save the plot to BytesIO and encode as base64
    plot_bytes_simsvar.seek(0)
    plt.savefig(plot_bytes_simsvar, format='png')
    plot_base64_simsvar = base64.b64encode(plot_bytes_simsvar.getvalue()).decode('utf-8')

    plt.close()

    return plot_base64_arch, plot_base64_simsvar
