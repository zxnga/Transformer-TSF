import matplotlib.pyplot as plt
import pandas as pd

def plot(ts_index, test_data, forecasts, prediction_length, filename=None):
    fig, ax = plt.subplots()

    index = pd.period_range(
        start=test_data[ts_index][FieldName.START],
        periods=len(test_data[ts_index][FieldName.TARGET]),
        freq=freq,
    ).to_timestamp()

    # Major ticks every half year, minor ticks every month,
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    # # ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H'))

    ax.plot(
        index[-2*prediction_length:],
        test_data[ts_index]["target"][-2*prediction_length:],
        label="actual",
    )

    plt.plot(
        index[-prediction_length:],
        np.median(forecasts[ts_index], axis=0),
        label="median",
    )

    plt.fill_between(
        index[-prediction_length:],
        forecasts[ts_index].mean(0) - forecasts[ts_index].std(axis=0),
        forecasts[ts_index].mean(0) + forecasts[ts_index].std(axis=0),
        alpha=0.3,
        interpolate=True,
        label="+/- 1-std",
    )
    plt.legend()
    plt.gcf().autofmt_xdate()

    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()