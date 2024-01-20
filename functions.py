import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import least_squares
from statsmodels.nonparametric.smoothers_lowess import lowess

DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/"
    "master/csse_covid_19_data/csse_covid_19_time_series/"
    "time_series_covid19_{kind}_{group}.csv"
)

GROUPS = "world", "usa"
KINDS = "deaths", "cases"


# Function 1
def download_data(group, kind):
    """
    Fetches and returns COVID-19 data from the John Hopkins GitHub repository.
    Selects data type ('deaths' or 'cases') and scope ('world' or 'usa').

    Parameters
    ----------
    group : str
        'world' for global data or 'usa' for US data.
    kind : str
        'deaths' for death data or 'cases' for case data.

    Returns
    -------
    DataFrame
        Pandas DataFrame with the requested data.
    """
    group_change_dict = {"world": "global", "usa": "US"}
    kind_change_dict = {"deaths": "deaths", "cases": "confirmed"}
    group = group_change_dict[group]
    kind = kind_change_dict[kind]
    return pd.read_csv(DOWNLOAD_URL.format(kind=kind, group=group))


# Function 2
def read_all_data():
    """
    Downloads all data combinations (world/usa and deaths/cases) from the repository.

    Returns
    -------
    dict
        Dictionary of DataFrames, keyed by "{group}_{kind}".
    """
    data = {}
    for group in GROUPS:
        for kind in KINDS:
            df = download_data(group, kind)
            data[f"{group}_{kind}"] = df
    return data


# Function 3
def write_data(data, directory, **kwargs):
    """
    Saves each DataFrame in 'data' to CSV files in the specified directory.

    Parameters
    ----------
    data : dict
        Dictionary of DataFrames to save.
    directory : str
        Target directory for CSV files.

    Returns
    -------
    None
    """
    for name, df in data.items():
        df.to_csv(f"{directory}/{name}.csv", **kwargs)


# Function 4
def read_local_data(group, kind, directory):
    """
    Reads a specific CSV file as a DataFrame from a given directory.

    Parameters
    ----------
    group : str
        'world' or 'usa'.
    kind : str
        'deaths' or 'cases'.
    directory : str
        Directory path to read the file from.

    Returns
    -------
    DataFrame
    """
    return pd.read_csv(f"{directory}/{group}_{kind}.csv")


# Function 5
def run():
    """
    Executes data loading and transformation steps for all data combinations.

    Returns
    -------
    dict
        Dictionary of transformed DataFrames.
    """
    data = {}
    for group in GROUPS:
        for kind in KINDS:
            df = read_local_data(group, kind, "data/raw")
            data[f"{group}_{kind}"] = df
    return data


# Function 6
def select_columns(df):
    """
    Filters the DataFrame to include only relevant columns for analysis.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame to filter.

    Returns
    -------
    DataFrame
    """
    cols = df.columns

    # we don't need to know the group since the world and usa have
    # different column names for their areas
    areas = ["Country/Region", "Province_State"]
    is_area = cols.isin(areas)

    # date columns are the only ones with two slashes
    has_two_slashes = cols.str.count("/") == 2
    filt = is_area | has_two_slashes
    return df.loc[:, filt]


# Function 7
def run2():
    """
    Executes a sequence of data loading, transformation, and column selection steps.

    Returns
    -------
    dict
        Dictionary of processed DataFrames.
    """
    data = {}
    for group in GROUPS:
        for kind in KINDS:
            df = read_local_data(group, kind, "data/raw")
            df = select_columns(df)
            data[f"{group}_{kind}"] = df
    return data


# Function 8
REPLACE_AREA = {
    "Korea, South": "South Korea",
    "Taiwan*": "Taiwan",
    "Burma": "Myanmar",
    "Holy See": "Vatican City",
    "Diamond Princess": "Cruise Ship",
    "Grand Princess": "Cruise Ship",
    "MS Zaandam": "Cruise Ship",
}


def update_areas(df):
    """
    Updates area names in the DataFrame using a predefined mapping.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    DataFrame
    """
    area_col = df.columns[0]
    df[area_col] = df[area_col].replace(REPLACE_AREA)
    return df


# Function 9
def run3():
    """
    Executes data loading, transformation, column selection, and area update steps.

    Returns
    -------
    dict
        Dictionary of fully processed DataFrames.
    """
    data = {}
    for group in GROUPS:
        for kind in KINDS:
            df = read_local_data(group, kind, "data/raw")
            df = select_columns(df)
            df = update_areas(df)
            data[f"{group}_{kind}"] = df
    return data


# Function 10
def group_area(df):
    """
    Aggregates data by area, summing up all values.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    DataFrame
    """
    grouping_col = df.columns[0]
    return df.groupby(grouping_col).sum()


# Function 11
def run4():
    """
    Carries out a complete data processing pipeline including grouping by area.

    Returns
    -------
    dict
        Dictionary of aggregated DataFrames.
    """
    data = {}
    for group in GROUPS:
        for kind in KINDS:
            df = read_local_data(group, kind, "data/raw")
            df = select_columns(df)
            df = update_areas(df)
            df = group_area(df)
            data[f"{group}_{kind}"] = df
    return data


# Function 12
def transpose_to_ts(df):
    """
    Transposes the DataFrame and converts the index to datetime format.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    DataFrame
    """
    df = df.T
    df.index = pd.to_datetime(df.index)
    return df


# Function 13
def run5():
    """
    Executes a comprehensive data processing pipeline, including transposition to timeseries.

    Returns
    -------
    dict
        Dictionary of processed timeseries DataFrames.
    """
    data = {}
    for group in GROUPS:
        for kind in KINDS:
            df = read_local_data(group, kind, "data/raw")
            df = select_columns(df)
            df = update_areas(df)
            df = group_area(df)
            df = transpose_to_ts(df)
            data[f"{group}_{kind}"] = df
    return data


# Exercise 14
def fix_bad_data(df):
    """
    Corrects any anomalies in the data where daily counts decrease.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    DataFrame
    """
    mask = df < df.cummax()
    df = df.mask(mask).interpolate().round(0).astype("int64")
    return df


# Exercise 15
def run6():
    """
    Executes a complete data processing and cleaning pipeline.

    Returns
    -------
    dict
        Dictionary of cleaned and processed DataFrames.
    """
    data = {}
    for group in GROUPS:
        for kind in KINDS:
            df = read_local_data(group, kind, "data/raw")
            df = select_columns(df)
            df = update_areas(df)
            df = group_area(df)
            df = transpose_to_ts(df)
            df = fix_bad_data(df)
            data[f"{group}_{kind}"] = df
    return data


# Function 16

class PrepareData:
    def __init__(self, download_new=True):
        self.download_new = download_new

# Function 17
def smooth(s, n):
    """
    Smooths the data series using LOWESS.

    Parameters
    ----------
    s : Series
        Time series data.
    n : int
        Number of points for LOWESS.

    Returns
    -------
    Series
    """
    if s.values[0] == 0:
        # Filter the data if the first value is 0
        last_zero_date = s[s == 0].index[-1]
        s = s.loc[last_zero_date:]
        s_daily = s.diff().dropna()
    else:
        # If first value not 0, use it to fill in the
        # first missing value
        s_daily = s.diff().fillna(s.iloc[0])

    # Don't smooth data with less than 15 values
    if len(s_daily) < 15:
        return s

    y = s_daily.values
    frac = n / len(y)
    x = np.arange(len(y))
    y_pred = lowess(y, x, frac=frac, is_sorted=True, return_sorted=False)
    s_pred = pd.Series(y_pred, index=s_daily.index).clip(0)
    s_pred_cumulative = s_pred.cumsum()
    last_actual = s.values[-1]
    last_smoothed = s_pred_cumulative.values[-1]
    s_pred_cumulative *= last_actual / last_smoothed
    return s_pred_cumulative


def optimize_func(params, x, y, model):
    """
    Function to be passed as first argument to least_squares

    Parameters
    ----------
    params : sequence of parameter values for model

    x : x-values from data

    y : y-values from data

    model : function to be evaluated at x with params

    Returns
    -------
    Error between function and actual data
    """
    y_pred = model(x, *params)
    error = y - y_pred
    return error


# Function 18
def train_model(s, last_date, model, bounds, p0, **kwargs):
    """
    Trains a model on the series up to the specified date.

    Parameters
    ----------
    s : Series
    last_date : str
    model : function
    bounds : tuple
    p0 : tuple
    kwargs : dict

    Returns
    -------
    numpy array
    """
    y = s.loc[:last_date]
    n_train = len(y)
    x = np.arange(n_train)
    res = least_squares(optimize_func, p0, args=(x, y, model), bounds=bounds, **kwargs)
    return res.x


# Function 19
def get_daily_pred(model, params, n_train, n_pred):
    """
    Generates daily predictions using the trained model.

    Parameters
    ----------
    model : function
    params : array
    n_train : int
    n_pred : int

    Returns
    -------
    array
    """
    x_pred = np.arange(n_train - 1, n_train + n_pred)
    y_pred = model(x_pred, *params)
    y_pred_daily = np.diff(y_pred)
    return y_pred_daily


# Function 20
def get_cumulative_pred(last_actual_value, y_pred_daily, last_date):
    """
    Calculates cumulative predictions from daily forecasts.

    Parameters
    ----------
    last_actual_value : int
    y_pred_daily : array
    last_date : str

    Returns
    -------
    Series
    """
    first_pred_date = pd.Timestamp(last_date) + pd.Timedelta("1D")
    n_pred = len(y_pred_daily)
    index = pd.date_range(start=first_pred_date, periods=n_pred)
    return pd.Series(y_pred_daily.cumsum(), index=index) + last_actual_value


# Function 21
def plot_prediction(s, s_pred, title=""):
    """
    Plots original and predicted values.

    Parameters
    ----------
    s : Series
    s_pred : Series
    title : str

    Returns
    -------
    None
    """
    last_pred_date = s_pred.index[-1]
    ax = s[:last_pred_date].plot(label="Actual")
    s_pred.plot(label="Predicted")
    ax.legend()
    ax.set_title(title)


# Function 22
def predict_all(s, start_date, last_date, n_smooth, n_pred, model, bounds, p0, title="", **kwargs):
    """
    Full pipeline to smooth, train, predict, and plot a data series.

    Parameters
    ----------
    s : Series
    start_date, last_date : str
    n_smooth, n_pred : int
    model : function
    bounds, p0 : tuple
    title : str

    Returns
    -------
    tuple
    """
    # Smooth up to the last date
    s_smooth = smooth(s[:last_date], n=n_smooth)

    # Filter for the start of the modeling period
    s_smooth = s_smooth[start_date:]
    params = train_model(
        s_smooth, last_date=last_date, model=model, bounds=bounds, p0=p0, **kwargs
    )
    n_train = len(s_smooth)
    y_daily_pred = get_daily_pred(model, params, n_train, n_pred)
    last_actual_value = s.loc[last_date]
    s_cumulative_pred = get_cumulative_pred(last_actual_value, y_daily_pred, last_date)
    plot_prediction(s[start_date:], s_cumulative_pred, title=title)
    return params, y_daily_pred


# Function 23
def logistic_func(x, L, x0, k):
    """
    Logistic function for modeling data.

    Parameters
    ----------
    x : array
    L, x0, k : float

    Returns
    -------
    array
    """
    return L / (1 + np.exp(-k * (x - x0)))


# Function 24
def logistic_guess_plot(s, L, x0, k):
    """
    Plots data with a logistic function estimate.

    Parameters
    ----------
    s : Series
    L, x0, k : float

    Returns
    -------
    None
    """
    x = np.arange(len(s))
    y = logistic_func(x, L, x0, k)
    s_guess = pd.Series(y, index=s.index)
    s.plot()
    s_guess.plot()


# My guess
# logistic_guess_plot(italyc, 220_000, 50, .1)

# Exercise 25
def plot_ks(s, ks, L, x0):
    """
    Visualizes various logistic curves to assist in parameter tuning.

    Parameters
    ----------
    s : Series
    ks : list of floats
    L, x0 : float

    Returns
    -------
    None
    """
    start = s.index[0]
    index = pd.date_range(start, periods=2 * x0)
    x = np.arange(len(index))
    s.plot(label="smoothed", lw=3, title=f"L={L:,} $x_0$={x0}", zorder=3)
    for k in ks:
        y = logistic_func(x, L, x0, k)
        y = pd.Series(y, index=index)
        y.plot(label=f"k={k}").legend()


# Function 26
def area_bar_plot(df, group, area, kind, last_date, first_pred_date):
    """
    Generates a bar plot of actual vs. predicted values for an area.

    Parameters
    ----------
    df : DataFrame
    group, area : str
    kind : str
    last_date, first_pred_date : str

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = df.query("group == @group and area == @area").set_index("date")
    df_actual = df[:last_date]
    df_pred = df[first_pred_date:]
    fig = go.Figure()
    fig.add_bar(x=df_actual.index, y=df_actual[kind], name="actual")
    fig.add_bar(x=df_pred.index, y=df_pred[kind], name="prediction")
    return fig


    
