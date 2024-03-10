import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

sns.set(style='dark')


# Helper functions
def create_daily_df(df):
    daily = df.resample(rule='D', on='dteday').agg({
        "cnt": "sum",
        "casual": "sum",
        "registered": "sum"
    })
    daily = daily.reset_index()

    return daily


def create_daily_clasified_df(df):
    daily_clasified = df.groupby('dteday').agg({
        'casual': 'sum',
        'registered': 'sum'
    }).reset_index()

    return daily_clasified


def create_dist_clasify_df(df):
    df['total'] = df['casual'] + df['registered']
    quantiles = df['total'].quantile([0.33, 0.66])

    def classify_usage(row):
        if row['total'] <= quantiles.iloc[0]:
            return 'Low'
        elif row['total'] <= quantiles.iloc[1]:
            return 'Medium'
        else:
            return 'High'

    df['usage_class'] = df.apply(classify_usage, axis=1)
    df['weekday'] = df['dteday'].dt.weekday

    return df


# Load cleaned data
all_df = pd.read_csv('./submission/dashboard/main_data.csv')

all_df.sort_values(by='dteday', inplace=True)
all_df.reset_index(inplace=True)
all_df['dteday'] = pd.to_datetime(all_df['dteday'])


# Filter data
min_date = all_df['dteday'].min()
max_date = all_df['dteday'].max()


# Sidebar
with st.sidebar:
    # Logo
    st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png")

    # Date range
    start_date, end_date = st.date_input(
        label='Date Range', min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

    # Reset button
    if st.button('Reset Date Range'):
        start_date = min_date
        end_date = max_date

main_df = all_df[(all_df['dteday'] >= str(start_date)) &
                 (all_df['dteday'] <= str(end_date))]


# Prepare dataframes
daily_df = create_daily_df(main_df)
daily_clasified_df = create_daily_clasified_df(main_df)
daily_dist_clasified_df = create_dist_clasify_df(daily_clasified_df)
time_series_df = daily_dist_clasified_df


# Title and subtitle
st.header('Dicoding Bike Sharing Dashboard')
st.subheader('Date Range: \n{} — {}'.format(start_date.strftime('%B %d, %Y'), end_date.strftime('%B %d, %Y')))

st.metric("Total Rides", daily_df['cnt'].sum())


# Plot daily rides
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    daily_df["dteday"],
    daily_df["cnt"],
    marker='o',
    linewidth=2,
    color="#90CAF9"
)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)

st.pyplot(fig)


# Daily rides classified
col1, col2 = st.columns(2)

with col1:
    st.metric("Total Casual Rides", daily_df['casual'].sum())

with col2:
    st.metric("Total Registered Rides", daily_df['registered'].sum())


# Plot daily rides classified
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    daily_clasified_df["dteday"],
    daily_clasified_df["casual"],
    marker='o',
    linewidth=2,
    color="#90CAF9",
    label='Casual'
)
ax.plot(
    daily_clasified_df["dteday"],
    daily_clasified_df["registered"],
    marker='o',
    linewidth=2,
    color="#FFAB91",
    label='Registered'
)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)
ax.legend()

st.pyplot(fig)


# Daily rides classified by usage
st.subheader('Rides Classified by Usage')

scol1, scol2, scol3 = st.columns(3)

with scol1:
    st.metric("Low Usage", daily_dist_clasified_df[daily_dist_clasified_df['usage_class'] == 'Low']['total'].sum())

with scol2:
    st.metric("Medium Usage", daily_dist_clasified_df[daily_dist_clasified_df['usage_class'] == 'Medium']['total'].sum())

with scol3:
    st.metric("High Usage", daily_dist_clasified_df[daily_dist_clasified_df['usage_class'] == 'High']['total'].sum())

fig, ax = plt.subplots(figsize=(16, 8))
sns.countplot(x='weekday', hue='usage_class', data=daily_dist_clasified_df)
plt.legend(title='Usage Class')

days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
ax.set_xticklabels(days_of_week)
ax.set_xlabel('')

st.pyplot(fig)


# Time series
st.subheader('Time Series')
st.metric("Average Rides", time_series_df['total'].mean())

tcol1, tcol2 = st.columns(2)

with tcol1:
    st.metric("Minimum", time_series_df['total'].min())

with tcol2:
    st.metric("Maximum", time_series_df['total'].max())

if len(time_series_df['total']) >= 14:
    time_series_df.set_index('dteday', inplace=True)
    decomposition = seasonal_decompose(time_series_df['total'], model='additive')
    fig = decomposition.plot()
    plt.rcParams.update(
        {'axes.titlesize': 'large', 'axes.labelsize': 'medium', 'xtick.labelsize': 'small', 'ytick.labelsize': 'small'})

    st.pyplot(fig)
else:
    st.write(f"Data tidak cukup untuk dekomposisi musiman: hanya ada {len(time_series_df['total'])} dari 14 observasi "
             f"yang tersedia. Coba untuk memperluas rentang tanggal.")
