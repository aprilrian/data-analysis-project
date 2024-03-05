import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def load_data():
    data = pd.read_csv('main_data.csv')
    data['dteday'] = pd.to_datetime(data['dteday'])
    return data

def display_time_series(data, column='total_rentals', title='Time Series Analysis'):
    fig = px.line(data, x='dteday', y=column, title=title)
    return fig

def comparative_analysis(data, year1, year2, column='total_rentals'):
    data1 = data[data['year'] == year1]
    data2 = data[data['year'] == year2]
    fig = make_subplots(rows=2, cols=1, subplot_titles=[f'{year1} {column.title()}', f'{year2} {column.title()}'])
    fig.add_trace(go.Scatter(x=data1['dteday'], y=data1[column], name=f'{year1}'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data2['dteday'], y=data2[column], name=f'{year2}'), row=2, col=1)
    fig.update_layout(height=600, title_text=f'Comparative Analysis: {column.title()}')
    return fig

def main():
    st.title('Dicoding Bike Sharing Dashboard')
    st.caption('This dashboard is created for the final submission of the "Belajar Pengembangan Machine Learning" course on Dicoding.')
    st.caption('The dataset used in this dashboard is the "Bike Sharing Dataset" from the UCI Machine Learning Repository.')
    st.caption('The dataset contains bike sharing data from 2011 to 2012, and the goal of this dashboard is to provide an interactive analysis of the dataset.')
    st.caption('The dashboard is created using Streamlit and Plotly.')

    data = load_data()

    # Sidebar for user inputs
    analysis_type = st.sidebar.selectbox("Select Analysis Type", ("Overview", "Time Series Analysis", "Comparative Analysis", "Distribution Analysis"))

    if analysis_type == "Overview":
        st.header("Dataset Overview")
        st.write(data.describe())
        st.dataframe(data.head())

    elif analysis_type == "Time Series Analysis":
        st.header("Time Series Analysis")
        metric = st.selectbox("Select Metric", data.columns[2:], index=data.columns.get_loc("total_rentals")-2)
        fig = display_time_series(data, metric, f'Time Series - {metric}')
        st.plotly_chart(fig)

    elif analysis_type == "Comparative Analysis":
        st.header("Comparative Analysis Between Years")
        year_options = sorted(data['year'].unique())
        year1, year2 = st.select_slider("Select Two Years to Compare", options=year_options, value=(year_options[0], year_options[-1]))
        metric = st.selectbox("Select Metric for Comparison", data.columns[2:], index=data.columns.get_loc("total_rentals")-2)
        fig = comparative_analysis(data, year1, year2, metric)
        st.plotly_chart(fig)

    elif analysis_type == "Distribution Analysis":
        st.header("Distribution Analysis")
        metric = st.selectbox("Select Metric to Analyze Distribution", data.columns[2:], index=data.columns.get_loc("total_rentals")-2)
        fig = px.histogram(data, x=metric, nbins=50, marginal="box")
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
