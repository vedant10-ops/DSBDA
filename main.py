import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


def perform_eda(data):
    # Clean data
    clean_data = clean_data_function(data)

    # Summary statistics
    summary_stats = clean_data.describe()

    # Data visualization
    for column in clean_data.columns:
        if clean_data[column].dtype in ['int64', 'float64']:
            plt.figure(figsize=(8, 6))
            sns.histplot(clean_data[column], kde=True)
            plt.title(f'Distribution of {column}')
            st.pyplot()
        else:
            # For categorical data
            plt.figure(figsize=(8, 6))
            category_counts = clean_data[column].value_counts()
            if not category_counts.empty:
                sns.barplot(x=category_counts.index, y=category_counts.values)
                plt.title(f'Count of {column}')
                plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
                plt.tight_layout()  # Adjust layout to prevent overlap
                st.pyplot()
            else:
                st.write(f"No data to plot for column {column}")

    # Other analysis if needed
    # Example: correlation matrix, pairplot

    # Display summary statistics
    st.write("Summary Statistics:")
    st.write(summary_stats)

    # Display data cleaning information
    st.write("Data Cleaning Information:")
    st.write(get_data_info(clean_data))

    # Display null values
    st.write("Null Values:")
    st.write(clean_data.isnull().sum())

    # Display outliers
    st.write("Outliers:")
    st.write(find_outliers(clean_data))


def clean_data_function(data):
    # Drop duplicates
    clean_data = data.drop_duplicates()

    # Handle missing values
    clean_data.dropna(inplace=True)

    # Handle outliers (assuming z-score method)
    clean_data = clean_data[
        (np.abs(stats.zscore(clean_data.select_dtypes(include=['int64', 'float64']))) < 3).all(axis=1)]

    return clean_data


def find_outliers(data):
    # Find outliers using z-score method
    z_scores = np.abs(stats.zscore(data.select_dtypes(include=['int64', 'float64'])))
    outliers = np.where(z_scores > 3)
    return outliers


def get_data_info(data):
    # Get information about the dataframe
    info_str = "DataFrame Information:\n"
    info_str += str(data.info())
    return info_str


# Example usage
if __name__ == "__main__":
    st.title("Exploratory Data Analysis (EDA) Web App")

    # Upload file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        perform_eda(data)
