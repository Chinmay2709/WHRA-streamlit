'''

    EDA Visualization: Clean, Smooth, design for static graph.

    Goal: Developing static graphs, to get understanding of the data.

'''

# scripts/eda_visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_cleaned_data(filepath):

    df = pd.read_csv(filepath)
    return df

def basic_info(df):

    print('Dataset Head:')
    print(df.head())

    print('\nSummary Statistic:')
    print(df.describe())

    print('\nMissing Values:')
    print(df.isnull().sum())


# Bar Plot: To plot top 10 countries in the world.
def top_10_happienst_countries(df):

    top10 = df.sort_values( by="Life Ladder", ascending= False ).head(10)
    plt.figure(figsize = (10, 10))
    sns.barplot(x = "Life Ladder", y = "Country name", data = top10, palette='viridis')
    plt.title('Top 10 Happiest Countries (2024)')
    plt.xlabel('Happiness Score')
    plt.ylabel('Countries')
    plt.tight_layout()
    plt.show()

# Heatmap Plot: Showing correlation Heatmap.
def correlation_heatmap(df):

    nummeric_df = df.select_dtypes(include = ['float64', 'int64'])

    plt.figure(figsize = (10, 8))
    sns.heatmap(nummeric_df.corr(), annot = True, cmap = 'coolwarm', linewidths=0.5)
    plt.title("Correlation heatmap of the Factors")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    cleaned_data_path = 'Data/WHRA-cleaned_2024.csv'

    df = load_cleaned_data(cleaned_data_path)

    # Data Exploration
    basic_info(df)

    # Visualization
    top_10_happienst_countries(df)
    # correlation_heatmap(df)