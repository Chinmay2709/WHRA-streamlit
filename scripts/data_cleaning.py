'''

    Data Cleaning: Drop N/A, Null, Missing values.
    
    Goal: To Generate cleaned effective dataset.

'''

# scripts/data_cleaning.py



import pandas as pd 


def load_data(file_path):
    
    df = pd.read_csv(file_path, encoding = 'latin1')
    return df

def clean_data(df):
    
    df_cleaned = df.dropna()
    return df_cleaned

def save_clean_data(df_cleaned, output_path):
    
    df_cleaned.to_csv(output_path, index = False)
    

if __name__ == "__main__":

    raw_data_path = "Data/World-happiness-report-updated_2024.csv"
    cleaned_data_path = "Data/WHRA-cleaned_2024.csv"

    df = load_data(raw_data_path)
    print("Initial shape of Data: ", df.shape)

    df_cleaned = clean_data(df)
    print("Cleaned Data shape: ", df_cleaned.shape)

    save_clean_data(df_cleaned, cleaned_data_path)
    print("Cleaned Data Saved!")




