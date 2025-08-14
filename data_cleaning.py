import pandas as pd

def clean_data(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.fillna({'TotalCharges': df['TotalCharges'].median()}, inplace=True)
    return df
