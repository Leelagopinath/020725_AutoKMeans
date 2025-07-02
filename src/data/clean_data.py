import pandas as pd
import sys

def clean_data(input_file):
    """Clean raw data for clustering"""
    df = pd.read_csv(input_file)
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype.kind in 'iuf':  # Numeric columns
            df[col] = df[col].fillna(df[col].median())
        else:  # Categorical columns
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Save cleaned data
    df.to_csv("cleaned.csv", index=False)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean_data.py <input_file>")
        sys.exit(1)
    clean_data(sys.argv[1])