import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import os

def load_data(filepath):
    """Load dataset and replace '...' with NaN."""
    df = pd.read_csv(filepath)
    df.replace('...', np.nan, inplace=True)
    return df

def remove_geographical_dividers(df):
    """Remove rows that contain no numerical data."""
    return df[df.iloc[:, 1:].notna().any(axis=1)]

def clean_percentage_columns(df):
    """Convert percentage strings to float values."""
    for col in df.columns[1:]:
        df[col] = df[col].astype(str).str.replace('%', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to float, forcing errors to NaN
    return df

def impute_missing_values(df, method="linear", n_neighbors=5):
    """Impute missing values using specified method, row-wise."""
    df = clean_percentage_columns(df)
    numerical_data = df.iloc[:, 1:]  # Ensure numerical columns
    
    if method == "mean":
        df.iloc[:, 1:] = numerical_data.T.fillna(numerical_data.T.mean()).T  # Row-wise mean imputation
    elif method == "median":
        df.iloc[:, 1:] = numerical_data.T.fillna(numerical_data.T.median()).T  # Row-wise median imputation
    elif method == "knn":
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df.iloc[:, 1:] = imputer.fit_transform(numerical_data)  # KNN naturally imputes row-wise
    elif method == "linear":
        df.iloc[:, 1:] = numerical_data.interpolate(method='linear', axis=1),  # Linear interpolation row-wise, add ,limit_direction='both' to bracket for extrapolation
    elif method == "polynomial":
        df.iloc[:, 1:] = numerical_data.interpolate(method='polynomial', order=2, axis=1)  # Quadratic interpolation
    elif method == "spline":
        df.iloc[:, 1:] = numerical_data.interpolate(method='spline', order=3, axis=1)  # Cubic spline interpolation
    elif method == "ffill":
        df.iloc[:, 1:] = numerical_data.fillna(method='ffill', axis=1)  # Forward fill row-wise
    elif method == "bfill":
        df.iloc[:, 1:] = numerical_data.fillna(method='bfill', axis=1)  # Backward fill row-wise
    else:
        raise ValueError("Invalid imputation method. Choose from 'mean', 'median', 'knn', 'linear', 'polynomial', 'spline', 'ffill', or 'bfill'.")
    
    return df

def save_data(df, filepath):
    """Save the cleaned dataset with '_imputed' suffix."""
    filename, ext = os.path.splitext(filepath)
    new_filepath = f"{filename}_imputed{ext}"
    df.to_csv(new_filepath, index=False)
    print(f"File saved as: {new_filepath}")

def main(filepath, method="linear", n_neighbors=5):
    df = load_data(filepath)
    df = remove_geographical_dividers(df)
    df = impute_missing_values(df, method, n_neighbors)
    save_data(df, filepath)

# Example usage
filepath = "/Users/macbook/Library/Mobile Documents/com~apple~CloudDocs/Desktop/University Engmaths/Applied data science/ADS Coursework/Percentage_Share_of_GDP_Military_Expenditure.csv"  # Replace with actual file path
method = "linear"  # Change to "median", "knn", "polynomial", "spline", etc. as needed
n_neighbors = 5  # Change if using KNN imputation
main(filepath, method, n_neighbors)
