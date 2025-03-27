import pandas as pd
import numpy as np
import statsmodels.api as sm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys, os

# Add the parent directory to path so we can import from other directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ess.dataset import Dataset
from src.ARIMAX import prepare_unified_dataset

if __name__ == "__main__":
    # Prepare the unified dataset
    data = prepare_unified_dataset()
    
    # Filter data for a specific country, e.g., Ireland
    ireland_df = data[data['Country'] == 'Ireland'].sort_values('Year').reset_index(drop=True)
    print("Ireland Data:")
    print(ireland_df[['Country', 'Year', 'GDP_Expenditure', 'impenv', 'iplylfr']])
    
    # Define the features and target variable for the model
    # Here we use 'Year', 'impenv', and 'iplylfr' as predictors.
    features = ['Year', 'impenv', 'iplylfr']
    target = 'GDP_Expenditure'
    
    # Drop rows with missing feature or target values
    ireland_df = ireland_df.dropna(subset=features + [target])
    
    X = ireland_df[features].values
    y = ireland_df[target].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=1)
    
    # Evaluate the model on the test set
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Display model summary
    model.summary()
    
    # Plot predictions vs. actual values
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual GDP Expenditure")
    plt.ylabel("Predicted GDP Expenditure")
    plt.title("Predicted vs. Actual GDP Expenditure")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.show()