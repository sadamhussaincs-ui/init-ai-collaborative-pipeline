import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def load_data():
    print("🤖 Phase 1: Loading AI dataset...")
    # Generates a synthetic dataset for binary classification
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    return X, y
def preprocess_data(X):
    print("⚙️ Phase 2: Preprocessing features (Scaling data)...")
    # Simple normalization simulation
    return (X - X.mean(axis=0)) / X.std(axis=0)

if __name__ == "__main__":
    X, y = load_data()
    X = preprocess_data(X) # Added by Student A
    print(f"Data scaled. Features shape: {X.shape}")

