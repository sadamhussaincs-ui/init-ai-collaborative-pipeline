import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def load_data():
    print("🤖 Phase 1: Loading AI dataset...")
    # Generates a synthetic dataset for binary classification
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    return X, y

if __name__ == "__main__":
    X, y = load_data()
    print(f"Dataset successfully loaded. Features shape: {X.shape}, Labels shape: {y.shape}")
