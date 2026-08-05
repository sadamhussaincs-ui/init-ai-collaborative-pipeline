import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


def load_data():
    print("🤖 Phase 1: Loading AI dataset...")
    # Generates a synthetic dataset for binary classification
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    return X, y
def preprocess_data(X):
    print("⚙️ Phase 2: Preprocessing features (Scaling data)...")
    # Simple normalization simulation
    return (X - X.mean(axis=0)) / X.std(axis=0)
def train_model(X, y):
    print("🧠 Phase 2: Initializing Machine Learning Model...")
    model = LogisticRegression(max_iter=100)
    return model

if __name__ == "__main__":
    X, y = load_data()
    model = train_model(X, y) # Added by Student B
    print(f"Model initialized: {model}")


