import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def load_data():
    # Baseline data-loading logic (Student A setup)
    print("Loading AI pipeline dataset...")
    # Generates 100 samples with 5 features (X) and binary labels 0 or 1 (y)
    return np.random.rand(100, 5), np.random.randint(0, 2, 100)

# --- STUDENT B WORKSPACE AREA ---
def initialize_model():
    print("Initializing Machine Learning Model...")
    # Creating a simple Logistic Regression model with hyperparameters
    model = LogisticRegression(max_iter=100)
    return model

if __name__ == "__main__":
    # 1. Fetch data arrays
    X, y = load_data()
    
    # 2. Instantiate the classifier matrix
    model = initialize_model()
    
    # 3. Train the model using the dataset (Added Step)
    print("Training Machine Learning Model on dataset...")
    model.fit(X, y)
    
    # 4. Verify training performance
    accuracy = model.score(X, y)
    print(f"Model Training Complete! Training Accuracy: {accuracy * 100:.2f}%")
