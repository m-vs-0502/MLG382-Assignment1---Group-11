import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.cluster import KMeans

def train():
    # Load the scaled features and the NEW encoded target
    X_scaled = pd.read_csv('../data/train_scaled.csv')
    y = pd.read_csv('../data/train_target_encoded.csv')['diabetes_stage']
    
    # Train XGBoost
    model_xgb = XGBClassifier(random_state=42)
    model_xgb.fit(X_scaled, y)
    joblib.dump(model_xgb, '../artifacts/model_1.pkl')
    
    # Train K-Means
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)
    joblib.dump(kmeans, '../artifacts/model_2.pkl')
    
    print("Training Complete: model_1 and model_2 saved successfully.")

if __name__ == "__main__":
    train()