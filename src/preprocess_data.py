import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def preprocess():
    train = pd.read_csv('../data/train.csv')
    
    # 1. Encode the Target (The part XGBoost was complaining about)
    target_le = LabelEncoder()
    train['diabetes_stage'] = target_le.fit_transform(train['diabetes_stage'])
    
    # Save this specific encoder so you can "reverse" the numbers back 
    # to text in your web app later
    joblib.dump(target_le, '../artifacts/target_encoder.pkl')

    # 2. Encode any other text columns (like Gender or Diet)
    le = LabelEncoder()
    for col in train.select_dtypes(include=['object']).columns:
        if col != 'diabetes_stage':
            train[col] = le.fit_transform(train[col])

    # 3. Scale and Save
    X = train.drop('diabetes_stage', axis=1) 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    joblib.dump(scaler, '../artifacts/scaler.pkl')
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    X_scaled_df.to_csv('../data/train_scaled.csv', index=False)
    
    # Save the updated target to a new file so the trainer can find the numbers
    train[['diabetes_stage']].to_csv('../data/train_target_encoded.csv', index=False)
    
    print("Preprocessing Complete: Target encoded to numbers [0, 1, 2, 3, 4].")

if __name__ == "__main__":
    preprocess()