import os
import joblib
import numpy as np
import dash
from dash import dcc, html, Input, Output, State

# --- 1. Load Artifacts ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
base_path = os.path.join(root_dir, 'artifacts')

scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
xgb_model = joblib.load(os.path.join(base_path, 'xgboost_model.pkl'))
kmeans_model = joblib.load(os.path.join(base_path, 'kmeans_model.pkl'))

# --- 2. Configuration ---
DASHBOARD_FEATURES = [
    'hba1c', 'glucose_fasting', 'bmi', 'waist_to_hip_ratio', 'diet_score',
    'Age', 'cholesterol_total', 'systolic_bp', 'triglycerides', 'physical_activity_minutes_per_week'
]

KMEANS_FEATURES = [
    'Age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 
    'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi', 
    'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate', 
    'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides', 
    'glucose_fasting', 'glucose_postprandial', 'insulin_level', 'hba1c'
]

XGB_FEATURES = [
    'Age', 'gender', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 
    'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 'family_history_diabetes', 
    'hypertension_history', 'cardiovascular_history', 'bmi', 'waist_to_hip_ratio', 
    'systolic_bp', 'diastolic_bp', 'heart_rate', 'cholesterol_total', 'hdl_cholesterol', 
    'ldl_cholesterol', 'triglycerides', 'glucose_fasting', 'glucose_postprandial', 
    'insulin_level', 'hba1c', 'ethnicity_Black', 'ethnicity_Hispanic', 'ethnicity_Other', 
    'ethnicity_White', 'education_level_Highschool', 'education_level_No formal', 
    'education_level_Postgraduate', 'income_level_Low', 'income_level_Lower-Middle', 
    'income_level_Middle', 'income_level_Upper-Middle', 'employment_status_Retired', 
    'employment_status_Student', 'employment_status_Unemployed', 'smoking_status_Former', 
    'smoking_status_Never'
]

MEANS = {
    'Age': 50.145, 'gender': 0.477, 'alcohol_consumption_per_week': 2.009, 
    'physical_activity_minutes_per_week': 119.048, 'diet_score': 5.989, 
    'sleep_hours_per_day': 6.995, 'screen_time_hours_per_day': 5.993, 
    'family_history_diabetes': 0.217, 'hypertension_history': 0.251, 
    'cardiovascular_history': 0.079, 'bmi': 25.628, 'waist_to_hip_ratio': 0.856, 
    'systolic_bp': 115.765, 'diastolic_bp': 75.228, 'heart_rate': 69.617, 
    'cholesterol_total': 185.982, 'hdl_cholesterol': 54.047, 'ldl_cholesterol': 102.986, 
    'triglycerides': 121.528, 'glucose_fasting': 111.066, 'glucose_postprandial': 159.942, 
    'insulin_level': 9.052, 'hba1c': 6.518, 'ethnicity_Black': 0.179, 
    'ethnicity_Hispanic': 0.201, 'ethnicity_Other': 0.051, 'ethnicity_White': 0.449, 
    'education_level_Highschool': 0.448, 'education_level_No formal': 0.050, 
    'education_level_Postgraduate': 0.149, 'income_level_Low': 0.147, 
    'income_level_Lower-Middle': 0.251, 'income_level_Middle': 0.352, 
    'income_level_Upper-Middle': 0.198, 'employment_status_Retired': 0.217, 
    'employment_status_Student': 0.061, 'employment_status_Unemployed': 0.118, 
    'smoking_status_Former': 0.200, 'smoking_status_Never': 0.598
}

# --- 3. App Setup ---
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Diabetes Risk & Lifestyle Dashboard"),
    html.Div([
        html.Div([
            html.Label(feat.replace('_', ' ').title()),
            dcc.Input(id={'type': 'in', 'id': feat}, type='number', value=round(MEANS[feat], 2))
        ], style={'margin': '10px'}) for feat in DASHBOARD_FEATURES
    ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr'}),
    html.Button("Analyze", id="run-btn", n_clicks=0),
    html.Div(id="results-out", style={'marginTop': '20px', 'fontSize': '20px'})
])

@app.callback(
    Output("results-out", "children"),
    Input("run-btn", "n_clicks"),
    [State({'type': 'in', 'id': feat}, "value") for feat in DASHBOARD_FEATURES]
)
def run_analysis(n, *values):
    if n == 0: return ""
    user_data = dict(zip(DASHBOARD_FEATURES, values))
    
    # Path 1: K-Means (19 features, scaled)
    km_input = [user_data.get(f, MEANS[f]) for f in KMEANS_FEATURES]
    km_scaled = scaler.transform([km_input])
    cluster = kmeans_model.predict(km_scaled)[0]
    
    # Path 2: XGBoost (39 features, unscaled)
    xgb_input = [user_data.get(f, MEANS[f]) for f in XGB_FEATURES]
    risk_pred = xgb_model.predict([xgb_input])[0]
    
    return f"Risk Prediction: {risk_pred} | Lifestyle Cluster: {cluster}"

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
