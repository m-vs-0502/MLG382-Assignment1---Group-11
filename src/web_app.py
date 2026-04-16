import os
import joblib
import numpy as np
import dash
import warnings
from dash import dcc, html, Input, Output, State, ALL

# Ignore version mismatch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- 1. Load Artifacts ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
base_path = os.path.join(root_dir, 'artifacts')

scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
xgb_model = joblib.load(os.path.join(base_path, 'xgboost_model.pkl'))
kmeans_model = joblib.load(os.path.join(base_path, 'kmeans_model.pkl'))

# --- 2. Configuration ---
# 10 Dashboard Inputs
DASHBOARD_FEATURES = [
    'hba1c', 'glucose_fasting', 'bmi', 'waist_to_hip_ratio', 'diet_score',
    'Age', 'cholesterol_total', 'systolic_bp', 'triglycerides', 'physical_activity_minutes_per_week'
]

# K-MEANS EXPECTS 19 (Health Metrics + Scaler)
KMEANS_FEATURES = [
    'Age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 
    'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi', 
    'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate', 
    'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides', 
    'glucose_fasting', 'glucose_postprandial', 'insulin_level', 'hba1c'
]

# XGBOOST EXPECTS 39 (Full Encoded Dataset)
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
    'Age': 50.14, 'gender': 0.47, 'alcohol_consumption_per_week': 2.01, 
    'physical_activity_minutes_per_week': 119.05, 'diet_score': 5.99, 
    'sleep_hours_per_day': 7.00, 'screen_time_hours_per_day': 5.99, 
    'family_history_diabetes': 0.22, 'hypertension_history': 0.25, 
    'cardiovascular_history': 0.08, 'bmi': 25.63, 'waist_to_hip_ratio': 0.86, 
    'systolic_bp': 115.77, 'diastolic_bp': 75.23, 'heart_rate': 69.62, 
    'cholesterol_total': 185.98, 'hdl_cholesterol': 54.05, 'ldl_cholesterol': 102.99, 
    'triglycerides': 121.53, 'glucose_fasting': 111.07, 'glucose_postprandial': 159.94, 
    'insulin_level': 9.05, 'hba1c': 6.52, 'ethnicity_Black': 0.18, 
    'ethnicity_Hispanic': 0.20, 'ethnicity_Other': 0.05, 'ethnicity_White': 0.45, 
    'education_level_Highschool': 0.45, 'education_level_No formal': 0.05, 
    'education_level_Postgraduate': 0.15, 'income_level_Low': 0.15, 
    'income_level_Lower-Middle': 0.25, 'income_level_Middle': 0.35, 
    'income_level_Upper-Middle': 0.20, 'employment_status_Retired': 0.22, 
    'employment_status_Student': 0.06, 'employment_status_Unemployed': 0.12, 
    'smoking_status_Former': 0.20, 'smoking_status_Never': 0.60
}

# --- 3. App Setup ---
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

app.layout = html.Div([
    html.Div([
        html.H2("Diabetes Risk Diagnostic", style={'textAlign': 'center', 'color': '#2c3e50'}),
        html.Hr(),
        html.Div([
            html.Div([
                html.Label(f.replace('_', ' ').title(), style={'fontWeight': 'bold'}),
                dcc.Input(id={'type': 'in', 'index': f}, type='number', value=round(MEANS[f], 2), style={'width': '100%'})
            ], className="four columns", style={'padding': '10px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px', 'margin': '5px'}) 
            for f in DASHBOARD_FEATURES
        ], className="row"),
        html.Br(),
        html.Button('Run Analysis', id='submit-val', n_clicks=0, className="button-primary", style={'width': '100%'}),
        html.Div(id='prediction-output', style={'padding': '20px', 'textAlign': 'center'})
    ], style={'maxWidth': '1000px', 'margin': 'auto', 'padding': '30px', 'backgroundColor': '#fff', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'})
], style={'backgroundColor': '#f0f2f5', 'minHeight': '100vh', 'paddingTop': '50px'})

@app.callback(
    Output('prediction-output', 'children'),
    Input('submit-val', 'n_clicks'),
    State({'type': 'in', 'index': ALL}, 'value'),
    State({'type': 'in', 'index': ALL}, 'id')
)
def update_output(n_clicks, values, ids):
    if n_clicks > 0:
        user_input = {item['index']: val for item, val in zip(ids, values)}
        
        try:
            # 1. K-Means Path (19 features, scaled)
            km_raw = np.array([[user_input.get(f, MEANS[f]) for f in KMEANS_FEATURES]])
            km_scaled = scaler.transform(km_raw)
            cluster = kmeans_model.predict(km_scaled)[0]
            
            # 2. XGBoost Path (39 features, unscaled)
            xgb_raw = np.array([[user_input.get(f, MEANS[f]) for f in XGB_FEATURES]])
            risk_score = xgb_model.predict(xgb_raw)[0]
            
            risk_label = "High Risk" if risk_score == 1 else "Low Risk"
            color = "#e74c3c" if risk_score == 1 else "#27ae60"

            return html.Div([
                html.H3(f"Prediction: {risk_label}", style={'color': color}),
                html.P(f"Assigned Lifestyle Cluster: {cluster}", style={'fontSize': '1.2em'})
            ], style={'border': f'2px solid {color}', 'borderRadius': '10px', 'padding': '20px'})
            
        except Exception as e:
            return html.Div(f"Internal processing error: {str(e)}", style={'color': 'red', 'fontWeight': 'bold'})
    
    return "Enter clinical data and click Run Analysis."

if __name__ == '__main__':
    app.run_server(debug=True)
