import os
import joblib
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, ALL

# --- 1. Load Artifacts ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
base_path = os.path.join(root_dir, 'artifacts')

scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
xgb_model = joblib.load(os.path.join(base_path, 'xgboost_model.pkl'))
kmeans_model = joblib.load(os.path.join(base_path, 'kmeans_model.pkl'))

# --- 2. Configuration ---
# The 10 features for your UI
DASHBOARD_FEATURES = [
    'hba1c', 'glucose_fasting', 'bmi', 'waist_to_hip_ratio', 'diet_score',
    'Age', 'cholesterol_total', 'systolic_bp', 'triglycerides', 'physical_activity_minutes_per_week'
]

# The 19 features the Scaler/KMeans expects
KMEANS_FEATURES = [
    'Age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 
    'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi', 
    'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate', 
    'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides', 
    'glucose_fasting', 'glucose_postprandial', 'insulin_level', 'hba1c'
]

# The 39 features XGBoost expects
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

# Your actual dataset means for the other 29 features
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
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

app.layout = html.Div([
    html.Div([
        html.H2("Health Risk Analysis", style={'textAlign': 'center', 'color': '#2c3e50'}),
        html.Hr(),
        
        # Grid of Input Cards
        html.Div([
            html.Div([
                html.Label(f.replace('_', ' ').title(), style={'fontWeight': 'bold'}),
                dcc.Input(id={'type': 'input-field', 'index': f}, type='number', value=round(MEANS[f], 2), style={'width': '100%'})
            ], className="four columns", style={'padding': '10px', 'border': '1px solid #eee', 'borderRadius': '5px', 'margin': '5px'}) 
            for f in DASHBOARD_FEATURES
        ], className="row"),

        html.Br(),
        html.Button('Run Diagnostic', id='submit-val', n_clicks=0, className="button-primary", style={'width': '100%', 'height': '50px'}),
        
        html.Br(),
        html.Div(id='prediction-output', style={'padding': '20px', 'textAlign': 'center'})
    ], style={'maxWidth': '900px', 'margin': 'auto', 'padding': '20px', 'backgroundColor': '#fff', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'})
], style={'backgroundColor': '#f4f7f6', 'minHeight': '100vh', 'paddingTop': '50px'})

@app.callback(
    Output('prediction-output', 'children'),
    Input('submit-val', 'n_clicks'),
    State({'type': 'input-field', 'index': ALL}, 'value'),
    State({'type': 'input-field', 'index': ALL}, 'id')
)
def update_output(n_clicks, values, ids):
    if n_clicks > 0:
        # Map values back to their feature names
        user_input = {item['index']: val for item, val in zip(ids, values)}
        
        try:
            # 1. K-Means Path (19 features, scaled)
            km_raw = [user_input.get(f, MEANS[f]) for f in KMEANS_FEATURES]
            km_scaled = scaler.transform([km_raw])
            cluster = kmeans_model.predict(km_scaled)[0]
            
            # 2. XGBoost Path (39 features, unscaled)
            xgb_raw = [user_input.get(f, MEANS[f]) for f in XGB_FEATURES]
            risk = xgb_model.predict([xgb_raw])[0]
            
            risk_text = "High Risk" if risk == 1 else "Low Risk"
            
            return html.Div([
                html.H4(f"Diagnostic Result: {risk_text}", style={'color': '#e74c3c' if risk == 1 else '#27ae60'}),
                html.P(f"Lifestyle Cluster: {cluster}", style={'fontSize': '18px', 'color': '#7f8c8d'})
            ])
        except Exception as e:
            return html.Div(f"Error in processing: {str(e)}", style={'color': 'red'})
    
    return "Enter values and click Run Diagnostic."

if __name__ == '__main__':
    app.run_server(debug=True)
