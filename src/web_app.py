import os
import joblib
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, ALL

# --- 1. Path Resolution ---
# Script is in /project/src/, artifacts are in /project/artifacts/
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
base_path = os.path.join(root_dir, 'artifacts')

# --- 2. Load Artifacts ---
try:
    scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
    # Updated to your specific filenames
    xgb_model = joblib.load(os.path.join(base_path, 'xgboost_model.pkl'))
    kmeans_model = joblib.load(os.path.join(base_path, 'kmeans_model.pkl'))
    load_status = "System Ready: All models and scaler loaded."
except Exception as e:
    load_status = f"System Error: {e}"

# --- 3. Feature Definitions ---
feature_names = [
    'Age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 
    'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi', 
    'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate', 
    'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides', 
    'glucose_fasting', 'glucose_postprandial', 'insulin_level', 'hba1c'
]

# --- 4. Dash App ---
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Health Analysis Dashboard"),
    html.P(load_status, style={'color': '#666', 'fontSize': '14px', 'marginBottom': '20px'}),
    
    # Dynamic Input Grid
    html.Div([
        html.Div([
            html.Label(name.replace('_', ' ').title(), style={'fontWeight': 'bold', 'display': 'block'}),
            dcc.Input(
                id={'type': 'feature-input', 'index': i}, 
                type='number', 
                value=0,
                style={'width': '100%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc'}
            )
        ], style={'padding': '10px'}) for i, name in enumerate(feature_names)
    ], style={
        'display': 'grid', 
        'gridTemplateColumns': 'repeat(auto-fit, minmax(220px, 1fr))', 
        'gap': '15px',
        'backgroundColor': '#f9f9f9',
        'padding': '20px',
        'borderRadius': '8px'
    }),
    
    html.Button('Run Analysis', id='predict-btn', n_clicks=0, 
                style={
                    'marginTop': '20px', 'padding': '12px 30px', 'fontSize': '16px',
                    'backgroundColor': '#28a745', 'color': 'white', 'border': 'none', 
                    'borderRadius': '5px', 'cursor': 'pointer'
                }),
    
    html.Hr(style={'marginTop': '30px'}),
    html.Div(id='prediction-output', style={'marginTop': '20px'})
], style={'maxWidth': '1100px', 'margin': '0 auto', 'padding': '40px', 'fontFamily': 'system-ui'})

# --- 5. Prediction Callback ---
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State({'type': 'feature-input', 'index': ALL}, 'value')
)
def predict(n_clicks, values):
    if n_clicks == 0:
        return html.Div("Awaiting input for analysis...", style={'color': '#888'})
    
    try:
        # Prepare data (Shape: 1, 19)
        input_data = np.array([values])
        
        # 1. Scale inputs
        scaled_data = scaler.transform(input_data)
        
        # 2. Get XGBoost Prediction
        xgb_res = xgb_model.predict(scaled_data)[0]
        
        # 3. Get K-Means Cluster Assignment
        cluster_res = kmeans_model.predict(scaled_data)[0]
        
        return html.Div([
            html.Div([
                html.H3("XGBoost Prediction Result"),
                html.P(f"{xgb_res:.4f}", style={'fontSize': '24px', 'color': '#28a745', 'fontWeight': 'bold'})
            ], style={'display': 'inline-block', 'width': '45%', 'verticalAlign': 'top'}),
            
            html.Div([
                html.H3("K-Means Cluster Assignment"),
                html.P(f"Cluster Group: {cluster_res}", style={'fontSize': '24px', 'color': '#007bff', 'fontWeight': 'bold'})
            ], style={'display': 'inline-block', 'width': '45%', 'verticalAlign': 'top'})
        ], style={'textAlign': 'center', 'padding': '20px', 'border': '1px solid #eee', 'borderRadius': '8px'})
        
    except Exception as e:
        return html.Div(f"Analysis Error: {str(e)}", style={'color': '#dc3545', 'fontWeight': 'bold'})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run_server(host='0.0.0.0', port=port)
