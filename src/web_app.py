import os
import joblib
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, ALL

# --- 1. Path Resolution ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
base_path = os.path.join(root_dir, 'artifacts')

# --- 2. Load Artifacts ---
try:
    scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
    model1 = joblib.load(os.path.join(base_path, 'model1_pkl'))
    model2 = joblib.load(os.path.join(base_path, 'model2_pkl'))
    load_status = "All models loaded."
except Exception as e:
    load_status = f"Load Error: {e}"

# --- 3. Feature Definitions ---
# These MUST match the order from your scaler check
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
    html.H1("Health Prediction Dashboard"),
    html.P(load_status, style={'color': 'gray', 'fontSize': '12px'}),
    
    # Input Grid
    html.Div([
        html.Div([
            html.Label(name.replace('_', ' ').title(), style={'fontSize': '14px'}),
            dcc.Input(
                id={'type': 'feature-input', 'index': i}, 
                type='number', 
                value=0,
                style={'width': '100%'}
            )
        ], style={'padding': '5px'}) for i, name in enumerate(feature_names)
    ], style={
        'display': 'grid', 
        'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))', 
        'gap': '10px',
        'marginBottom': '20px'
    }),
    
    html.Button('Generate Predictions', id='predict-btn', n_clicks=0, 
                style={'padding': '10px 20px', 'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 'borderRadius': '5px'}),
    
    html.Hr(),
    html.Div(id='prediction-output')
], style={'padding': '20px', 'fontFamily': 'sans-serif'})

# --- 5. Prediction Callback ---
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State({'type': 'feature-input', 'index': ALL}, 'value')
)
def predict(n_clicks, values):
    if n_clicks == 0:
        return "Enter values and click predict."
    
    try:
        # Convert to 2D array: shape (1, 19)
        features_array = np.array([values])
        
        # Scale the data
        scaled_features = scaler.transform(features_array)
        
        # Get predictions
        p1 = model1.predict(scaled_features)[0]
        p2 = model2.predict(scaled_features)[0]
        
        return html.Div([
            html.H3("Results:"),
            html.P(f"Model 1 Prediction: {p1:.4f}"),
            html.P(f"Model 2 Prediction: {p2:.4f}")
        ])
    except Exception as e:
        return html.P(f"Error: {str(e)}", style={'color': 'red'})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run_server(host='0.0.0.0', port=port)
