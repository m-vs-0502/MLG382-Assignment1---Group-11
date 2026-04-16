import os
import joblib
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State

# --- 1. Robust Path Resolution ---
# Points to /project/artifacts/ while script runs in /project/src/
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
base_path = os.path.join(root_dir, 'artifacts')

def load_artifact(filename):
    path = os.path.join(base_path, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing artifact: {path}")
    return joblib.load(path)

# --- 2. Global Model Loading ---
# We load these once at startup to stay under the 512MB RAM limit
try:
    scaler = load_artifact('scaler.pkl')
    # Use the exact filenames you confirmed are on GitHub
    model1 = load_artifact('model1_pkl') 
    model2 = load_artifact('model2_pkl')
    print("All models loaded successfully.")
except Exception as e:
    print(f"Startup Error: {e}")

# --- 3. Dash App Setup ---
app = dash.Dash(__name__)
server = app.server # Required for Gunicorn

app.layout = html.Div([
    html.H1("Machine Learning Model Dashboard"),
    html.P("Enter values below to generate a prediction."),
    
    html.Div([
        html.Label("Input Feature 1:"),
        dcc.Input(id='input-1', type='number', value=0),
        
        html.Label("Input Feature 2:"),
        dcc.Input(id='input-2', type='number', value=0),
        
        html.Button('Run Prediction', id='predict-btn', n_clicks=0),
    ], style={'display': 'flex', 'flexDirection': 'column', 'width': '300px', 'gap': '10px'}),
    
    html.Hr(),
    html.Div(id='prediction-output', style={'fontSize': '20px', 'fontWeight': 'bold'})
])

# --- 4. Prediction Logic ---
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('input-1', 'value'),
    State('input-2', 'value')
)
def predict(n_clicks, val1, val2):
    if n_clicks == 0:
        return "Awaiting input..."
    
    try:
        # 1. Format the data for the scaler
        raw_data = np.array([[val1, val2]])
        
        # 2. Scale and Predict
        scaled_data = scaler.transform(raw_data)
        res1 = model1.predict(scaled_data)[0]
        res2 = model2.predict(scaled_data)[0]
        
        return f"Model 1 Result: {res1:.2f} | Model 2 Result: {res2:.2f}"
    except Exception as e:
        return f"Prediction Error: {str(e)}"

if __name__ == '__main__':
    # Use Render's assigned port
    port = int(os.environ.get("PORT", 10000))
    app.run_server(host='0.0.0.0', port=port)
