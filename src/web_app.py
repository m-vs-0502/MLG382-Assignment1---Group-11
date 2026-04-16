import os
import joblib
import pandas as pd # Heavy library
import dash
from dash import html

# --- Path Logic ---
# Finds /artifacts from /src
base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifacts')

# Phase 1: Try loading ONLY the scaler
try:
    scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
    status = "Scaler loaded successfully!"
except Exception as e:
    status = f"Error: {e}"

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Phase 1: Library & Scaler Test"),
    html.P(status)
])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run_server(host='0.0.0.0', port=port)
