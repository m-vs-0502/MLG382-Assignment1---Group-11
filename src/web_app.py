import os
import joblib
import pandas as pd # Heavy library
import dash
from dash import html

# --- 1. Efficient Path Logic ---
# Since web_app.py is in /src, we go UP one level to find /artifacts
current_dir = os.path.dirname(os.path.abspath(__file__)) # This is /src
root_dir = os.path.dirname(current_dir) # This is / (root)
base_path = os.path.join(root_dir, 'artifacts')

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
