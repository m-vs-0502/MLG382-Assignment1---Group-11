import os
import joblib
import pandas as pd # Heavy library
import dash
from dash import html

# --- 1. Efficient Path Logic ---
# Since web_app.py is in /src, we go UP one level to find /artifacts
base_path = '/opt/render/project/artifacts'

# Fallback for local testing (so it doesn't break on your PC)
if not os.path.exists(base_path):
    # This looks for 'artifacts' in the directory above 'src'
    base_path = os.path.join(os.path.dirname(os.getcwd()), 'artifacts')

try:
    scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
    status = "Scaler loaded successfully!"
except Exception as e:
    # This will print the actual path it tried to use in your logs
    status = f"Error: {e} | Path attempted: {base_path}"

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Phase 1: Library & Scaler Test"),
    html.P(status)
])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run_server(host='0.0.0.0', port=port)
