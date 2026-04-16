import os
import joblib
import dash
from dash import html

# --- 1. Efficient Path Logic ---
# Since web_app.py is in /src, we go UP one level to find /artifacts
current_dir = os.path.dirname(os.path.abspath(__file__)) # This is /src
root_dir = os.path.dirname(current_dir) # This is / (root)
base_path = os.path.join(root_dir, 'artifacts')

# Load artifacts here - if this fails, check the logs immediately
try:
    # We will start with JUST the scaler to test memory
    scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
    # xgb_model = joblib.load(os.path.join(base_path, 'xgb_model.pkl')) 
except Exception as e:
    print(f"Artifact Load Error: {e}")

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Structure Test Successful"),
    html.P(f"Artifacts folder located at: {base_path}")
])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run_server(host='0.0.0.0', port=port, debug=False)
