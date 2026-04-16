import os
import joblib
import dash
from dash import html

# --- Path Resolution ---
# 1. Get the directory of this file (/project/src/)
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up to the repo root (/project/)
root_dir = os.path.dirname(current_script_dir)

# 3. Target the sibling artifacts folder (/project/artifacts/)
base_path = os.path.join(root_dir, 'artifacts')

try:
    # Use the exact name you see on GitHub
    scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
    status = "SUCCESS: Scaler loaded from root sibling folder."
except Exception as e:
    # If this fails, the error message will show us the exact path it tried
    status = f"FAILED: {e} | Looking in: {base_path}"

# --- Minimal App ---
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Scaler Integration Test"),
    html.P(status, style={'fontWeight': 'bold', 'color': 'green' if "SUCCESS" in status else 'red'}),
    html.Hr(),
    html.Code(f"Resolved Root: {root_dir}")
])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run_server(host='0.0.0.0', port=port)
