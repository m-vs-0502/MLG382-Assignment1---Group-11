import dash
from dash import dcc, html
import joblib
import pandas as pd

# Load the artifacts
model = joblib.load('artifacts/model_1.pkl')
scaler = joblib.load('artifacts/scaler.pkl')

app = dash.Dash(__name__)
server = app.server # Required for Render deployment

app.layout = html.Div([
    html.H1("Diabetes Risk & Segmentation Portal"),
    html.P("This app predicts patient risk using XGBoost and K-Means clustering."),
    # You will add Input fields here later!
])

if __name__ == '__main__':
    app.run_server(debug=True)