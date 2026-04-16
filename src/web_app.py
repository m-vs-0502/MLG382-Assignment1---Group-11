import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import joblib
import plotly.graph_objects as go
import os

# --- 1. Load Artifacts ---
# This looks for the directory where THIS file (web_app.py) is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# This moves UP one level to the root, then DOWN into artifacts
# This works whether you are on Windows, Linux, or Render
base_path = os.path.join(current_dir, '..', 'artifacts')

scaler_path = os.path.join(base_path, 'scaler.pkl')
xgb_path = os.path.join(base_path, 'xgb_model.pkl')
kmeans_path = os.path.join(base_path, 'kmeans_model.pkl')

# Check if files exist before loading to help with debugging
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Could not find scaler at: {scaler_path}")

scaler = joblib.load(scaler_path)
xgb_model = joblib.load(xgb_path)
kmeans_model = joblib.load(kmeans_path)

app = dash.Dash(__name__)

# --- 2. Helper Functions ---

def create_risk_gauge(risk_percent):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_percent,
        title = {'text': "Diabetes Risk Probability (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 30], 'color': "#d4edda"}, # Green
                {'range': [30, 70], 'color': "#fff3cd"}, # Yellow
                {'range': [70, 100], 'color': "#f8d7da"} # Red
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=30, r=30, t=50, b=20))
    return fig

def get_cluster_advice(cluster_id):
    advice_map = {
        0: {
            "title": "Healthy Baseline Group",
            "body": "Your metrics are within optimal ranges. Maintain your current physical activity and balanced diet.",
            "color": "#d4edda"
        },
        1: {
            "title": "High Risk / Clinical Group",
            "body": "Profile shows elevated glucose and lower activity. We recommend consulting a healthcare provider and focusing on 150 min/week cardio.",
            "color": "#f8d7da"
        },
        2: {
            "title": "Lifestyle Management Group",
            "body": "Moderate risk detected. Focus on reducing processed sugar intake and improving sleep hygiene to stabilize glucose.",
            "color": "#fff3cd"
        }
    }
    selected = advice_map.get(cluster_id, advice_map[0])
    return html.Div([
        html.H4(selected["title"]),
        html.P(selected["body"])
    ], style={'backgroundColor': selected['color'], 'padding': '20px', 'borderRadius': '10px', 'marginTop': '20px'})

# --- 3. App Layout ---

app.layout = html.Div([
    html.H1("Diabetes Health Analytics Dashboard", style={'textAlign': 'center'}),
    html.Hr(),
    
    html.Div([
        html.H3("Patient Data Entry"),
        html.Label("Age"),
        dcc.Slider(id='in-age', min=18, max=100, step=1, value=45, marks={18: '18', 100: '100'}),
        
        html.Label("BMI"),
        dcc.Input(id='in-bmi', type='number', value=25.0, style={'width': '100%'}),
        
        html.Label("Fasting Glucose (mg/dL)"),
        dcc.Input(id='in-glucose', type='number', value=100.0, style={'width': '100%'}),
        
        html.Label("HbA1c Level"),
        dcc.Input(id='in-hba1c', type='number', value=5.5, style={'width': '100%'}),
        
        html.Label("Physical Activity (min/week)"),
        dcc.Input(id='in-activity', type='number', value=150, style={'width': '100%'}),
        
        html.Label("Ethnicity"),
        dcc.Dropdown(id='in-eth', options=[
            {'label': 'White', 'value': 'White'},
            {'label': 'Black', 'value': 'Black'},
            {'label': 'Hispanic', 'value': 'Hispanic'},
            {'label': 'Other', 'value': 'Other'}
        ], value='White'),
        
        html.Button('Generate Prediction', id='btn-calc', n_clicks=0, 
                    style={'marginTop': '30px', 'width': '100%', 'backgroundColor': '#007bff', 'color': 'white'})
    ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px', 'borderRight': '1px solid #ccc'}),
    
    html.Div([
        html.H3("Clinical Analysis Results"),
        dcc.Graph(id='risk-gauge'),
        html.Div(id='advice-card')
    ], style={'width': '65%', 'display': 'inline-block', 'padding': '20px', 'verticalAlign': 'top'})
])

# --- 4. Callbacks ---

@app.callback(
    [Output('risk-gauge', 'figure'), Output('advice-card', 'children')],
    [Input('btn-calc', 'n_clicks')],
    [State('in-age', 'value'), State('in-bmi', 'value'), State('in-glucose', 'value'),
     State('in-hba1c', 'value'), State('in-activity', 'value'), State('in-eth', 'value')]
)
def update_output(n, age, bmi, glucose, hba1c, activity, ethnicity):
    if n > 0:
        # Define exact 39-column list from training
        cols = [
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
        
        # 1. Map inputs to dictionary
        data = {c: 0 for c in cols}
        data.update({
            'Age': age, 'bmi': bmi, 'glucose_fasting': glucose, 
            'hba1c': hba1c, 'physical_activity_minutes_per_week': activity,
            f'ethnicity_{ethnicity}': 1
        })
        
        # 2. Scale & Predict
        input_df = pd.DataFrame([data])[cols] # Force column order
        scaled_input = scaler.transform(input_df)
        
        risk = xgb_model.predict_proba(scaled_input)[0][1] * 100
        cluster = kmeans_model.predict(scaled_input)[0]
        
        return create_risk_gauge(risk), get_cluster_advice(cluster)
    
    return go.Figure(), ""

if __name__ == '__main__':
    app.run_server(debug=True)