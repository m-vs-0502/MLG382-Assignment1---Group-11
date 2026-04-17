import os
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import joblib
import pickle

# ----------------------------------------------------------------------------
# LOAD ARTIFACTS
# ----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')

print(f"Loading artifacts from: {ARTIFACTS_DIR}")

try:
    xgb_model = joblib.load(os.path.join(ARTIFACTS_DIR, 'xgboost_model.pkl'))
    kmeans_model = joblib.load(os.path.join(ARTIFACTS_DIR, 'kmeans_model.pkl'))
    
    with open(os.path.join(ARTIFACTS_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    with open(os.path.join(ARTIFACTS_DIR, 'feature_columns.pkl'), 'rb') as f:
        FEATURE_COLUMNS = pickle.load(f)
    
    with open(os.path.join(ARTIFACTS_DIR, 'feature_means.pkl'), 'rb') as f:
        FEATURE_MEANS = pickle.load(f)
    
    with open(os.path.join(ARTIFACTS_DIR, 'target_encoder.pkl'), 'rb') as f:
        target_encoder = pickle.load(f)
    
    with open(os.path.join(ARTIFACTS_DIR, 'cluster_labels.pkl'), 'rb') as f:
        CLUSTER_LABELS = pickle.load(f)
    
    print("All artifacts loaded successfully!")

except Exception as e:
    print(f"Error loading artifacts: {e}")
    raise

# ----------------------------------------------------------------------------
# FEATURE DEFINITIONS
# ----------------------------------------------------------------------------

# All 39 features the model expects (for autofilling)
ALL_FEATURES = FEATURE_COLUMNS

# The 8 user inputs
USER_INPUTS = [
    # Core metabolic (both models)
    'hba1c',
    'glucose_fasting',
    
    # K-Means drivers
    'bmi',
    'ldl_cholesterol',
    
    # XGBoost drivers
    'Age',
    'gender',
    'glucose_postprandial',
    'physical_activity_minutes_per_week'
]

INPUT_GROUPS = [
    {
        'title': 'Core Metabolic Markers',
        'fields': [
            {'id': 'hba1c', 'label': 'HbA1c (%)', 'type': 'number', 
             'min': 3.0, 'max': 15.0, 'step': 0.1, 'default': 5.7},
            {'id': 'glucose_fasting', 'label': 'Fasting Glucose (mg/dL)', 'type': 'number',
             'min': 50, 'max': 400, 'step': 1, 'default': 95},
            {'id': 'glucose_postprandial', 'label': 'Postprandial Glucose (mg/dL)', 'type': 'number',
             'min': 70, 'max': 500, 'step': 1, 'default': 140},
        ]
    },
    {
        'title': 'Body & Lipid Metrics',
        'fields': [
            {'id': 'bmi', 'label': 'BMI', 'type': 'number',
             'min': 15, 'max': 50, 'step': 0.1, 'default': 25},
            {'id': 'ldl_cholesterol', 'label': 'LDL Cholesterol (mg/dL)', 'type': 'number',
             'min': 30, 'max': 250, 'step': 1, 'default': 100},
        ]
    },
    {
        'title': 'Demographics & Lifestyle',
        'fields': [
            {'id': 'Age', 'label': 'Age (years)', 'type': 'number',
             'min': 18, 'max': 100, 'step': 1, 'default': 50},
            {'id': 'gender', 'label': 'Gender', 'type': 'radio',
             'options': [
                 {'label': 'Male', 'value': 'Male'},
                 {'label': 'Female', 'value': 'Female'},
             ],
             'default': 'Female'},
            {'id': 'physical_activity_minutes_per_week', 'label': 'Physical Activity (min/week)', 'type': 'number',
             'min': 0, 'max': 500, 'step': 10, 'default': 150},
        ]
    }
]

VALIDATION_RULES = {
    'hba1c': {'min': 3.0, 'max': 15.0, 'message': 'HbA1c should be between 3.0% and 15.0%'},
    'glucose_fasting': {'min': 50, 'max': 400, 'message': 'Fasting glucose should be between 50-400 mg/dL'},
    'glucose_postprandial': {'min': 70, 'max': 500, 'message': 'Postprandial glucose should be between 70-500 mg/dL'},
    'bmi': {'min': 15, 'max': 50, 'message': 'BMI should be between 15 and 50'},
    'ldl_cholesterol': {'min': 30, 'max': 250, 'message': 'LDL cholesterol should be between 30-250 mg/dL'},
    'Age': {'min': 18, 'max': 100, 'message': 'Age should be between 18 and 100 years'},
    'physical_activity_minutes_per_week': {'min': 0, 'max': 500, 'message': 'Activity should be between 0-500 min/week'},
}

# ----------------------------------------------------------------------------
# FEATURE CONVERSION
# ----------------------------------------------------------------------------

def prepare_input_vector(form_data):
    """Convert form data to complete 39-feature vector."""
    input_data = FEATURE_MEANS.copy()
    
    for key, value in form_data.items():
        if key == 'gender':
            input_data['gender'] = 1 if value == 'Male' else 0
        elif key in ALL_FEATURES:
            input_data[key] = float(value) if value is not None else FEATURE_MEANS[key]
    
    return pd.DataFrame([input_data])[ALL_FEATURES]


def validate_inputs(form_data):
    """Validate all numeric inputs."""
    errors = []
    for field, rules in VALIDATION_RULES.items():
        if field in form_data and form_data[field] is not None:
            try:
                value = float(form_data[field])
                if value < rules['min'] or value > rules['max']:
                    errors.append(rules['message'])
            except (ValueError, TypeError):
                errors.append(f"{field} must be a valid number")
    return errors


def predict_risk(form_data):
    """Make prediction and return results."""
    input_df = prepare_input_vector(form_data)
    
    # Scale all features (scaler was fitted on all 39 features)
    input_scaled = input_df.copy()
    input_scaled = pd.DataFrame(
        scaler.transform(input_df),
        columns=ALL_FEATURES
    )
    
    # XGBoost prediction (uses all 39 features)
    risk_class = int(xgb_model.predict(input_scaled)[0])
    risk_label = target_encoder.inverse_transform([risk_class])[0]
    proba = xgb_model.predict_proba(input_scaled)[0]
    
    class_probas = []
    for i, prob in enumerate(proba):
        class_name = target_encoder.inverse_transform([i])[0]
        class_probas.append((class_name, prob * 100))
    class_probas.sort(key=lambda x: x[1], reverse=True)
    
    # K-Means clustering (uses all 39 features)
    cluster = int(kmeans_model.predict(input_scaled)[0])
    cluster_label = CLUSTER_LABELS.get(cluster, f"Cluster {cluster}")
    
    return risk_label, cluster_label, class_probas


def get_interpretation(risk_label, cluster_label):
    """Generate human-readable interpretation."""
    if risk_label == 'Type 1':
        return "Type 1 diabetes is an autoimmune condition requiring insulin therapy. Consult an endocrinologist for proper management."
    
    if risk_label == 'Gestational':
        return "Gestational diabetes requires careful monitoring during pregnancy. Consult your obstetrician for appropriate care."
    
    interpretations = {
        ('No Diabetes', 'Low Risk'): "Patient shows no signs of diabetes and has healthy metabolic and lifestyle indicators. Recommend maintaining current habits with regular check-ups.",
        ('No Diabetes', 'Moderate Risk'): "No diabetes detected, but metabolic and lifestyle factors suggest moderate risk. Consider increasing physical activity and monitoring diet.",
        ('No Diabetes', 'High Risk'): "No diabetes currently, but high-risk metabolic profile detected. Strongly recommend lifestyle changes and regular glucose monitoring.",
        ('Pre-Diabetes', 'Low Risk'): "Pre-diabetes detected despite healthy lifestyle. This may indicate genetic factors. Consult healthcare provider for management plan.",
        ('Pre-Diabetes', 'Moderate Risk'): "Pre-diabetes with moderate lifestyle risk. Lifestyle improvements may help prevent progression to Type 2 diabetes.",
        ('Pre-Diabetes', 'High Risk'): "Pre-diabetes with high-risk metabolic profile. Urgent lifestyle intervention recommended to prevent progression.",
        ('Type 2', 'Low Risk'): "Type 2 diabetes with well-managed lifestyle factors. Continue current management and regular monitoring.",
        ('Type 2', 'Moderate Risk'): "Type 2 diabetes with room for lifestyle improvement. Focus on diet, exercise, and medication adherence.",
        ('Type 2', 'High Risk'): "Type 2 diabetes with high-risk profile. Immediate lifestyle changes and medical consultation strongly advised.",
    }
    
    key = (risk_label, cluster_label)
    if key in interpretations:
        return interpretations[key]
    
    return f"Based on the model, this patient is classified as '{risk_label}' with a '{cluster_label}' lifestyle profile. Clinical correlation is recommended."


# ----------------------------------------------------------------------------
# DASH APP LAYOUT
# ----------------------------------------------------------------------------

app = dash.Dash(__name__)
server = app.server

input_sections = []
for group in INPUT_GROUPS:
    fields = []
    for field in group['fields']:
        if field['type'] == 'number':
            fields.append(html.Div([
                html.Label(field['label'], style={'fontWeight': 'bold', 'marginTop': 10}),
                dcc.Input(
                    id=f"input-{field['id']}",
                    type='number',
                    min=field.get('min'),
                    max=field.get('max'),
                    step=field.get('step', 1),
                    value=field.get('default'),
                    style={'width': '100%', 'padding': '8px', 'marginBottom': '5px',
                           'border': '1px solid #ddd', 'borderRadius': '4px'}
                )
            ]))
        elif field['type'] == 'radio':
            fields.append(html.Div([
                html.Label(field['label'], style={'fontWeight': 'bold', 'marginTop': 10}),
                dcc.RadioItems(
                    id=f"input-{field['id']}",
                    options=field['options'],
                    value=field.get('default'),
                    style={'marginBottom': '10px'}
                )
            ]))
    
    input_sections.append(html.Div([
        html.H3(group['title'], style={'color': '#2c3e50', 'borderBottom': '2px solid #3498db', 'paddingBottom': '5px'}),
        html.Div(fields, style={'padding': '10px'})
    ], style={'marginBottom': '20px'}))

app.layout = html.Div([
    html.H1("Diabetes Risk Predictor", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
    
    html.P("Enter patient data. Missing features use population averages.",
           style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': 30}),
    
    html.Div(input_sections, style={
        'maxWidth': '700px', 'margin': '0 auto', 'padding': '20px',
        'backgroundColor': '#f8f9fa', 'borderRadius': '8px'
    }),
    
    html.Div(id='validation-errors', style={
        'maxWidth': '700px', 'margin': '10px auto', 'color': '#e74c3c'
    }),
    
    html.Div([
        html.Button('Predict Risk', id='predict-button', n_clicks=0,
                   style={'backgroundColor': '#3498db', 'color': 'white',
                          'padding': '12px 40px', 'fontSize': '16px',
                          'border': 'none', 'borderRadius': '5px',
                          'cursor': 'pointer', 'marginTop': 20})
    ], style={'textAlign': 'center'}),
    
    html.Div(id='prediction-output', style={
        'maxWidth': '700px', 'margin': '30px auto', 'padding': '20px',
        'backgroundColor': '#f8f9fa', 'borderRadius': '8px',
        'textAlign': 'center', 'fontSize': '18px'
    }),
    
    html.Footer([
        html.P("This tool is for educational purposes only. Consult a healthcare professional for medical advice.",
               style={'color': '#95a5a6', 'fontSize': '12px', 'marginTop': 50})
    ], style={'textAlign': 'center'})
])

# ----------------------------------------------------------------------------
# CALLBACK
# ----------------------------------------------------------------------------

@app.callback(
    [Output('prediction-output', 'children'),
     Output('validation-errors', 'children')],
    Input('predict-button', 'n_clicks'),
    [State(f"input-{feat}", 'value') for feat in USER_INPUTS]
)
def update_prediction(n_clicks, *values):
    if n_clicks == 0:
        return html.Div([
            html.P("Enter patient data and click 'Predict Risk' to see results.",
                  style={'color': '#7f8c8d'})
        ]), ""
    
    form_data = {feat: val for feat, val in zip(USER_INPUTS, values)}
    
    errors = validate_inputs(form_data)
    if errors:
        error_div = html.Div([html.P(f"⚠ {err}", style={'margin': '5px 0'}) for err in errors])
        return html.Div(), error_div
    
    try:
        risk_label, cluster_label, class_probas = predict_risk(form_data)
        
        risk_colors = {
            'No Diabetes': '#27ae60',
            'Pre-Diabetes': '#f39c12',
            'Type 1': '#e74c3c',
            'Type 2': '#c0392b',
            'Gestational': '#8e44ad'
        }
        
        cluster_colors = {
            'Low Risk': '#27ae60',
            'Moderate Risk': '#f39c12',
            'High Risk': '#e74c3c'
        }
        
        proba_bars = []
        for class_name, prob in class_probas:
            bar_color = risk_colors.get(class_name, '#95a5a6')
            is_top = (class_name == risk_label)
            proba_bars.append(html.Div([
                html.Div([
                    html.Span(class_name, style={'flex': '1', 'textAlign': 'left'}),
                    html.Span(f"{prob:.1f}%", style={'fontWeight': 'bold' if is_top else 'normal'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '3px'}),
                html.Div(style={
                    'width': f'{prob}%',
                    'height': '20px',
                    'backgroundColor': bar_color,
                    'borderRadius': '4px',
                    'opacity': 1.0 if is_top else 0.5,
                })
            ], style={'marginBottom': '10px'}))
        
        return html.Div([
            html.H3("Prediction Results", style={'marginBottom': 20}),
            
            html.Div([
                html.Div([
                    html.P("Diabetes Risk:", style={'fontWeight': 'bold', 'marginBottom': 5}),
                    html.P(risk_label, style={'fontSize': '24px', 'fontWeight': 'bold',
                           'color': risk_colors.get(risk_label, '#2c3e50')}),
                ], style={'flex': '1'}),
                html.Div([
                    html.P("Lifestyle Cluster:", style={'fontWeight': 'bold', 'marginBottom': 5}),
                    html.P(cluster_label, style={'fontSize': '20px', 'fontWeight': 'bold',
                           'color': cluster_colors.get(cluster_label, '#2c3e50')}),
                ], style={'flex': '1'}),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': 20}),
            
            html.Hr(),
            
            html.P("Probability Distribution:", style={'fontWeight': 'bold', 'marginTop': 20}),
            html.Div(proba_bars, style={'marginBottom': 20}),
            
            html.Hr(),
            
            html.P("Interpretation:", style={'fontWeight': 'bold', 'marginTop': 20}),
            html.P(get_interpretation(risk_label, cluster_label), style={'color': '#555', 'lineHeight': '1.6'}),
            
        ]), ""
        
    except Exception as e:
        return html.Div([
            html.P("Error making prediction", style={'color': '#e74c3c'}),
            html.P(str(e), style={'color': '#e74c3c', 'fontSize': '14px'})
        ]), ""


# ----------------------------------------------------------------------------
# RUN APP
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run_server(host='0.0.0.0', port=port, debug=False)
