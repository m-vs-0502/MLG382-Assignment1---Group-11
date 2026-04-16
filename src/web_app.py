import os
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import joblib
import pickle

# ============================================================================
# 1. LOAD ARTIFACTS
# ============================================================================

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
    
    print("✅ All artifacts loaded successfully!")

except Exception as e:
    print(f"❌ Error loading artifacts: {e}")
    raise

# ============================================================================
# 2. FEATURE DEFINITIONS
# ============================================================================

CLINICAL_FEATURES = [
    'Age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week',
    'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
    'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
    'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides',
    'glucose_fasting', 'glucose_postprandial', 'insulin_level', 'hba1c'
]

# Simplified input groups for better UX
INPUT_GROUPS = [
    # Group 1: Core Clinical Markers (most important)
    {
        'title': '🩸 Core Clinical Markers',
        'fields': [
            {'id': 'hba1c', 'label': 'HbA1c (%)', 'type': 'number', 
             'min': 3.0, 'max': 15.0, 'step': 0.1, 'default': 5.7,
             'help': 'Average blood sugar over past 3 months'},
            {'id': 'glucose_fasting', 'label': 'Fasting Glucose (mg/dL)', 'type': 'number',
             'min': 50, 'max': 400, 'step': 1, 'default': 95,
             'help': 'Blood sugar after 8+ hours of fasting'},
            {'id': 'glucose_postprandial', 'label': 'Postprandial Glucose (mg/dL)', 'type': 'number',
             'min': 70, 'max': 500, 'step': 1, 'default': 140,
             'help': 'Blood sugar 2 hours after eating'},
        ]
    },
    # Group 2: Demographics
    {
        'title': '👤 Demographics',
        'fields': [
            {'id': 'Age', 'label': 'Age (years)', 'type': 'number',
             'min': 18, 'max': 100, 'step': 1, 'default': 50},
            {'id': 'gender', 'label': 'Gender', 'type': 'checkbox', 'default': False,
             'checkbox_label': 'Male (unchecked = Female)',
             'help': 'Biological sex'},
            {'id': 'family_history_diabetes', 'label': 'Family History', 'type': 'checkbox', 'default': False,
             'checkbox_label': 'Family history of diabetes',
             'help': 'Parent or sibling with diabetes'},
        ]
    },
    # Group 3: Socioeconomic (combined dropdown)
    {
        'title': '💼 Socioeconomic Status',
        'fields': [
            {'id': 'socioeconomic', 'label': 'Income & Employment', 'type': 'dropdown',
             'options': [
                 {'label': 'Low Income', 'value': 'low'},
                 {'label': 'Lower-Middle Income', 'value': 'lower_middle'},
                 {'label': 'Middle Income', 'value': 'middle'},
                 {'label': 'Upper-Middle Income', 'value': 'upper_middle'},
                 {'label': 'Unemployed', 'value': 'unemployed'},
                 {'label': 'Employed - Middle Income', 'value': 'employed_middle'},
                 {'label': 'Retired - Middle Income', 'value': 'retired_middle'},
                 {'label': 'Student - Low Income', 'value': 'student_low'},
             ],
             'default': 'employed_middle',
             'help': 'Combined income level and employment status'},
        ]
    },
    # Group 4: Lifestyle Factors
    {
        'title': '🏃 Lifestyle Factors',
        'fields': [
            {'id': 'physical_activity_minutes_per_week', 'label': 'Physical Activity (min/week)', 'type': 'number',
             'min': 0, 'max': 500, 'step': 10, 'default': 150,
             'help': 'Moderate to vigorous activity per week'},
            {'id': 'alcohol_consumption_per_week', 'label': 'Alcohol (drinks/week)', 'type': 'number',
             'min': 0, 'max': 30, 'step': 1, 'default': 2},
            {'id': 'diet_score', 'label': 'Diet Score (0-10)', 'type': 'number',
             'min': 0, 'max': 10, 'step': 0.5, 'default': 6,
             'help': 'Higher = healthier diet'},
            {'id': 'sleep_hours_per_day', 'label': 'Sleep (hours/day)', 'type': 'number',
             'min': 3, 'max': 12, 'step': 0.5, 'default': 7},
            {'id': 'screen_time_hours_per_day', 'label': 'Screen Time (hours/day)', 'type': 'number',
             'min': 0, 'max': 16, 'step': 0.5, 'default': 6},
        ]
    }
]

# Validation ranges for all numeric inputs
VALIDATION_RULES = {
    'hba1c': {'min': 3.0, 'max': 15.0, 'message': 'HbA1c should be between 3.0% and 15.0%'},
    'glucose_fasting': {'min': 50, 'max': 400, 'message': 'Fasting glucose should be between 50-400 mg/dL'},
    'glucose_postprandial': {'min': 70, 'max': 500, 'message': 'Postprandial glucose should be between 70-500 mg/dL'},
    'Age': {'min': 18, 'max': 100, 'message': 'Age should be between 18 and 100 years'},
    'physical_activity_minutes_per_week': {'min': 0, 'max': 500, 'message': 'Activity should be between 0-500 min/week'},
    'alcohol_consumption_per_week': {'min': 0, 'max': 30, 'message': 'Alcohol should be between 0-30 drinks/week'},
    'diet_score': {'min': 0, 'max': 10, 'message': 'Diet score should be between 0 and 10'},
    'sleep_hours_per_day': {'min': 3, 'max': 12, 'message': 'Sleep should be between 3-12 hours/day'},
    'screen_time_hours_per_day': {'min': 0, 'max': 16, 'message': 'Screen time should be between 0-16 hours/day'},
}

# ============================================================================
# 3. HELPER FUNCTIONS FOR FEATURE CONVERSION
# ============================================================================

def socioeconomic_to_features(value):
    """
    Convert dropdown selection to the correct one-hot encoded features.
    Your model expects these specific columns:
    - income_level_Low
    - income_level_Lower-Middle
    - income_level_Middle
    - income_level_Upper-Middle
    - employment_status_Unemployed
    - employment_status_Retired
    - employment_status_Student
    """
    features = {
        'income_level_Low': 0,
        'income_level_Lower-Middle': 0,
        'income_level_Middle': 0,
        'income_level_Upper-Middle': 0,
        'employment_status_Unemployed': 0,
        'employment_status_Retired': 0,
        'employment_status_Student': 0,
    }
    
    mapping = {
        'low': {'income_level_Low': 1},
        'lower_middle': {'income_level_Lower-Middle': 1},
        'middle': {'income_level_Middle': 1},
        'upper_middle': {'income_level_Upper-Middle': 1},
        'unemployed': {'income_level_Low': 1, 'employment_status_Unemployed': 1},
        'employed_middle': {'income_level_Middle': 1},
        'retired_middle': {'income_level_Middle': 1, 'employment_status_Retired': 1},
        'student_low': {'income_level_Low': 1, 'employment_status_Student': 1},
    }
    
    if value in mapping:
        features.update(mapping[value])
    
    return features


def prepare_input_vector(form_data):
    """
    Convert form data to complete 39-feature vector.
    """
    # Start with mean values for ALL features
    input_data = FEATURE_MEANS.copy()
    
    # Process each form field
    for key, value in form_data.items():
        if key == 'socioeconomic':
            # Handle socioeconomic dropdown
            se_features = socioeconomic_to_features(value)
            input_data.update(se_features)
        elif key == 'gender':
            input_data['gender'] = 1 if value else 0
        elif key == 'family_history_diabetes':
            input_data['family_history_diabetes'] = 1 if value else 0
        elif key in FEATURE_COLUMNS:
            input_data[key] = float(value) if value is not None else FEATURE_MEANS[key]
    
    # Create DataFrame with EXACT column order from training
    df = pd.DataFrame([input_data])[FEATURE_COLUMNS]
    
    return df


def validate_inputs(form_data):
    """Validate all numeric inputs."""
    errors = []
    
    for field, rules in VALIDATION_RULES.items():
        if field in form_data and form_data[field] is not None:
            value = float(form_data[field])
            if value < rules['min'] or value > rules['max']:
                errors.append(rules['message'])
    
    return errors


def predict_risk_with_probas(form_data):
    """Make prediction and return all class probabilities."""
    input_df = prepare_input_vector(form_data)
    
    # Scale numeric columns
    input_scaled = input_df.copy()
    input_scaled[CLINICAL_FEATURES] = scaler.transform(input_df[CLINICAL_FEATURES])
    
    # XGBoost prediction
    risk_class = int(xgb_model.predict(input_scaled)[0])
    risk_label = target_encoder.inverse_transform([risk_class])[0]
    proba = xgb_model.predict_proba(input_scaled)[0]
    
    # Get all class probabilities
    class_probas = []
    for i, prob in enumerate(proba):
        class_name = target_encoder.inverse_transform([i])[0]
        class_probas.append((class_name, prob * 100))
    class_probas.sort(key=lambda x: x[1], reverse=True)
    
    # K-Means clustering (19 features only)
    clinical_only = input_scaled[CLINICAL_FEATURES]
    cluster = int(kmeans_model.predict(clinical_only)[0])
    cluster_label = CLUSTER_LABELS.get(cluster, f"Cluster {cluster}")
    
    return risk_class, risk_label, cluster, cluster_label, class_probas


# ============================================================================
# 4. DASH APP LAYOUT
# ============================================================================

app = dash.Dash(__name__)
server = app.server

# Build input sections dynamically
input_sections = []
for group in INPUT_GROUPS:
    fields = []
    for field in group['fields']:
        if field['type'] == 'number':
            fields.append(html.Div([
                html.Label([
                    field['label'],
                    html.Span(" ⓘ", title=field.get('help', ''), 
                             style={'cursor': 'help', 'color': '#7f8c8d'})
                ], style={'fontWeight': 'bold', 'marginTop': 10}),
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
        elif field['type'] == 'checkbox':
            fields.append(html.Div([
                html.Label([
                    field['label'],
                    html.Span(" ⓘ", title=field.get('help', ''), 
                             style={'cursor': 'help', 'color': '#7f8c8d'})
                ], style={'fontWeight': 'bold', 'marginTop': 10}),
                dcc.Checklist(
                    id=f"input-{field['id']}",
                    options=[{'label': field['checkbox_label'], 'value': True}],
                    value=[True] if field.get('default') else [],
                    style={'marginBottom': '10px'}
                )
            ]))
        elif field['type'] == 'dropdown':
            fields.append(html.Div([
                html.Label([
                    field['label'],
                    html.Span(" ⓘ", title=field.get('help', ''), 
                             style={'cursor': 'help', 'color': '#7f8c8d'})
                ], style={'fontWeight': 'bold', 'marginTop': 10}),
                dcc.Dropdown(
                    id=f"input-{field['id']}",
                    options=field['options'],
                    value=field.get('default'),
                    clearable=False,
                    style={'marginBottom': '10px'}
                )
            ]))
    
    input_sections.append(html.Div([
        html.H3(group['title'], style={'color': '#2c3e50', 'borderBottom': '2px solid #3498db', 'paddingBottom': '5px'}),
        html.Div(fields, style={'padding': '10px'})
    ], style={'marginBottom': '20px'}))

app.layout = html.Div([
    html.H1("🩺 Diabetes Risk Predictor", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
    
    html.P("Enter patient data. Hover over ⓘ for help. Missing features use population averages.",
           style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': 30}),
    
    # Input form
    html.Div(input_sections, style={
        'maxWidth': '900px', 'margin': '0 auto', 'padding': '20px',
        'backgroundColor': '#f8f9fa', 'borderRadius': '8px'
    }),
    
    # Error display area
    html.Div(id='validation-errors', style={
        'maxWidth': '900px', 'margin': '10px auto', 'color': '#e74c3c'
    }),
    
    # Predict button
    html.Div([
        html.Button('🔮 Predict Risk', id='predict-button', n_clicks=0,
                   style={'backgroundColor': '#3498db', 'color': 'white',
                          'padding': '12px 40px', 'fontSize': '16px',
                          'border': 'none', 'borderRadius': '5px',
                          'cursor': 'pointer', 'marginTop': 20})
    ], style={'textAlign': 'center'}),
    
    # Results section
    html.Div(id='prediction-output', style={
        'maxWidth': '900px', 'margin': '30px auto', 'padding': '20px',
        'backgroundColor': '#f8f9fa', 'borderRadius': '8px',
        'textAlign': 'center', 'fontSize': '18px'
    }),
    
    # Footer
    html.Footer([
        html.P("⚠️ This tool is for educational purposes only. Consult a healthcare professional for medical advice.",
               style={'color': '#95a5a6', 'fontSize': '12px', 'marginTop': 50})
    ], style={'textAlign': 'center'})
])


# ============================================================================
# 5. CALLBACK FOR PREDICTION
# ============================================================================

# Collect all input IDs for the callback
all_input_ids = []
for group in INPUT_GROUPS:
    for field in group['fields']:
        all_input_ids.append(f"input-{field['id']}")

@app.callback(
    [Output('prediction-output', 'children'),
     Output('validation-errors', 'children')],
    Input('predict-button', 'n_clicks'),
    [State(input_id, 'value') for input_id in all_input_ids]
)
def update_prediction(n_clicks, *values):
    if n_clicks == 0:
        return html.Div([
            html.P("Enter patient data and click 'Predict Risk' to see results.",
                  style={'color': '#7f8c8d'})
        ]), ""
    
    # Build form data dictionary
    form_data = {}
    for i, input_id in enumerate(all_input_ids):
        field_id = input_id.replace('input-', '')
        value = values[i]
        
        # Handle checkbox (returns list)
        if field_id in ['gender', 'family_history_diabetes']:
            form_data[field_id] = bool(value) if isinstance(value, list) else bool([value] if value else [])
        else:
            form_data[field_id] = value
    
    # Validate inputs
    errors = validate_inputs(form_data)
    if errors:
        error_div = html.Div([
            html.P(f"⚠️ {err}", style={'margin': '5px 0'}) for err in errors
        ])
        return html.Div(), error_div
    
    try:
        risk_class, risk_label, cluster, cluster_label, class_probas = predict_risk_with_probas(form_data)
        
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
        
        # Build probability bars
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
            html.P(get_interpretation(risk_label, cluster_label), style={'color': '#555'}),
            
        ]), ""
        
    except Exception as e:
        return html.Div([
            html.P("❌ Error making prediction", style={'color': '#e74c3c'}),
            html.P(str(e), style={'color': '#e74c3c', 'fontSize': '14px'})
        ]), ""


def get_interpretation(risk_label, cluster_label):
    """Generate human-readable interpretation."""
    if risk_label == 'Type 1':
        return "Type 1 diabetes is an autoimmune condition requiring insulin therapy. Consult an endocrinologist for proper management."
    
    if risk_label == 'Gestational':
        return "Gestational diabetes requires careful monitoring during pregnancy. Consult your obstetrician for appropriate care."
    
    interpretations = {
        ('No Diabetes', 'Low Risk'): "Patient shows no signs of diabetes and has healthy lifestyle indicators. Recommend maintaining current habits.",
        ('No Diabetes', 'Moderate Risk'): "No diabetes detected, but lifestyle factors suggest moderate risk. Consider increasing physical activity.",
        ('No Diabetes', 'High Risk'): "No diabetes currently, but high-risk lifestyle profile. Strongly recommend lifestyle changes.",
        ('Pre-Diabetes', 'Low Risk'): "Pre-diabetes detected despite healthy lifestyle. May indicate genetic factors. Consult healthcare provider.",
        ('Pre-Diabetes', 'Moderate Risk'): "Pre-diabetes with moderate lifestyle risk. Lifestyle improvements may prevent progression.",
        ('Pre-Diabetes', 'High Risk'): "Pre-diabetes with high-risk lifestyle. Urgent intervention recommended.",
        ('Type 2', 'Low Risk'): "Type 2 diabetes with well-managed lifestyle factors. Continue current management.",
        ('Type 2', 'Moderate Risk'): "Type 2 diabetes with room for lifestyle improvement. Focus on diet and exercise.",
        ('Type 2', 'High Risk'): "Type 2 diabetes with high-risk profile. Immediate lifestyle changes advised.",
    }
    
    key = (risk_label, cluster_label)
    if key in interpretations:
        return interpretations[key]
    
    return f"Patient classified as '{risk_label}' with a '{cluster_label}' lifestyle profile."


# ============================================================================
# 6. RUN THE APP
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run_server(host='0.0.0.0', port=port, debug=False)
