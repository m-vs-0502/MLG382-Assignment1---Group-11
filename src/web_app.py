import os
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import joblib
import pickle

# ============================================================================
# 1. LOAD ARTIFACTS (With proper Render path handling)
# ============================================================================

# Render mounts your repo at /opt/render/project/src/
# Your artifacts folder is at the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')

print(f"Loading artifacts from: {ARTIFACTS_DIR}")

# Load models and encoders
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
    print(f"   - Feature count: {len(FEATURE_COLUMNS)}")
    print(f"   - Target classes: {list(target_encoder.classes_)}")

except Exception as e:
    print(f"❌ Error loading artifacts: {e}")
    raise

# ============================================================================
# 2. FEATURE DEFINITIONS
# ============================================================================

# The 19 clinical features used by K-Means (must match exactly)
CLINICAL_FEATURES = [
    'Age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week',
    'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
    'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
    'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides',
    'glucose_fasting', 'glucose_postprandial', 'insulin_level', 'hba1c'
]

# Top 10 features from XGBoost importance (from your notebook)
TOP_10_FEATURES = [
    'hba1c',
    'glucose_fasting',
    'glucose_postprandial',
    'gender',
    'family_history_diabetes',
    'Age',
    'income_level_Middle',
    'physical_activity_minutes_per_week',
    'income_level_Low',
    'employment_status_Unemployed'
]

# Human-readable names for the input form
FEATURE_DISPLAY_NAMES = {
    'hba1c': 'HbA1c (%)',
    'glucose_fasting': 'Fasting Glucose (mg/dL)',
    'glucose_postprandial': 'Postprandial Glucose (mg/dL)',
    'gender': 'Gender (1=Male, 0=Female)',
    'family_history_diabetes': 'Family History of Diabetes (1=Yes, 0=No)',
    'Age': 'Age (years)',
    'income_level_Middle': 'Middle Income (1=Yes, 0=No)',
    'physical_activity_minutes_per_week': 'Physical Activity (min/week)',
    'income_level_Low': 'Low Income (1=Yes, 0=No)',
    'employment_status_Unemployed': 'Unemployed (1=Yes, 0=No)'
}

# Default/placeholder values for the form
DEFAULT_VALUES = {
    'hba1c': 5.7,
    'glucose_fasting': 95,
    'glucose_postprandial': 140,
    'gender': 0,
    'family_history_diabetes': 0,
    'Age': 50,
    'income_level_Middle': 0,
    'physical_activity_minutes_per_week': 150,
    'income_level_Low': 0,
    'employment_status_Unemployed': 0
}

# ============================================================================
# 3. PREDICTION HELPER FUNCTIONS
# ============================================================================

def prepare_input_vector(form_values):
    """
    Takes the 10 user-provided values and creates a complete 39-feature vector
    by filling missing features with their training set means.
    """
    # Start with mean values for ALL features
    input_data = FEATURE_MEANS.copy()
    
    # Override with user-provided values
    for feature, value in form_values.items():
        if feature in input_data:
            input_data[feature] = float(value)
    
    # Create DataFrame with EXACT column order from training
    df = pd.DataFrame([input_data])[FEATURE_COLUMNS]
    
    return df


def predict_risk(form_values):
    """
    Main prediction function.
    Returns: (risk_class, risk_label, cluster, cluster_label, confidence)
    """
    # 1. Prepare input vector
    input_df = prepare_input_vector(form_values)
    
    # 2. Scale numeric columns (same as training)
    input_scaled = input_df.copy()
    input_scaled[CLINICAL_FEATURES] = scaler.transform(input_df[CLINICAL_FEATURES])
    
    # 3. XGBoost prediction (uses all 39 features)
    risk_class = int(xgb_model.predict(input_scaled)[0])
    risk_label = target_encoder.inverse_transform([risk_class])[0]
    
    # Get prediction probabilities for confidence
    proba = xgb_model.predict_proba(input_scaled)[0]
    confidence = float(proba[risk_class] * 100)
    
    # 4. K-Means clustering (uses ONLY 19 clinical features)
    clinical_only = input_scaled[CLINICAL_FEATURES]
    cluster = int(kmeans_model.predict(clinical_only)[0])
    cluster_label = CLUSTER_LABELS.get(cluster, f"Cluster {cluster}")
    
    return risk_class, risk_label, cluster, cluster_label, confidence


# ============================================================================
# 4. DASH APP LAYOUT
# ============================================================================

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("🩺 Diabetes Risk Predictor", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
    
    html.P("Enter patient data for the 10 most important risk factors. Other features will use population averages.",
           style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': 30}),
    
    # Input form
    html.Div([
        html.Div([
            html.Label(FEATURE_DISPLAY_NAMES.get(feat, feat), 
                      style={'fontWeight': 'bold', 'marginTop': 10}),
            dcc.Input(
                id=f'input-{feat}',
                type='number',
                value=DEFAULT_VALUES.get(feat, 0),
                style={'width': '100%', 'padding': '8px', 'marginBottom': '10px',
                       'border': '1px solid #ddd', 'borderRadius': '4px'}
            )
        ]) for feat in TOP_10_FEATURES
    ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(2, 1fr)', 
              'gap': '20px', 'maxWidth': '800px', 'margin': '0 auto'}),
    
    # Predict button
    html.Div([
        html.Button('Predict Risk', id='predict-button', n_clicks=0,
                   style={'backgroundColor': '#3498db', 'color': 'white',
                          'padding': '12px 30px', 'fontSize': '16px',
                          'border': 'none', 'borderRadius': '5px',
                          'cursor': 'pointer', 'marginTop': 30})
    ], style={'textAlign': 'center'}),
    
    # Results section
    html.Div(id='prediction-output', style={
        'maxWidth': '800px', 'margin': '30px auto', 'padding': '20px',
        'backgroundColor': '#f8f9fa', 'borderRadius': '8px',
        'textAlign': 'center', 'fontSize': '18px'
    }),
    
    # Footer
    html.Footer([
        html.P("Note: This tool is for educational purposes only. Consult a healthcare professional for medical advice.",
               style={'color': '#95a5a6', 'fontSize': '12px', 'marginTop': 50})
    ], style={'textAlign': 'center'})
])


# ============================================================================
# 5. CALLBACK FOR PREDICTION
# ============================================================================

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [State(f'input-{feat}', 'value') for feat in TOP_10_FEATURES]
)
def update_prediction(n_clicks, *values):
    if n_clicks == 0:
        return html.Div([
            html.P("Enter patient data and click 'Predict Risk' to see results.",
                  style={'color': '#7f8c8d'})
        ])
    
    # Build form values dictionary
    form_values = {feat: val for feat, val in zip(TOP_10_FEATURES, values)}
    
    try:
        risk_class, risk_label, cluster, cluster_label, confidence = predict_risk(form_values)
        
        # Color-code based on risk
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
        
        return html.Div([
            html.H3("Prediction Results", style={'marginBottom': 20}),
            
            html.Div([
                html.P("Diabetes Risk Classification:", 
                      style={'fontWeight': 'bold', 'marginBottom': 5}),
                html.P(risk_label, 
                      style={'fontSize': '24px', 'fontWeight': 'bold',
                             'color': risk_colors.get(risk_label, '#2c3e50')}),
                html.P(f"Confidence: {confidence:.1f}%",
                      style={'color': '#7f8c8d'})
            ], style={'marginBottom': 20}),
            
            html.Div([
                html.P("Lifestyle/Clinical Cluster:", 
                      style={'fontWeight': 'bold', 'marginBottom': 5}),
                html.P(cluster_label, 
                      style={'fontSize': '20px', 'fontWeight': 'bold',
                             'color': cluster_colors.get(cluster_label, '#2c3e50')})
            ], style={'marginBottom': 20}),
            
            html.Hr(),
            
            html.P("Interpretation:", style={'fontWeight': 'bold', 'marginTop': 20}),
            html.P(get_interpretation(risk_label, cluster_label, confidence),
                  style={'color': '#555', 'lineHeight': '1.6'})
        ])
        
    except Exception as e:
        return html.Div([
            html.P("❌ Error making prediction", style={'color': '#e74c3c'}),
            html.P(str(e), style={'color': '#e74c3c', 'fontSize': '14px'})
        ])


def get_interpretation(risk_label, cluster_label, confidence):
    """Generate human-readable interpretation."""
    interpretations = {
        ('No Diabetes', 'Low Risk'): "Patient shows no signs of diabetes and has healthy lifestyle indicators. Recommend maintaining current habits with regular check-ups.",
        ('No Diabetes', 'Moderate Risk'): "No diabetes detected, but lifestyle factors suggest moderate risk. Consider increasing physical activity and monitoring diet.",
        ('No Diabetes', 'High Risk'): "No diabetes currently, but high-risk lifestyle profile. Strongly recommend lifestyle changes and regular glucose monitoring.",
        ('Pre-Diabetes', 'Low Risk'): "Pre-diabetes detected despite healthy lifestyle. This may indicate genetic factors. Consult healthcare provider for management plan.",
        ('Pre-Diabetes', 'Moderate Risk'): "Pre-diabetes with moderate lifestyle risk. Lifestyle improvements may help prevent progression to Type 2 diabetes.",
        ('Pre-Diabetes', 'High Risk'): "Pre-diabetes with high-risk lifestyle. Urgent lifestyle intervention recommended to prevent progression.",
        ('Type 2', 'Low Risk'): "Type 2 diabetes with well-managed lifestyle factors. Continue current management and regular monitoring.",
        ('Type 2', 'Moderate Risk'): "Type 2 diabetes with room for lifestyle improvement. Focus on diet, exercise, and medication adherence.",
        ('Type 2', 'High Risk'): "Type 2 diabetes with high-risk profile. Immediate lifestyle changes and medical consultation strongly advised.",
        ('Type 1', _): "Type 1 diabetes is an autoimmune condition requiring insulin therapy. Consult endocrinologist for management.",
        ('Gestational', _): "Gestational diabetes requires monitoring during pregnancy. Consult obstetrician for appropriate care."
    }
    
    # Check for specific combination first
    key = (risk_label, cluster_label)
    if key in interpretations:
        return interpretations[key]
    
    # Fallback
    return f"Based on the model, this patient is classified as '{risk_label}' with a '{cluster_label}' lifestyle profile. Clinical correlation is recommended."


# ============================================================================
# 6. RUN THE APP
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run_server(host='0.0.0.0', port=port, debug=False)
