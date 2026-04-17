# Diabetes Risk Segmentation & Decision support system 


**Institution:** Belgium Campus ITversity 

**Module:** MLG382 – Machine Learning Project

**Methodology:** CRISP-DM Framework


---

## Project Overview

This project presents an end-to-end **Diabetes Risk Segmentation & Decision Support System** developed for BC Analytics, a health-tech startup focused on improving patient outcomes through data-driven healthcare solutions.

The system leverages supervised and unsupervised machine learning techniques to:

* Predict patient diabetes risk stage
* Segment patients into lifestyle-based clusters
* Identify key risk drivers using explainable AI (SHAP)
* Provide personalized health recommendations
* Deliver insights through an interactive Dash web application

---

## Business Problem

Healthcare providers often:

* React too late to diabetes warning signs
* Lack visibility into lifestyle factors influencing patient risk
* Have limited tools for real-time data-driven decision support

This project addresses those challenges by providing clinicians with an intelligent decision support platform for early intervention and patient stratification.

---

## Objectives

1. **Risk Classification**

   * Predict patient diabetes stage

2. **Patient Segmentation**

   * Group patients into lifestyle-based clusters using K-Means

3. **Key Driver Analysis**

   * Explain predictions and identify influential features using SHAP

4. **Decision Support Dashboard**

   * Deliver predictions, explanations, segmentation, and recommendations interactively

---

## Machine Learning Models Used

### Classification Models

* Decision Tree Classifier
* Random Forest Classifier
* XGBoost Classifier

### Clustering Model

* K-Means Clustering (`k=3`)

### Explainability

* SHAP (Shapley Additive Explanations)

---

## Web Application Features

### Diabetes Risk Prediction
- Determine likely diabetes status.
  
### Lifestyle Clustering
- Match health metrics to cluster (Low Risk - High Risk).
  
### Probability Distribution
- Give a percentage score for how confident the model is in its classification.
  
### Interpretation
- Comments on Prediction results and gives advice for possible benifits.
    
---

## Project Structure


```
MLG382-ASSIGNMENT1---GROUP-11/
│
├── artifacts/
│   ├── cluster_labels.pkl
│   ├── Decision_tree_model.pkl
│   ├── feature_columns.pkl
│   ├── feature_means.pkl
│   ├── kmeans_model.pkl
│   ├── Random_forrest.pkl
│   ├── scaler.pkl
│   ├── target_encoder.pkl
│   ├── xgboost_model.pkl
│   └── xgboost_model_initial.pkl
│
├── data/
│   ├── processed/
│   │   ├── X_test_scaled.csv
│   │   ├── X_test.csv
│   │   ├── X_train_scaled.csv
│   │   ├── X_train.csv
│   │   ├── X_val_scaled.csv
│   │   ├── X_val.csv
│   │   ├── y_test.csv
│   │   ├── y_train.csv
│   │   └── y_val.csv
│   │
│   └── Diabetes_and_LifeStyle_Dataset_.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── cluster_modelling.ipynb
│   ├── regression_modelling.ipynb
│   └── SHAP_Analysis.ipynb
│
├── src/
│   ├── preprocess_data.ipynb
│   └── web_app.py
│
├── README.md
├── requirements.txt
│
├── .gitattributes
└── .gitignore

```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/m-vs-0502/MLG382-Assignment1---Group-11.git
cd [filename]  >> optional
```

---


### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application Locally

```bash
python app/app.py
```

---

## Deployment

The application is deployed using **Render**.

**Live Web App:**
https://diabetes-diagnosis-ml-group-11.onrender.com/ 

---

## Technical Documentation

### Technical Report (ondrive link)

https://belgiumcampusacza-my.sharepoint.com/:b:/g/personal/601847_student_belgiumcampus_ac_za/IQBO6UEj9nfySqEy2IwEuChGAduX01TBfSv5DQHFIiGg5wQ?e=HhOQ4U


---

## Dataset Information

**Dataset Used:** `Diabetes_and_LifeStyle_Dataset_`

**File_Type:** .csv


The dataset contains demographic, lifestyle, clinical, and metabolic indicators used to predict diabetes risk and support patient segmentation.


### Input Features

#### Demographic Features
- Age  
- Gender  
- Ethnicity  
- Education Level  
- Income Level  
- Employment Status  

#### Lifestyle Features
- Smoking Status  
- Alcohol Consumption Per Week  
- Physical Activity Minutes Per Week  
- Diet Score  
- Sleep Hours Per Day  
- Screen Time Hours Per Day  

#### Medical History Features
- Family History Diabetes  
- Hypertension History  
- Cardiovascular History  

#### Clinical / Biomarker Features
- BMI  
- Waist-to-Hip Ratio  
- Systolic Blood Pressure  
- Diastolic Blood Pressure  
- Heart Rate  
- Total Cholesterol  
- HDL Cholesterol  
- LDL Cholesterol  
- Triglycerides  
- Fasting Glucose  
- Postprandial Glucose  
- Insulin Level  
- HbA1c  

---

### Target Variables

- **Primary Classification Target:** `diabetes_stage`  
- **Secondary Indicator:** `diagnosed_diabetes`  
- **Risk Score Reference:** `diabetes_risk_score`

---

## CRISP-DM Implementation

### 1. Business Understanding

Defined healthcare risk prediction and segmentation objectives.

### 2. Data Understanding

Explored diabetes types and feature distributions.

### 3. Data Preparation

Cleaned data, encoded categoricals, scaled numerical features.

### 4. Modeling

Trained classification and clustering models.

### 5. Evaluation

Compared classification metrics and cluster interpretability.

### 6. Deployment

Developed and deployed interactive Dash web application.

---

## Academic use

Academic Project – Belgium Campus ITversity

---

## Acknowledgements

* Belgium Campus ITversity
* BC Analytics Project Documentation
* Scikit-Learn Documentation
* XGBoost Documentation
* SHAP Documentation
