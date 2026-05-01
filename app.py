import streamlit as st
import pickle
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score
import random

# Load model
with open('model.pkl', 'rb') as f:
    model_dict = pickle.load(f)
scaler = model_dict['scaler']
level1_model = model_dict['level1_model']
meta_model = model_dict['super_meta_model']

# Load head data to fit encoders
csv_path = 'Clean_Data.csv'
df = pd.read_csv(csv_path)
df.columns = df.columns.str.lower()

# Map to model features
feature_map = {
    'subcounty': 'subcounty',
    'age': 'age',
    'gender': 'gender',
    'marital_status': 'marital_status',
    'occupation': 'occupation',
    'education_level': 'education_level',
    'person_present': 'person_present',
    'differentiated_care': 'differentiated_care',
    'months_on_art': 'months_on_art',
    'time_to_linkage': 'time_to_linkage',
    'baseline_who_staging': 'baseline_who_staging',
    'current_who_staging': 'current_who_staging',
    'start_regimen_line': 'start_regimen_line',
    'current_regimen_line': 'current_regimen_line',
    'arv_adherence': 'arv_adherence',
    'dm_screening_result': 'dm_screening_result',
    'nutrition_status': 'nutrition_status',
    'tb_status_within_last_6_months': 'tb_status_within_last_6_months',
    'visit_type': 'visit_type',
    'next_appointment_reason': 'next_appointment_reason',
    'start_regimen': 'start_regimen',
    'current_regimen': 'current_regimen',
    'appointment_consent': 'appointment_consent',
    'age_category': 'age',
    'months_on_art_cat': 'months_on_art'
}

# Get unique values for categoricals
categorical_features = [f for f in scaler.feature_names_in_ if f in ['subcounty', 'gender', 'marital_status', 'occupation', 'education_level', 'person_present', 'differentiated_care', 'time_to_linkage', 'baseline_who_staging', 'current_who_staging', 'start_regimen_line', 'current_regimen_line', 'arv_adherence', 'dm_screening_result', 'nutrition_status', 'tb_status_within_last_6_months', 'visit_type', 'next_appointment_reason', 'start_regimen', 'current_regimen', 'appointment_consent', 'age_category', 'months_on_art_cat']]
numerical_features = [f for f in scaler.feature_names_in_ if f not in categorical_features]

# Fit encoders
encoders = {}
for col in categorical_features:
    if col in df.columns:
        encoders[col] = LabelEncoder()
        encoders[col].fit(df[col].astype(str))
    else:
        # Handle one-hot encoded columns
        prefix = f"{col}_"
        candidates = [c for c in df.columns if c.startswith(prefix)]
        if candidates:
            unique_vals = [c.replace(prefix, '') for c in candidates]
            encoders[col] = LabelEncoder()
            encoders[col].fit(unique_vals)

# Define bins for months_on_art_cat
bins = [0, 12, 24, 36, 48, 60, 120, float('inf')]
labels = ['0-12', '12-24', '24-36', '36-48', '48-60', '60-120', '120+']

# For months_on_art_cat
if 'months_on_art_cat' in categorical_features:
    df['months_on_art_cat'] = pd.cut(df['months_on_art'], bins=bins, labels=labels, right=False)
    encoders['months_on_art_cat'] = LabelEncoder()
    encoders['months_on_art_cat'].fit(labels)

# For age_category
age_category_labels = []
if 'age_groups' in df.columns:
    age_category_labels = sorted(df['age_groups'].dropna().astype(str).unique().tolist())
elif any(c.startswith('age_range_') for c in df.columns):
    age_category_labels = sorted([c.replace('age_range_', '') for c in df.columns if c.startswith('age_range_')])
else:
    age_category_labels = ['0-12', '12-24', '24-36', '36-48', '48-60', '60-120', '120+']

if 'age_category' in categorical_features:
    encoders['age_category'] = LabelEncoder()
    encoders['age_category'].fit(age_category_labels)

def age_to_category(age):
    try:
        age = float(age)
    except Exception:
        return age_category_labels[0]

    for label in age_category_labels:
        if '+' in label:
            low = float(label.replace('+', ''))
            if age >= low:
                return label
        elif '-' in label:
            low, high = [float(x) for x in label.split('-')]
            if low <= age <= high:
                return label

    return age_category_labels[0]

def safe_encoder_transform(encoder, value):
    try:
        if value is None or str(value).strip() == '':
            if hasattr(encoder, 'classes_') and len(encoder.classes_) > 0:
                return encoder.transform([encoder.classes_[0]])[0]
            return 0
        return encoder.transform([str(value)])[0]
    except Exception as e:
        if hasattr(encoder, 'classes_') and len(encoder.classes_) > 0:
            return encoder.transform([encoder.classes_[0]])[0]
        return 0

def preprocess_row(row_dict):
    processed = []
    for f in scaler.feature_names_in_:
        if f == 'months_on_art_cat':
            months = float(row_dict.get('months_on_art', 0.0) or 0.0)
            cat = pd.cut([months], bins=bins, labels=labels, right=False)[0]
            processed.append(safe_encoder_transform(encoders['months_on_art_cat'], cat))
            continue

        if f == 'age_category':
            cat_val = row_dict.get('age_category', '')
            if not cat_val or str(cat_val).strip() == '':
                cat_val = age_to_category(row_dict.get('age', 0))
            processed.append(safe_encoder_transform(encoders['age_category'], cat_val))
        elif f in encoders:
            processed.append(safe_encoder_transform(encoders[f], row_dict.get(f, '')))
        else:
            val = row_dict.get(f, 0.0)
            try:
                processed.append(float(val) if val != '' and pd.notnull(val) else 0.0)
            except (ValueError, TypeError):
                processed.append(0.0)

    return np.array(processed, dtype=float)

def transform_with_feature_names(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return scaler.transform(pd.DataFrame(arr, columns=scaler.feature_names_in_))

def meta_model_predict_proba(level1_probs):
    return meta_model.predict_proba(level1_probs)

def meta_model_predict(level1_probs):
    return meta_model.predict(level1_probs)

def compute_local_sensitivity(input_row_dict):
    """
    Compute prediction sensitivity for each feature in the given input.
    Shows how much the suppressed probability changes when each feature is perturbed.
    """
    base_processed = preprocess_row(input_row_dict)
    base_scaled = transform_with_feature_names(base_processed.reshape(1, -1))
    base_level1 = level1_model.predict_proba(base_scaled)
    base_proba = meta_model_predict_proba(base_level1)[0][0]

    sensitivity = {}
    features_to_test = [f for f in scaler.feature_names_in_ if f != 'months_on_art_cat']

    for feat_idx, f in enumerate(features_to_test):
        if f in encoders:
            # For categorical, try all possible values
            possible_values = list(encoders[f].classes_)
            prob_shifts = []
            for val in possible_values:
                test_row = input_row_dict.copy()
                test_row[f] = val
                test_processed = preprocess_row(test_row)
                test_scaled = transform_with_feature_names(test_processed.reshape(1, -1))
                test_level1 = level1_model.predict_proba(test_scaled)
                test_proba = meta_model_predict_proba(test_level1)[0][0]
                prob_shifts.append(abs(test_proba - base_proba))
            sensitivity[f] = max(prob_shifts)
        else:
            # For numerical, test min/max bounds from data
            col_name = feature_map.get(f, f)
            if col_name in df.columns:
                min_val = df[col_name].min()
                max_val = df[col_name].max()

                prob_shifts = []
                for val in [min_val, max_val]:
                    test_row = input_row_dict.copy()
                    test_row[f] = val
                    test_processed = preprocess_row(test_row)
                    test_scaled = transform_with_feature_names(test_processed.reshape(1, -1))
                    test_level1 = level1_model.predict_proba(test_scaled)
                    test_proba = meta_model_predict_proba(test_level1)[0][0]
                    prob_shifts.append(abs(test_proba - base_proba))
                sensitivity[f] = max(prob_shifts)
            else:
                sensitivity[f] = 0.0

    total_sensitivity = sum(sensitivity.values())
    if total_sensitivity > 0:
        return {k: round((v / total_sensitivity) * 100, 2) for k, v in sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)}
    return {}

# Streamlit app
st.title("Viral Load Suppression Prediction")

st.sidebar.header("Patient Information")

if sklearn.__version__ != '1.7.2':
    st.sidebar.warning(f"Model was trained with scikit-learn 1.7.2, but current runtime is {sklearn.__version__}. This may affect prediction quality.")

input_data = {}

# Numerical inputs
for f in numerical_features:
    if f == 'age':
        input_data[f] = st.sidebar.number_input(f"Age", min_value=0, max_value=120, value=30)
    elif f == 'months_on_art':
        input_data[f] = st.sidebar.number_input(f"Months on ART", min_value=0, max_value=500, value=12)
    else:
        input_data[f] = st.sidebar.number_input(f.replace('_', ' ').title(), value=0.0)

# Categorical inputs
for f in categorical_features:
    if f == 'months_on_art_cat':
        continue  # derived
    if f == 'age_category':
        continue  # derived
    if f in encoders:
        options = list(encoders[f].classes_)
        input_data[f] = st.sidebar.selectbox(f.replace('_', ' ').title(), options, index=0)
    else:
        input_data[f] = st.sidebar.text_input(f.replace('_', ' ').title(), "")

if st.sidebar.button("Predict"):
    # Preprocess
    processed = preprocess_row(input_data)
    processed = np.array(processed, dtype=float).reshape(1, -1)
    processed_scaled = transform_with_feature_names(processed)

    # Predict
    level1_pred = level1_model.predict_proba(processed_scaled)
    final_pred = meta_model_predict_proba(level1_pred)
    pred_class = meta_model_predict(level1_pred)[0]
    prob_suppressed = final_pred[0][0] * 100
    prob_unsuppressed = final_pred[0][1] * 100

    prediction = 'Suppressed' if pred_class == 0 else 'Unsuppressed'

    st.header("Prediction Result")
    st.write(f"**Prediction:** {prediction}")
    st.write(f"**Probability Suppressed:** {prob_suppressed:.2f}%")
    st.write(f"**Probability Unsuppressed:** {prob_unsuppressed:.2f}%")

    st.header("Input Data")
    st.json(input_data)

    st.header("Feature Strength")
    feature_strength = compute_local_sensitivity(input_data)
    if feature_strength:
        strength_df = pd.DataFrame(
            sorted(feature_strength.items(), key=lambda x: x[1], reverse=True),
            columns=['Feature', 'Relative Importance']
        )
        st.table(strength_df)
    else:
        st.info("Feature strength could not be computed for this input.")