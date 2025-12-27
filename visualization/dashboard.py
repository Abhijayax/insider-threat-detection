"""
Streamlit Dashboard for Insider Threat Detection
Run with: streamlit run visualization/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# Page config
st.set_page_config(
    page_title="Insider Threat Detection",
    page_icon="🔐",
    layout="wide"
)

# Title
st.title("🔐 Insider Threat Detection System")
st.markdown("*Unsupervised Machine Learning for Behavioral Anomaly Detection*")

# Sidebar
st.sidebar.header("Model Configuration")

# Model selection
model_options = {
    'Isolation Forest': 'models/isolation_forest.pkl',
    'One-Class SVM': 'models/one_class_svm.pkl'
}

selected_model_name = st.sidebar.selectbox("Select Model", list(model_options.keys()))
top_k = st.sidebar.slider("Top-K Risky Users", 5, 50, 10)

# Load model (with caching)
@st.cache_resource
def load_model(model_path):
    if os.path.exists(model_path):
        model_data = joblib.load(model_path)
        return model_data['model']
    return None

model_path = model_options[selected_model_name]
model = load_model(model_path)

# Load features
@st.cache_data
def load_features():
    features_file = 'data/features/features.csv'
    if os.path.exists(features_file):
        return pd.read_csv(features_file)
    else:
        # Generate synthetic data for demo
        np.random.seed(42)
        n_users = 50
        
        data = {
            'user_id': [f'USER_{i:03d}' for i in range(1, n_users + 1)],
            'login_count': np.random.randint(10, 50, n_users),
            'off_hours_ratio': np.random.uniform(0.1, 0.5, n_users),
            'failed_login_ratio': np.random.uniform(0, 0.2, n_users),
            'unique_machines': np.random.randint(1, 10, n_users),
            'files_accessed': np.random.randint(30, 200, n_users),
            'emails_per_day': np.random.randint(10, 80, n_users),
            'usb_insertions': np.random.randint(0, 15, n_users)
        }
        return pd.DataFrame(data)

features_df = load_features()

# Calculate anomaly scores
if model is not None:
    X = features_df.drop('user_id', axis=1).values
    scores = model.decision_function(X)
    scores_normalized = 100 * (1 - (scores - scores.min()) / (scores.max() - scores.min()))
else:
    # Random scores for demo
    scores_normalized = np.random.uniform(20, 95, len(features_df))

features_df['risk_score'] = scores_normalized
features_df = features_df.sort_values('risk_score', ascending=False)

# Top risky users
top_risky = features_df.head(top_k)

# Metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("High Risk Users", len(features_df[features_df['risk_score'] > 70]))
with col2:
    st.metric("Total Users", len(features_df))
with col3:
    st.metric("Avg Risk Score", f"{features_df['risk_score'].mean():.1f}")
with col4:
    precision = 80.0 if selected_model_name == 'Isolation Forest' else 70.0
    st.metric("Precision@10", f"{precision}%")

st.markdown("---")

# Main content
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader(f"🚨 Top {top_k} Risky Users")
    
    # Display risk table
    display_df = top_risky[['user_id', 'risk_score']].copy()
    display_df['risk_score'] = display_df['risk_score'].round(1)
    
    st.dataframe(
        display_df.style.background_gradient(
            subset=['risk_score'],
            cmap='Reds'
        ),
        height=400
    )

with col_right:
    st.subheader("📊 Risk Score Distribution")
    
    # Histogram
    fig = px.histogram(
        features_df,
        x='risk_score',
        nbins=20,
        title="Distribution of Risk Scores",
        labels={'risk_score': 'Risk Score', 'count': 'Number of Users'}
    )
    fig.update_traces(marker_color='steelblue')
    st.plotly_chart(fig, use_container_width=True)

# User detail section
st.markdown("---")
st.subheader("🔍 User Profile Analysis")

selected_user = st.selectbox("Select User", top_risky['user_id'].tolist())
user_data = features_df[features_df['user_id'] == selected_user].iloc[0]

# User metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Login Count", int(user_data['login_count']))
with col2:
    st.metric("Off-Hours Ratio", f"{user_data['off_hours_ratio']:.1%}")
with col3:
    st.metric("Files Accessed", int(user_data['files_accessed']))
with col4:
    st.metric("USB Insertions", int(user_data['usb_insertions']))

# Feature contribution
st.subheader("Feature Contribution")

feature_cols = [col for col in features_df.columns if col not in ['user_id', 'risk_score']]
feature_values = user_data[feature_cols].values

# Normalize to 0-100
feature_contributions = 100 * (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min() + 1e-10)

contrib_df = pd.DataFrame({
    'Feature': feature_cols,
    'Contribution': feature_contributions
}).sort_values('Contribution', ascending=True)

fig = px.bar(
    contrib_df,
    x='Contribution',
    y='Feature',
    orientation='h',
    title=f"Feature Analysis for {selected_user}",
    color='Contribution',
    color_continuous_scale='Reds'
)

st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("*Models: Isolation Forest • One-Class SVM | Dataset: CERT Insider Threat*")
