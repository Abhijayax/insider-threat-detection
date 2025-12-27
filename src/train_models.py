"""
Model Training Module
Trains unsupervised anomaly detection models
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import argparse
import os
import time

def load_features(data_path):
    """Load engineered features"""
    features_file = os.path.join(data_path, 'features.csv')
    df = pd.read_csv(features_file)
    
    # Separate user IDs from features
    user_ids = df['user_id'].values
    X = df.drop('user_id', axis=1).values
    
    return X, user_ids, df.drop('user_id', axis=1).columns.tolist()

def train_isolation_forest(X, contamination=0.1):
    """Train Isolation Forest model"""
    print("Training Isolation Forest...")
    
    start_time = time.time()
    
    # Create pipeline with scaling
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Train
    model.fit(X)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, training_time

def train_one_class_svm(X, nu=0.1):
    """Train One-Class SVM model"""
    print("Training One-Class SVM...")
    
    start_time = time.time()
    
    # Create pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=nu
        ))
    ])
    
    # Train
    model.fit(X)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, training_time

def save_model(model, model_name, output_path, training_time):
    """Save trained model"""
    os.makedirs(output_path, exist_ok=True)
    model_file = os.path.join(output_path, f'{model_name}.pkl')
    
    # Save model with metadata
    model_data = {
        'model': model,
        'model_name': model_name,
        'training_time': training_time,
        'timestamp': pd.Timestamp.now()
    }
    
    joblib.dump(model_data, model_file)
    print(f"Model saved to {model_file}")

def main():
    parser = argparse.ArgumentParser(description='Train anomaly detection models')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['isolation_forest', 'one_class_svm', 'all'],
                        help='Model to train')
    parser.add_argument('--data', type=str, required=True, help='Features directory')
    parser.add_argument('--output', type=str, default='models', help='Output directory')
    parser.add_argument('--contamination', type=float, default=0.1, 
                        help='Expected proportion of anomalies')
    
    args = parser.parse_args()
    
    # Load features
    X, user_ids, feature_names = load_features(args.data)
    print(f"Loaded {X.shape[0]} users with {X.shape[1]} features")
    
    # Train models
    if args.model == 'isolation_forest' or args.model == 'all':
        model, training_time = train_isolation_forest(X, args.contamination)
        save_model(model, 'isolation_forest', args.output, training_time)
    
    if args.model == 'one_class_svm' or args.model == 'all':
        model, training_time = train_one_class_svm(X, args.contamination)
        save_model(model, 'one_class_svm', args.output, training_time)
    
    print("✓ Model training complete!")

if __name__ == "__main__":
    main()
