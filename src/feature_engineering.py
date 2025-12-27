"""
Feature Engineering Module
Creates behavioral features from processed logs
"""

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import os

def extract_authentication_features(user_logs):
    """Extract login-based features"""
    features = {}
    
    # Login count
    features['login_count'] = user_logs['logon_count'].sum()
    
    # Off-hours ratio (assume 9 AM - 5 PM is work hours)
    if 'hour' in user_logs.columns:
        off_hours = user_logs[(user_logs['hour'] < 9) | (user_logs['hour'] > 17)]
        features['off_hours_ratio'] = len(off_hours) / len(user_logs) if len(user_logs) > 0 else 0
    else:
        features['off_hours_ratio'] = 0
    
    # Failed login ratio
    if 'failed_logons' in user_logs.columns:
        total_attempts = user_logs['logon_count'].sum() + user_logs['failed_logons'].sum()
        features['failed_login_ratio'] = user_logs['failed_logons'].sum() / total_attempts if total_attempts > 0 else 0
    else:
        features['failed_login_ratio'] = 0
    
    # Unique machines
    if 'pc' in user_logs.columns:
        features['unique_machines'] = user_logs['pc'].nunique()
    else:
        features['unique_machines'] = 1
    
    return features

def extract_file_access_features(user_logs):
    """Extract file access features"""
    features = {}
    
    # Files accessed
    features['files_accessed'] = user_logs['file_access_count'].sum() if 'file_access_count' in user_logs.columns else 0
    
    # Rare file score (placeholder - needs file frequency data)
    features['rare_file_score'] = 0.0
    
    # Access spike (variance in daily access)
    if 'file_access_count' in user_logs.columns:
        features['access_spike'] = user_logs['file_access_count'].std() / (user_logs['file_access_count'].mean() + 1)
    else:
        features['access_spike'] = 0
    
    # Sensitive access count
    features['sensitive_access'] = 0  # Placeholder
    
    return features

def extract_communication_features(user_logs):
    """Extract email/communication features"""
    features = {}
    
    # Emails per day
    features['emails_per_day'] = user_logs['email_count'].mean() if 'email_count' in user_logs.columns else 0
    
    # New recipient ratio
    features['new_recipient_ratio'] = 0.0  # Placeholder
    
    # After-hours communication
    features['after_hours_comm'] = 0.0  # Placeholder
    
    return features

def extract_device_features(user_logs):
    """Extract USB/device features"""
    features = {}
    
    # USB insertions
    features['usb_insertions'] = user_logs['usb_count'].sum() if 'usb_count' in user_logs.columns else 0
    
    # Data copied
    features['data_copied'] = user_logs['data_transferred'].sum() if 'data_transferred' in user_logs.columns else 0
    
    # First-time device
    features['first_time_device'] = 0  # Placeholder
    
    return features

def extract_temporal_features(user_logs):
    """Extract time-based variance features"""
    features = {}
    
    # Activity variance
    if 'total_activity' in user_logs.columns:
        features['activity_variance'] = user_logs['total_activity'].std() / (user_logs['total_activity'].mean() + 1)
    else:
        features['activity_variance'] = 0
    
    # Baseline deviation
    features['baseline_deviation'] = 0.0  # Placeholder
    
    return features

def engineer_features(data):
    """Create all features for each user"""
    print("Engineering features...")
    
    feature_list = []
    
    # Group by user
    for user_id, user_logs in data.groupby('user'):
        user_features = {'user_id': user_id}
        
        # Extract all feature categories
        user_features.update(extract_authentication_features(user_logs))
        user_features.update(extract_file_access_features(user_logs))
        user_features.update(extract_communication_features(user_logs))
        user_features.update(extract_device_features(user_logs))
        user_features.update(extract_temporal_features(user_logs))
        
        feature_list.append(user_features)
    
    features_df = pd.DataFrame(feature_list)
    return features_df

def save_features(features, output_path):
    """Save engineered features"""
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'features.csv')
    features.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")
    print(f"Total features: {len(features.columns) - 1}")  # -1 for user_id

def main():
    parser = argparse.ArgumentParser(description='Engineer features for insider threat detection')
    parser.add_argument('--input', type=str, required=True, help='Input processed data directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Load processed data
    input_file = os.path.join(args.input, 'processed_data.csv')
    data = pd.read_csv(input_file)
    
    # Engineer features
    features = engineer_features(data)
    
    # Save features
    save_features(features, args.output)
    
    print("✓ Feature engineering complete!")

if __name__ == "__main__":
    main()
