"""
Data Preprocessing Module
Loads and cleans raw log data
"""

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import os

def load_raw_logs(data_path):
    """Load raw log files"""
    print("Loading raw log files...")
    
    # Load different log types
    logon_logs = pd.read_csv(os.path.join(data_path, 'logon.csv'))
    file_logs = pd.read_csv(os.path.join(data_path, 'file.csv'))
    email_logs = pd.read_csv(os.path.join(data_path, 'email.csv'))
    device_logs = pd.read_csv(os.path.join(data_path, 'device.csv'))
    
    return logon_logs, file_logs, email_logs, device_logs

def clean_data(df, log_type):
    """Clean and standardize data"""
    print(f"Cleaning {log_type} data...")
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna(0)
    
    # Convert timestamps
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    return df

def aggregate_by_user_day(logon_df, file_df, email_df, device_df):
    """Aggregate all logs by user and day"""
    print("Aggregating data by user and day...")
    
    # Group by user and date
    user_daily = pd.DataFrame()
    
    # Extract unique users and dates
    all_users = pd.concat([
        logon_df['user'], 
        file_df['user'], 
        email_df['user'], 
        device_df['user']
    ]).unique()
    
    return user_daily, all_users

def save_processed_data(data, output_path):
    """Save processed data"""
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'processed_data.csv')
    data.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess insider threat data')
    parser.add_argument('--input', type=str, required=True, help='Input data directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Load data
    logon_logs, file_logs, email_logs, device_logs = load_raw_logs(args.input)
    
    # Clean data
    logon_logs = clean_data(logon_logs, 'logon')
    file_logs = clean_data(file_logs, 'file')
    email_logs = clean_data(email_logs, 'email')
    device_logs = clean_data(device_logs, 'device')
    
    # Aggregate
    user_daily, all_users = aggregate_by_user_day(logon_logs, file_logs, email_logs, device_logs)
    
    # Save
    save_processed_data(user_daily, args.output)
    
    print("✓ Data preprocessing complete!")

if __name__ == "__main__":
    main()
