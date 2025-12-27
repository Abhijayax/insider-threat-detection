"""
Model Evaluation Module
Evaluates trained models and generates metrics
"""

import pandas as pd
import numpy as np
import joblib
import json
import argparse
import os
from sklearn.metrics import precision_score, recall_score

def load_model(model_path):
    """Load trained model"""
    model_data = joblib.load(model_path)
    return model_data['model'], model_data['model_name']

def load_features_and_labels(data_path):
    """Load features and optional ground truth labels"""
    features_file = os.path.join(data_path, 'features.csv')
    df = pd.read_csv(features_file)
    
    user_ids = df['user_id'].values
    X = df.drop('user_id', axis=1).values
    
    # Load labels if available (for evaluation)
    labels = None
    labels_file = os.path.join(data_path, 'labels.csv')
    if os.path.exists(labels_file):
        labels_df = pd.read_csv(labels_file)
        labels = labels_df['is_threat'].values
    
    return X, user_ids, labels

def predict_anomaly_scores(model, X):
    """Get anomaly scores from model"""
    # Get decision function (lower = more anomalous)
    scores = model.decision_function(X)
    
    # Convert to 0-100 scale (higher = more anomalous)
    scores_normalized = 100 * (1 - (scores - scores.min()) / (scores.max() - scores.min()))
    
    return scores_normalized

def calculate_precision_at_k(scores, labels, k_values=[5, 10, 20, 50]):
    """Calculate Precision@K"""
    if labels is None:
        print("No ground truth labels available")
        return {}
    
    # Sort by score (descending)
    sorted_indices = np.argsort(scores)[::-1]
    
    precision_at_k = {}
    for k in k_values:
        if k > len(scores):
            continue
        
        top_k_indices = sorted_indices[:k]
        top_k_labels = labels[top_k_indices]
        
        precision = np.sum(top_k_labels) / k
        precision_at_k[f'precision@{k}'] = precision * 100
    
    return precision_at_k

def calculate_false_positive_rate(scores, labels, threshold_percentile=90):
    """Calculate false positive rate"""
    if labels is None:
        return None
    
    threshold = np.percentile(scores, threshold_percentile)
    predictions = scores >= threshold
    
    # False positives: predicted as threat but actually normal
    false_positives = np.sum((predictions == 1) & (labels == 0))
    true_negatives = np.sum((predictions == 0) & (labels == 0))
    
    fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
    
    return fpr * 100

def generate_risk_report(user_ids, scores, top_k=20):
    """Generate report of top risky users"""
    risk_df = pd.DataFrame({
        'user_id': user_ids,
        'risk_score': scores
    })
    
    risk_df = risk_df.sort_values('risk_score', ascending=False)
    
    return risk_df.head(top_k)

def save_results(metrics, risk_report, model_name, output_path):
    """Save evaluation results"""
    os.makedirs(output_path, exist_ok=True)
    
    # Save metrics
    metrics_file = os.path.join(output_path, f'{model_name}_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save risk report
    report_file = os.path.join(output_path, f'{model_name}_risk_report.csv')
    risk_report.to_csv(report_file, index=False)
    
    print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate anomaly detection models')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, required=True, help='Features directory')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    # Load model
    model, model_name = load_model(args.model)
    print(f"Loaded model: {model_name}")
    
    # Load data
    X, user_ids, labels = load_features_and_labels(args.data)
    print(f"Evaluating {len(user_ids)} users")
    
    # Predict anomaly scores
    scores = predict_anomaly_scores(model, X)
    
    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'total_users': len(user_ids),
        'mean_risk_score': float(np.mean(scores)),
        'std_risk_score': float(np.std(scores))
    }
    
    if labels is not None:
        # Precision@K
        precision_metrics = calculate_precision_at_k(scores, labels)
        metrics.update(precision_metrics)
        
        # False positive rate
        fpr = calculate_false_positive_rate(scores, labels)
        metrics['false_positive_rate'] = fpr
        
        print(f"\nEvaluation Results:")
        print(f"Precision@10: {precision_metrics.get('precision@10', 'N/A'):.1f}%")
        print(f"False Positive Rate: {fpr:.1f}%")
    
    # Generate risk report
    risk_report = generate_risk_report(user_ids, scores)
    print(f"\nTop 10 Risky Users:")
    print(risk_report.head(10))
    
    # Save results
    save_results(metrics, risk_report, model_name, args.output)
    
    print("\n✓ Evaluation complete!")

if __name__ == "__main__":
    main()
