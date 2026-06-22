import numpy as np
import pandas as pd

class AntiGravityEngine:
    """
    The Anti-Gravity Intelligence Engine:
    An advanced anomaly detection layer that aggregates multiple ML model outputs
    and incorporates behavioral heuristics to generate a dynamic fraud risk score.
    """
    def __init__(self, model_weights=None):
        if model_weights is None:
            self.model_weights = {
                'iso_forest': 0.3,
                'autoencoder': 0.3,
                'ocsvm': 0.15,
                'lof': 0.15,
                'kmeans': 0.1
            }
        else:
            self.model_weights = model_weights

    def calculate_risk_score(self, model_scores_df):
        """
        Aggregates individual model scores into a final risk score (0-100).
        """
        final_score = np.zeros(len(model_scores_df))
        for model, weight in self.model_weights.items():
            if model in model_scores_df.columns:
                final_score += model_scores_df[model] * weight
        
        return final_score * 100

    def detect_behavioral_drift(self, df):
        """
        Calculates drift based on deviation from customer's mean transaction amount.
        """
        # Calculate customer-level stats
        cust_stats = df.groupby('customer_id')['amount'].agg(['mean', 'std']).reset_index()
        cust_stats.columns = ['customer_id', 'cust_avg_amount', 'cust_std_amount']
        
        df = df.merge(cust_stats, on='customer_id')
        
        # Z-score of amount relative to customer's own history
        df['amount_drift'] = (df['amount'] - df['cust_avg_amount']) / (df['cust_std_amount'] + 1e-6)
        df['amount_drift'] = df['amount_drift'].abs().clip(0, 10) / 10 # Normalize to 0-1
        
        return df['amount_drift']

    def generate_intelligence_report(self, df, model_scores_df):
        """
        Generates a final dataframe with all scores and intelligence markers.
        """
        results_df = df.copy()
        
        # Add model scores
        for col in model_scores_df.columns:
            results_df[f'score_{col}'] = model_scores_df[col]
            
        # Add behavioral drift
        results_df['behavioral_drift'] = self.detect_behavioral_drift(df)
        
        # Calculate final risk score (ensemble + drift)
        base_risk = self.calculate_risk_score(model_scores_df)
        drift_bonus = results_df['behavioral_drift'] * 20 # Add up to 20 points for drift
        
        results_df['final_risk_score'] = (base_risk * 0.8 + drift_bonus).clip(0, 100)
        
        # Tagging
        results_df['risk_level'] = pd.cut(
            results_df['final_risk_score'],
            bins=[0, 30, 60, 85, 100],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        return results_df
