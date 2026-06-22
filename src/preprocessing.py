import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy import stats

class FraudPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.pca = PCA(n_components=3)
        self.numerical_cols = ['amount', 'hour', 'day_of_week', 'dist_from_home', 'velocity_1h']
        self.categorical_cols = ['merchant_type', 'payment_mode', 'device_id', 'location_city']

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        # Simple Euclidean distance for simulation purposes
        return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

    def extract_features(self, df):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Distance from "Home" (using first seen location as home for each customer in this simple version)
        home_locations = df.groupby('customer_id')[['lat', 'lon']].first().reset_index()
        home_locations.columns = ['customer_id', 'home_lat', 'home_lon']
        df = df.merge(home_locations, on='customer_id')
        df['dist_from_home'] = self.calculate_distance(df['lat'], df['lon'], df['home_lat'], df['home_lon'])
        
        # Velocity: Cumulative transaction count for each customer as a proxy
        df = df.sort_values('timestamp')
        df['velocity_1h'] = df.groupby('customer_id').cumcount() + 1
        
        return df

    def fit_transform(self, df):
        df = self.extract_features(df)
        
        # Handle Categorical
        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col])
            self.label_encoders[col] = le
            
        # Prepare feature matrix
        feature_cols = self.numerical_cols + [col + '_enc' for col in self.categorical_cols]
        X = df[feature_cols].fillna(0)
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # PCA
        X_pca = self.pca.fit_transform(X_scaled)
        df['pca_1'] = X_pca[:, 0]
        df['pca_2'] = X_pca[:, 1]
        df['pca_3'] = X_pca[:, 2]
        
        return df, X_scaled

    def get_statistics(self, df):
        stats_dict = {
            "mean_amount": df['amount'].mean(),
            "median_amount": df['amount'].median(),
            "std_amount": df['amount'].std(),
            "max_amount": df['amount'].max(),
            "fraud_count": df['is_fraud'].sum() if 'is_fraud' in df.columns else 0,
            "total_transactions": len(df)
        }
        # Correlation matrix
        corr = df[self.numerical_cols].corr().to_dict()
        stats_dict['correlation'] = corr
        return stats_dict

if __name__ == "__main__":
    df = pd.read_csv("c:/Users/navya/OneDrive/ドキュメント/FDBS/data/transactions.csv")
    preprocessor = FraudPreprocessor()
    df_processed, X_scaled = preprocessor.fit_transform(df)
    print(f"Processed shape: {df_processed.shape}")
    print(f"Scaled features shape: {X_scaled.shape}")
    print("Top features stats:", preprocessor.get_statistics(df_processed))
