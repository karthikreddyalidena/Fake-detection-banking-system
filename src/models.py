import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
import torch
import torch.nn as nn
import torch.optim as optim


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class FraudDetectionModels:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.iso_forest = IsolationForest(contamination=0.05, random_state=42)
        self.lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        self.ocsvm = OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.autoencoder = Autoencoder(input_dim)
        
    def train_autoencoder(self, X_scaled, epochs=50, batch_size=32):
        X_tensor = torch.FloatTensor(X_scaled)
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        self.autoencoder.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.autoencoder(X_tensor)
            loss = criterion(outputs, X_tensor)
            loss.backward()
            optimizer.step()
        
    def get_autoencoder_scores(self, X_scaled):
        self.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)
            outputs = self.autoencoder(X_tensor)
            mse = torch.mean((outputs - X_tensor)**2, dim=1).numpy()
        return mse

    def fit_all(self, X_scaled):
        print("Fitting Isolation Forest...")
        self.iso_forest.fit(X_scaled)
        
        print("Fitting One-Class SVM...")
        self.ocsvm.fit(X_scaled)
        
        print("Fitting KMeans...")
        self.kmeans.fit(X_scaled)
        
        print("Training Autoencoder...")
        self.train_autoencoder(X_scaled)
        
        # LOF and DBSCAN don't have a separate fit/predict for all data in some versions, 
        # but for this script we'll just store them.

    def predict_all(self, X_scaled):
        results = {}
        
        # Isolation Forest: -1 for outliers, 1 for inliers. Score: higher means more anomalous? No, lower is more anomalous.
        # We transform to 0-1 where 1 is more anomalous.
        results['iso_forest'] = 1 - (self.iso_forest.decision_function(X_scaled) + 0.5)
        
        # LOF: higher score means more anomalous
        results['lof'] = -self.lof.fit_predict(X_scaled) # simplified for visualization
        
        # OCSVM
        results['ocsvm'] = 1 - (self.ocsvm.decision_function(X_scaled) + 0.5)
        
        # KMeans: distance to nearest cluster center
        distances = self.kmeans.transform(X_scaled)
        results['kmeans'] = np.min(distances, axis=1)
        
        # Autoencoder
        results['autoencoder'] = self.get_autoencoder_scores(X_scaled)
        
        # Normalize scores to 0-1
        for key in results:
            min_val = np.min(results[key])
            max_val = np.max(results[key])
            if max_val > min_val:
                results[key] = (results[key] - min_val) / (max_val - min_val)
            else:
                results[key] = 0.0
                
        return results
