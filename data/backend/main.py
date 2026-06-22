from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
import io
import os
from src.data_generator import BankingDataGenerator
from src.preprocessing import FraudPreprocessor
from src.models import FraudDetectionModels
from src.intelligence_engine import AntiGravityEngine
import uvicorn

app = FastAPI(title="Anti-Gravity Banking Fraud Detection API")

# Global state (In-memory for the demo)
DATA_PATH = "data/transactions.csv"
preprocessor = FraudPreprocessor()
engine = AntiGravityEngine()
generator = BankingDataGenerator(num_customers=500, num_transactions=1) # Generator for simulation
models = None
df_full = None
X_scaled = None

def run_pipeline(df):
    global models, X_scaled, df_full
    
    # 1. Preprocess
    df_processed, X_scaled = preprocessor.fit_transform(df)
    
    # 2. Train/Predict Models
    if models is None:
        models = FraudDetectionModels(input_dim=X_scaled.shape[1])
        models.fit_all(X_scaled)
    
    model_scores = models.predict_all(X_scaled)
    model_scores_df = pd.DataFrame(model_scores)
    
    # 3. Intelligence Engine
    df_final = engine.generate_intelligence_report(df_processed, model_scores_df)
    df_full = df_final
    return df_final

@app.on_event("startup")
async def startup_event():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        run_pipeline(df)
    else:
        print("Initial data not found. Please run generator first.")

@app.get("/transactions")
async def get_transactions(limit: int = 100):
    if df_full is not None:
        # Return top risk transactions and convert to list of dicts with standard types
        subset = df_full.sort_values("final_risk_score", ascending=False).head(limit)
        return subset.replace({np.nan: None}).to_dict(orient="records")
    return {"error": "No data available"}

@app.get("/stats")
async def get_stats():
    if df_full is not None:
        stats = preprocessor.get_statistics(df_full)
        # Convert risk distribution counts to standard int
        risk_dist = df_full['risk_level'].value_counts().to_dict()
        stats['risk_distribution'] = {str(k): int(v) for k, v in risk_dist.items()}
        # Ensure all float values in stats are standard floats
        for k, v in stats.items():
            if isinstance(v, (np.float32, np.float64)):
                stats[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                stats[k] = int(v)
        return stats
    return {"error": "No data available"}

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    global df_full
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    df_full = run_pipeline(df)
    return {"message": "Data uploaded and processed", "rows": len(df_full)}

@app.post("/simulate")
async def simulate_transaction():
    global df_full
    # Generate 1 new transaction
    new_txn = generator.generate_data()
    
    # Append to full dataset and re-process (simulating real-time update)
    if df_full is not None:
        # We simplify by just calculating the score for this one transaction
        # and prepending it to the list for the dashboard
        combined_df = pd.concat([new_txn, pd.read_csv(DATA_PATH)]).head(5000)
        df_full = run_pipeline(combined_df)
        return df_full.iloc[0].to_dict()
    return {"error": "System not initialized"}

@app.post("/retrain")
async def retrain():
    global models
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        models = None # Force retraining
        run_pipeline(df)
        return {"message": "Models retrained successfully"}
    return {"error": "Source data not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
