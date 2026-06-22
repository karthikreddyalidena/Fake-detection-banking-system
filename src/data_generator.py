import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class BankingDataGenerator:
    def __init__(self, num_customers=1000, num_transactions=10000):
        self.num_customers = num_customers
        self.num_transactions = num_transactions
        self.customers = [f"CUST_{i:04d}" for i in range(num_customers)]
        self.merchants = ["Retail", "Food", "Entertainment", "Travel", "Health", "Services", "Online"]
        self.payment_modes = ["Credit Card", "Debit Card", "Wire Transfer", "Mobile Wallet"]
        self.cities = [
            {"name": "New York", "lat": 40.7128, "lon": -74.0060},
            {"name": "London", "lat": 51.5074, "lon": -0.1278},
            {"name": "Tokyo", "lat": 35.6895, "lon": 139.6917},
            {"name": "Paris", "lat": 48.8566, "lon": 2.3522},
            {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
            {"name": "Dubai", "lat": 25.2048, "lon": 55.2708},
            {"name": "Singapore", "lat": 1.3521, "lon": 103.8198}
        ]
        self.device_ids = [f"DEV_{i:04d}" for i in range(num_customers // 2)]

    def generate_data(self):
        data = []
        start_date = datetime.now() - timedelta(days=30)
        
        # Pre-assign "home" locations to customers
        customer_homes = {cust: random.choice(self.cities) for cust in self.customers}
        customer_devices = {cust: random.sample(self.device_ids, random.randint(1, 2)) for cust in self.customers}
        
        for i in range(self.num_transactions):
            cust = random.choice(self.customers)
            home_city = customer_homes[cust]
            
            # Normal behavior vs Anomalous behavior
            is_fraud = 0
            rand_val = random.random()
            
            if rand_val < 0.05:  # 5% chance of being an anomaly/fraud
                is_fraud = 1
                amount = random.uniform(5000, 50000) if random.random() > 0.5 else random.uniform(1, 10)
                city = random.choice([c for c in self.cities if c != home_city])
                timestamp = start_date + timedelta(seconds=random.randint(0, 30*24*60*60))
                # Fraudulent hours (e.g., 2 AM)
                if random.random() > 0.7:
                    timestamp = timestamp.replace(hour=random.randint(0, 4))
                device = random.choice(self.device_ids) # Random device
            else:
                # Normal transaction
                amount = np.random.lognormal(mean=4, sigma=1) # Average around $50-$100
                city = home_city if random.random() > 0.1 else random.choice(self.cities)
                timestamp = start_date + timedelta(seconds=random.randint(0, 30*24*60*60))
                device = random.choice(customer_devices[cust])

            data.append({
                "transaction_id": f"TXN_{i:06d}",
                "customer_id": cust,
                "amount": round(amount, 2),
                "timestamp": timestamp,
                "location_city": city["name"],
                "lat": city["lat"],
                "lon": city["lon"],
                "merchant_type": random.choice(self.merchants),
                "device_id": device,
                "payment_mode": random.choice(self.payment_modes),
                "is_fraud": is_fraud
            })

        df = pd.DataFrame(data)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

if __name__ == "__main__":
    generator = BankingDataGenerator(num_customers=500, num_transactions=5000)
    df = generator.generate_data()
    df.to_csv("c:/Users/navya/OneDrive/ドキュメント/FDBS/data/transactions.csv", index=False)
    print("Generated 5000 transactions at data/transactions.csv")
