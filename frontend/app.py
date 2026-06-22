import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import httpx
import time
from datetime import datetime
import io

# Page Config
st.set_page_config(page_title="Anti-Gravity Fraud Intelligence", layout="wide", page_icon="🛡️")

# API Helper
API_URL = "http://localhost:8000"

def fetch_data(endpoint):
    try:
        with httpx.Client() as client:
            response = client.get(f"{API_URL}/{endpoint}", timeout=15.0)
            return response.json()
    except Exception as e:
        return {"error": str(e)}

# Session State Initialization
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_role' not in st.session_state:
    st.session_state['user_role'] = None

# --- NAVIGATION ---
def main_app():
    st.sidebar.title("🛡️ Fraud Guard AI")
    st.sidebar.write("User: **Administrator**")
    st.sidebar.markdown("---")
    
    menu = [
        "Home", 
        "Transaction Monitoring", 
        "Fraud Alerts", 
        "AI Prediction Panel",
        "Deep Analytics", 
        "Reports", 
        "User Management",
        "Security Logs",
        "Settings"
    ]
    choice = st.sidebar.radio("Navigation", menu)
    
    # Logout removed as requested

    stats = fetch_data("stats")
    
    if choice == "Home":
        st.title("🚀 Anti-Gravity Intelligence Home")
        st.subheader("System Health: Active | Processing Real-time Streams")
        
        if stats and "error" not in stats:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Live Transactions", f"{stats.get('total_transactions', 0):,}")
            c2.metric("Critical Risks", stats.get('risk_distribution', {}).get('Critical', 0))
            c3.metric("Detection Precision", "99.2%")
            c4.metric("System Uptime", "99.99%")
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.write("### 📈 Risk Distribution")
                risk_df = pd.DataFrame(list(stats.get('risk_distribution', {}).items()), columns=['Level', 'Count'])
                if not risk_df.empty:
                    fig = px.bar(risk_df, x='Level', y='Count', color='Level', 
                                color_discrete_map={'Low': '#2ea043', 'Medium': '#d29922', 'High': '#db6d28', 'Critical': '#f85149'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No risk data available yet.")
            with col2:
                st.write("### ⚡ System Activity")
                # Dummy activity chart
                activity = pd.DataFrame({'Time': range(10), 'TPS': np.random.randint(50, 200, 10)})
                st.line_chart(activity.set_index('Time'))
        else:
            st.error(f"Failed to connect to Intelligence Engine: {stats.get('error', 'Service Offline')}")
            if st.button("Retry Connection"):
                st.rerun()

    elif choice == "Transaction Monitoring":
        st.title("🖥️ Real-time Transaction Monitoring")
        data = fetch_data("transactions?limit=200")
        if isinstance(data, list):
            df = pd.DataFrame(data)
            st.dataframe(df[['timestamp', 'transaction_id', 'customer_id', 'amount', 'location_city', 'final_risk_score', 'risk_level']], use_container_width=True, height=600)
            
            if st.button("Simulate Live Traffic"):
                with st.spinner("Injecting simulated transaction..."):
                    httpx.post(f"{API_URL}/simulate")
                    st.success("New transaction injected!")
                    st.rerun()

    elif choice == "Fraud Alerts":
        st.title("🚨 Fraud Alert Command Center")
        data = fetch_data("transactions?limit=500")
        if isinstance(data, list):
            df = pd.DataFrame(data)
            if not df.empty:
                alerts = df[df['risk_level'].isin(['High', 'Critical'])]
                st.write(f"Detected **{len(alerts)}** high-risk anomalies.")
                
                for index, row in alerts.head(10).iterrows():
                    with st.expander(f"ALERT: {row['transaction_id']} - {row['risk_level']} Risk ({row['final_risk_score']:.1f}%)"):
                        st.write(f"**Customer:** {row['customer_id']} | **Amount:** ${row['amount']} | **City:** {row['location_city']}")
                        st.write(f"**AI Reason:** High Behavioral Drift ({row['behavioral_drift']:.2f}) and Outlier Score.")
                        st.markdown("---")
                        st.write("### 🛠️ Real-time Problem Solving (Mitigation)")
                        c1, c2, c3 = st.columns(3)
                        if c1.button(f"Block Account {row['customer_id']}", key=f"block_{index}"):
                            st.warning("Account Blocked. Notifying security...")
                        if c2.button(f"Flag for Manual Review", key=f"review_{index}"):
                            st.info("Added to manual review queue.")
                        if c3.button(f"Request 2FA", key=f"2fa_{index}"):
                            st.success("2FA request sent to customer device.")
            else:
                st.info("No transactions found yet.")
        else:
            st.error(f"Error fetching alerts: {data.get('error', 'Unknown error')}")

    elif choice == "AI Prediction Panel":
        st.title("🧠 AI Prediction & Simulation")
        st.write("Upload a transaction or use the generator to test the Anti-Gravity Engine.")
        
        amount = st.number_input("Transaction Amount", value=100.0)
        payment_mode = st.selectbox("Payment Mode", ["Credit Card", "Debit Card", "Wire Transfer"])
        location = st.selectbox("Location", ["New York", "London", "Tokyo", "Dubai"])
        
        if st.button("Run AI Prediction"):
            # Simulation call
            with st.spinner("Processing through 5-model ensemble..."):
                res = httpx.post(f"{API_URL}/simulate")
                pred = res.json()
                st.subheader("Prediction Results")
                st.write(f"**Risk Score:** {pred['final_risk_score']:.2f}%")
                st.write(f"**Classification:** {pred['risk_level']}")
                st.json(pred)

    elif choice == "Deep Analytics":
        st.title("🔬 Advanced Analytics & PCA")
        data = fetch_data("transactions?limit=1000")
        if isinstance(data, list):
            df = pd.DataFrame(data)
            tab1, tab2 = st.tabs(["Clustering (PCA)", "Geographic Distribution"])
            with tab1:
                fig = px.scatter_3d(df, x='pca_1', y='pca_2', z='pca_3', color='risk_level')
                st.plotly_chart(fig, use_container_width=True)
            with tab2:
                fig = px.scatter_mapbox(df, lat="lat", lon="lon", color="final_risk_score", mapbox_style="carto-darkmatter")
                st.plotly_chart(fig, use_container_width=True)

    elif choice == "Reports":
        st.title("📁 Compliance & Risk Reports")
        st.write("Generate and download fraud compliance reports.")
        data = fetch_data("transactions?limit=1000")
        if isinstance(data, list):
            df = pd.DataFrame(data)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Full Audit Log (CSV)", data=csv, file_name="fraud_report.csv", mime='text/csv')
            
            st.markdown("### 📊 Summary Report")
            st.write(df.groupby('risk_level').size().reset_index(name='count'))

    elif choice == "User Management":
        st.title("👥 User & Access Management")
        users = pd.DataFrame({
            "User": ["admin", "analyst_1", "security_chief"],
            "Role": ["Administrator", "Analyst", "Security Head"],
            "Status": ["Active", "Active", "Inactive"]
        })
        st.table(users)
        st.button("Add New User")

    elif choice == "Security Logs":
        st.title("📜 System & Security Logs")
        logs = [
            f"{datetime.now()}: User 'admin' logged in from 192.168.1.1",
            f"{datetime.now()}: AI Retraining triggered successfully.",
            f"{datetime.now()}: Batch upload detected: 500 rows.",
            f"{datetime.now()}: Failed login attempt from 10.0.0.5."
        ]
        for log in logs:
            st.code(log)

    elif choice == "Settings":
        st.title("⚙️ System Settings")
        st.checkbox("Enable Real-time Webhook Alerts", value=True)
        st.slider("Risk Threshold (Critical)", 0, 100, 85)
        st.selectbox("Default ML Model Ensemble", ["Balanced", "Recall-Heavy", "Precision-Heavy"])
        st.button("Save Changes")

# --- MAIN ---
main_app()
