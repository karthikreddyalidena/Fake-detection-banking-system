import sys
import os
print("Python executable:", sys.executable)
print("Current directory:", os.getcwd())
try:
    import fastapi
    print("FastAPI imported successfully:", fastapi.__version__)
except ImportError as e:
    print("FastAPI import failed:", e)

try:
    from src.data_generator import BankingDataGenerator
    print("src.data_generator imported successfully")
except ImportError as e:
    print("src.data_generator import failed:", e)
