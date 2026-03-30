"""
Package Verification Script - Fixed Version
Run this to verify all packages are working
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import xgboost
import lightgbm
import plotly
import streamlit

print('='*50)
print('PACKAGE VERIFICATION')
print('='*50)
print(f'Python: {sys.version}')
print(f'Python Path: {sys.executable}')
print()
print('📚 Package Versions:')
print(f'   Pandas: {pd.__version__}')
print(f'   NumPy: {np.__version__}')
print(f'   Matplotlib: {matplotlib.__version__}')
print(f'   Seaborn: {sns.__version__}')
print(f'   Scikit-learn: {sklearn.__version__}')
print(f'   XGBoost: {xgboost.__version__}')
print(f'   LightGBM: {lightgbm.__version__}')
print(f'   Plotly: {plotly.__version__}')
print(f'   Streamlit: {streamlit.__version__}')
print('='*50)

# Test data processing
print('\n📊 Testing Data Processing...')
test_data = pd.DataFrame({
    'Property_ID': [1, 2, 3],
    'Price': [200000, 250000, 180000],
    'Area': [1500, 1800, 1200]
})
print(f'   ✓ DataFrame created: {test_data.shape}')
print(f'   ✓ Sample data:\n{test_data}')

# Test matplotlib
print('\n📈 Testing Visualization...')
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(test_data['Property_ID'], test_data['Price'], color='steelblue')
ax.set_title('Test Visualization - Property Prices', fontsize=14)
ax.set_xlabel('Property ID')
ax.set_ylabel('Price ($)')
plt.close()
print('   ✓ Matplotlib plot created successfully')

# Test sklearn
print('\n🤖 Testing ML Models...')
X = np.random.randn(100, 5)
y = np.random.randn(100)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X, y)
print(f'   ✓ Random Forest trained successfully')
print(f'   ✓ Model score: {model.score(X, y):.4f}')

print('\n' + '='*50)
print('✅ ALL TESTS PASSED! Your environment is ready for the project.')
print('='*50)