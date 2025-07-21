import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
import json
import numpy as np


# Load dataset
data = pd.read_csv("data/Salary_Data.csv")

# Define features and target
X = data[['Total Experience', 'Team Lead Experience', 'Project Manager Experience', 'Certifications']]
y = data['Salary']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'salary_model.pkl')

# Evaluate model
score = r2_score(y_test, model.predict(X_test))

# Save score
with open("model_score.json", "w") as f:
    json.dump({"r2_score": score}, f)

residuals = y_test - model.predict(X_test)
residual_std = np.std(residuals)
with open('model_stats.json', 'w') as f:
    json.dump({'residual_std': residual_std}, f)
