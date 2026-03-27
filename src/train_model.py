import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/dataset.csv")
df["Revenue"] = df["Revenue"].astype(int)
df_model = pd.get_dummies(
    df,
    columns=["Month", "VisitorType"],
    drop_first=True)

X = df_model.drop("Revenue", axis=1)
y = df_model["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
artifacts = {
    "model": model,
    "columns": X.columns.tolist(),
    "threshold": 0.3}

with open("model/ecommerce_model.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("Model saved to model/ecommerce_model.pkl")