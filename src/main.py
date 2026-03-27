import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/dataset.csv")

print("First rows:")
print(df.head())

print("\nData info:")
print(df.info())

print("\nDescriptive statistics:")
print(df.describe())

print("\nRevenue counts:")
print(df["Revenue"].value_counts())
print("\nRevenue distribution:")
print(df["Revenue"].value_counts(normalize=True))

print("\nAverage numeric values by Revenue:")
print(df.groupby("Revenue").mean(numeric_only=True))

print("\nVisitorType vs Revenue:")
print(pd.crosstab(df["VisitorType"], df["Revenue"]))

print("\nMonth vs Revenue:")
print(pd.crosstab(df["Month"], df["Revenue"]))

print("\nPageValues by Revenue:")
print(df.groupby("Revenue")["PageValues"].mean())

df_model = df.copy()
df_model["Revenue"] = df_model["Revenue"].astype(int)
df_model = pd.get_dummies(
    df_model,
    columns=["Month", "VisitorType"],
    drop_first=True)

X = df_model.drop("Revenue", axis=1)
y = df_model["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("\nLogistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

print("\nGradient Boosting")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))

y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
y_pred_custom = (y_prob_rf > 0.3).astype(int)

print("\nRandom Forest with threshold 0.3")
print(classification_report(y_test, y_pred_custom))

conn = sqlite3.connect("ecommerce.db")
df.to_sql("sessions", conn, if_exists="replace", index=False)

query_1 = """
SELECT
    VisitorType,
    COUNT(*) AS sessions_count,
    SUM(CASE WHEN Revenue = 1 THEN 1 ELSE 0 END) AS purchases,
    ROUND(1.0 * SUM(CASE WHEN Revenue = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS conversion_rate
FROM sessions
GROUP BY VisitorType
"""
print("\nConversion by VisitorType:")
print(pd.read_sql(query_1, conn))

query_2 = """
SELECT
    Month,
    COUNT(*) AS sessions_count,
    SUM(CASE WHEN Revenue = 1 THEN 1 ELSE 0 END) AS purchases,
    ROUND(1.0 * SUM(CASE WHEN Revenue = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS conversion_rate
FROM sessions
GROUP BY Month
ORDER BY conversion_rate DESC
"""
print("\nConversion by Month:")
print(pd.read_sql(query_2, conn))

query_3 = """
SELECT
    Revenue,
    ROUND(AVG(PageValues), 4) AS avg_page_values
FROM sessions
GROUP BY Revenue
"""
print("\nAverage PageValues by Revenue:")
print(pd.read_sql(query_3, conn))

query_4 = """
SELECT
    Revenue,
    ROUND(AVG(ProductRelated), 2) AS avg_product_related
FROM sessions
GROUP BY Revenue
"""
print("\nAverage ProductRelated by Revenue:")
print(pd.read_sql(query_4, conn))

query_5 = """
SELECT
    Weekend,
    COUNT(*) AS sessions_count,
    SUM(CASE WHEN Revenue = 1 THEN 1 ELSE 0 END) AS purchases,
    ROUND(1.0 * SUM(CASE WHEN Revenue = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS conversion_rate
FROM sessions
GROUP BY Weekend
"""
print("\nConversion by Weekend:")
print(pd.read_sql(query_5, conn))

conn.close()
df["Revenue"].value_counts().plot(kind="bar")
plt.title("Revenue Distribution")
plt.xlabel("Revenue (0 = no purchase, 1 = purchase)")
plt.ylabel("Number of sessions")
plt.tight_layout()
plt.show()
month_revenue = pd.crosstab(df["Month"], df["Revenue"])
month_revenue[True].sort_values(ascending=False).plot(kind="bar")
plt.title("Number of Purchases by Month")
plt.xlabel("Month")
plt.ylabel("Purchases")
plt.tight_layout()
plt.show()
df.groupby("Revenue")["PageValues"].mean().plot(kind="bar")
plt.title("Average PageValues by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Average PageValues")
plt.tight_layout()
plt.show()