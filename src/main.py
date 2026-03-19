import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/customers.csv")
conn = sqlite3.connect("database.db")
# сохраняем в SQL
df.to_sql("customers", conn, if_exists="replace", index=False)

query = "SELECT * FROM customers"
df_sql = pd.read_sql(query, conn)
print("Purchase distribution:\n", df_sql["purchase"].value_counts())
print("Average income by purchase:\n", df_sql.groupby("purchase")["income"].mean())
X = df_sql[["age", "income", "visits"]]
y = df_sql["purchase"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

df_sql["purchase"].value_counts().plot(kind="bar")
plt.title("Purchase Distribution")
plt.xlabel("Purchase (0 = no, 1 = yes)")
plt.ylabel("Number of customers")
plt.tight_layout()
plt.show()

conn.close()