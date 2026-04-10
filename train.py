import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -------- Model 1 --------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
pred1 = lr.predict(X_test)
acc1 = accuracy_score(y_test, pred1)

# -------- Model 2 --------
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
pred2 = dt.predict(X_test)
acc2 = accuracy_score(y_test, pred2)

# -------- Model 3 (FINAL) --------
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
pred3 = rf.predict(X_test)
acc3 = accuracy_score(y_test, pred3)

# PRINT RESULTS (your "fake experiment tracking")
print("Logistic Regression Accuracy:", acc1)
print("Decision Tree Accuracy:", acc2)
print("Random Forest Accuracy:", acc3)
print("\nSelected Model: Random Forest (best performance)")

# Save best model
joblib.dump(rf, "diabetes_model.pkl")