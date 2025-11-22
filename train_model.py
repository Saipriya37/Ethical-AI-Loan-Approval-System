import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# ---------- 1. LOAD DATA ----------
data_path = os.path.join("data", "train.csv")
df = pd.read_csv(data_path)

# Drop ID column (not useful)
df = df.drop(columns=["Loan_ID"])

# ---------- 2. TARGET & FEATURES ----------
# Target column is 'Loan_Status' (values 'Y' / 'N')
y = df["Loan_Status"].map({"Y": 1, "N": 0})  # convert to 1/0
X = df.drop(columns=["Loan_Status"])

# Column types
numeric_features = ["ApplicantIncome", "CoapplicantIncome",
                    "LoanAmount", "Loan_Amount_Term", "Credit_History"]
categorical_features = ["Gender", "Married", "Dependents",
                        "Education", "Self_Employed", "Property_Area"]

# ---------- 3. PREPROCESSING ----------
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ---------- 4. MODEL PIPELINE ----------
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

clf = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", model),
    ]
)

# ---------- 5. TRAIN / TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 6. TRAIN MODEL ----------
clf.fit(X_train, y_train)

# ---------- 7. EVALUATE ----------
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {acc:.3f}")

# ---------- 8. SAVE MODEL PIPELINE ----------
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "loan_model.pkl")

with open(model_path, "wb") as f:
    pickle.dump(clf, f)

print(f"Model pipeline saved to: {model_path}")
