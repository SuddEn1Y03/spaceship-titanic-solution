#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')


# ----------------------
# ----------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission = pd.read_csv("sample_submission.csv")

# ----------------------
# ----------------------
def create_features(df):
    df = df.copy()
    
    # Group
    df["Group"] = df["PassengerId"].str.split("_").str[0]
    df["GroupSize"] = df.groupby("Group")["Group"].transform("count")
    df["Alone"] = (df["GroupSize"] == 1).astype(int)
    
    # Cabin split
    df[["Deck", "CabinNum", "Side"]] = df["Cabin"].str.split("/", expand=True)
    
    # Expenses
    expenses = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpent"] = df[expenses].fillna(0).sum(axis=1)
    df["NoSpending"] = (df["TotalSpent"] == 0).astype(int)
    
    # CryoSleep + NoSpending (strong feature)
    df["CryoNoExpense"] = (df["CryoSleep"] == True) & (df["NoSpending"] == 1)
    
    # Age bins
    df["AgeChild"] = (df["Age"] < 13).astype(int)
    df["AgeAdult"] = (df["Age"] >= 18).astype(int)
    
    # Fill simple missing values
    df["HomePlanet"] = df["HomePlanet"].fillna("Earth")
    df["Destination"] = df["Destination"].fillna("TRAPPIST-1e")
    df["CryoSleep"] = df["CryoSleep"].fillna(False)
    df["VIP"] = df["VIP"].fillna(False)
    
    # Drop unused
    drop_cols = ["PassengerId", "Cabin", "Name", "Group", "CabinNum"]
    df = df.drop(columns=drop_cols, errors="ignore")
    
    return df

# Process data
train_fe = create_features(train)
test_fe = create_features(test)

# X and y
X = train_fe.drop("Transported", axis=1)
y = train_fe["Transported"].astype(int)
test_final = test_fe

# ----------------------
# Sklearn Pipeline (Logistic Regression)
# ----------------------
cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "CryoNoExpense"]
num_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
            "TotalSpent", "NoSpending", "GroupSize", "Alone", "AgeChild", "AgeAdult"]

# Preprocessing
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])

# Model
model = Pipeline(steps=[
    ("pre", preprocessor),
    ("clf", LogisticRegression(C=0.5, max_iter=2000, random_state=42))
])

# ----------------------
# Cross Validation
# ----------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print(f"CV Accuracy: {scores.mean():.4f}")

# ----------------------
# Train & Predict
# ----------------------
model.fit(X, y)
preds = model.predict(test_final)
submission["Transported"] = preds.astype(bool)

# Save locally
submission.to_csv("submission.csv", index=False)
print("Submission.csv is ready.")





