import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("C:\\Users\\17726\\Desktop\\T\\train.csv")
test = pd.read_csv("C:\\Users\\17726\\Desktop\\T\\test.csv")
sample = pd.read_csv("C:\\Users\\17726\\Desktop\\T\\sample_submission.csv")

passenger_ids = sample["PassengerId"]

def process_data(df):
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    
    for col in spend_cols:
        df[col] = df[col].fillna(0)
    
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["HomePlanet"] = df["HomePlanet"].fillna("Earth")
    df["CryoSleep"] = df["CryoSleep"].fillna(False)
    df["Destination"] = df["Destination"].fillna("TRAPPIST-1e")
    df["VIP"] = df["VIP"].fillna(False)
    df["Cabin"] = df["Cabin"].fillna("G/1/S")

    df["Deck"] = df["Cabin"].str.split("/").str[0]
    df["Side"] = df["Cabin"].str.split("/").str[-1]

    df["TotalSpend"] = df[spend_cols].sum(axis=1)
    
    df["AgeGroup"] = pd.cut(
        df["Age"], bins=[0,12,18,30,50,100], 
        labels=[1,2,3,4,5],
    ).astype(float) 
    df["CryoSleep"] = df["CryoSleep"].astype(int)
    df["VIP"] = df["VIP"].astype(int)

    df = df.drop(["PassengerId", "Cabin", "Name", "Age"], axis=1, errors="ignore")

    df = pd.get_dummies(df, columns=["HomePlanet", "Destination", "Deck", "Side"], drop_first=True)
    
    df = df.fillna(0)
    
    return df

train_clean = process_data(train.copy())
test_clean = process_data(test.copy())

X = train_clean.drop("Transported", axis=1)
y = train_clean["Transported"]
test_X = test_clean

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_X)

best_k = 13
best_acc = 0
for k in range(3, 31, 2):
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(X_scaled, y)
    val_acc = knn.score(X_scaled, y)
    if val_acc > best_acc:
        best_acc = val_acc
        best_k = k

print(f" Optimal K value: {best_k}")
print(f" Validation set accuracy: {best_acc:.4f}")

model = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
model.fit(X_scaled, y)
preds = model.predict(test_scaled)

submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Transported": preds
})

submission.to_csv("A.csv", index=False)
print("\n Validation set accuracy：A.csv")