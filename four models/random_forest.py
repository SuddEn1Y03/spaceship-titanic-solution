import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)

def feature_engineering(df, is_train=True, le_dict=None):
    data = df.copy()
    
    data[['GroupId', 'PassengerNum']] = data['PassengerId'].str.split('_', expand=True)
    data['PassengerNum'] = data['PassengerNum'].astype(int)
    data['GroupSize'] = data.groupby('GroupId')['PassengerNum'].transform('count')
    
    cabin_split = data['Cabin'].str.split('/', expand=True)
    data['Deck'] = cabin_split[0]
    data['CabinNum'] = pd.to_numeric(cabin_split[1], errors='coerce')
    data['Side'] = cabin_split[2]
    
    amenities = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    data['TotalSpend'] = data[amenities].sum(axis=1)
    
    data['CryoSleep'] = data['CryoSleep'].fillna(False).astype(bool)
    data['CryoSpendConflict'] = ((data['CryoSleep']) & (data['TotalSpend'] > 0)).astype(int)
    
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 18, 30, 50, 80, 120],
                              labels=['Child', 'Young', 'Adult', 'Senior', 'Elder'])
    
    for col in amenities:
        data[f'Used_{col}'] = (data[col] > 0).astype(int)
        data[f'Log_{col}'] = np.log1p(data[col])
    
    data['VIP'] = data['VIP'].fillna(False).astype(bool)
    data['GroupMeanAge'] = data.groupby('GroupId')['Age'].transform('mean')
    data['GroupTotalSpend'] = data.groupby('GroupId')['TotalSpend'].transform('sum')
    data['GroupVIPRatio'] = data.groupby('GroupId')['VIP'].transform(lambda x: x.astype(int).mean())
    
    drop_cols = ['PassengerId', 'Name', 'Cabin', 'GroupId', 'PassengerNum']
    data = data.drop(columns=[c for c in drop_cols if c in data.columns])
    
    num_cols = ['Age', 'CabinNum', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                'TotalSpend', 'GroupMeanAge', 'GroupTotalSpend']
    for col in num_cols:
        if col in data.columns:
            median_val = data[col].median()
            data[col].fillna(median_val, inplace=True)
    
    cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'AgeGroup']
    for col in cat_cols:
        if col in data.columns:
            mode_val = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
            data[col].fillna(mode_val, inplace=True)
    
    data['CryoSleep'] = data['CryoSleep'].astype(int)
    data['VIP'] = data['VIP'].astype(int)
    
    cat_dtype_cols = data.select_dtypes(include=['category']).columns.tolist()
    for col in cat_dtype_cols:
        data[col] = data[col].astype(str)
    
    object_cols = data.select_dtypes(include=['object']).columns.tolist()
    if is_train:
        le_dict = {}
        for col in object_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            le_dict[col] = le
    else:
        for col in object_cols:
            if col in le_dict:
                le = le_dict[col]
                data[col] = data[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
            else:
                data[col] = 0
    
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        if data[col].dtype.name == 'category':
            data[col] = data[col].cat.codes
    data = data.astype(np.float32)
    
    if is_train:
        return data, le_dict
    else:
        return data

y_train = train['Transported'].astype(int)
train_features = train.drop('Transported', axis=1)

X_train, le_dict = feature_engineering(train_features, is_train=True)
X_test = feature_engineering(test, is_train=False, le_dict=le_dict)

print("Training features shape:", X_train.shape)
print("Test features shape:", X_test.shape)
print("All train dtypes:", X_train.dtypes.unique())

rf = RandomForestClassifier(n_estimators=300, max_depth=15,
                            min_samples_split=10, min_samples_leaf=4,
                            max_features='sqrt', random_state=42, n_jobs=-1)

cv = StratifiedKFold(5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

rf.fit(X_train, y_train)
train_acc = accuracy_score(y_train, rf.predict(X_train))
print(f"Train Accuracy: {train_acc:.4f}")

pred = rf.predict(X_test).astype(bool)
sample_sub['Transported'] = pred
sample_sub.to_csv('random_forest_submission.csv', index=False)
print("Submission saved: random_forest_submission.csv")