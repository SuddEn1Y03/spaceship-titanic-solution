#LightGBM Complete Optimized Solution
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

#1. Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')

print("Training set shape:", train.shape)
print("Test set shape:", test.shape)

#2. Feature engineering function
def feature_engineering(df, is_train=True, le_dict=None):
    data = df.copy()
    
    # Group information
    data[['GroupId', 'PassengerNum']] = data['PassengerId'].str.split('_', expand=True)
    data['PassengerNum'] = data['PassengerNum'].astype(int)
    data['GroupSize'] = data.groupby('GroupId')['PassengerNum'].transform('count')
    
    # Split Cabin
    cabin_split = data['Cabin'].str.split('/', expand=True)
    data['Deck'] = cabin_split[0]
    data['CabinNum'] = pd.to_numeric(cabin_split[1], errors='coerce')
    data['Side'] = cabin_split[2]
    
    # Total spending
    amenities = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    data['TotalSpend'] = data[amenities].sum(axis=1)
    
    # CryoSleep and spending conflict
    data['CryoSleep'] = data['CryoSleep'].fillna(False).astype(bool)
    data['CryoSpendConflict'] = ((data['CryoSleep']) & (data['TotalSpend'] > 0)).astype(int)
    
    # Age grouping
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 18, 30, 50, 80, 120],
                              labels=['Child', 'Young', 'Adult', 'Senior', 'Elder'])
    
    # Facility usage flags and log transforms
    for col in amenities:
        data[f'Used_{col}'] = (data[col] > 0).astype(int)
        data[f'Log_{col}'] = np.log1p(data[col])
    
    # Group-level statistics
    data['VIP'] = data['VIP'].fillna(False).astype(bool)
    data['GroupMeanAge'] = data.groupby('GroupId')['Age'].transform('mean')
    data['GroupTotalSpend'] = data.groupby('GroupId')['TotalSpend'].transform('sum')
    data['GroupVIPRatio'] = data.groupby('GroupId')['VIP'].transform(lambda x: x.astype(int).mean())
    
    # Drop unnecessary columns
    drop_cols = ['PassengerId', 'Name', 'Cabin', 'GroupId', 'PassengerNum']
    data = data.drop(columns=[c for c in drop_cols if c in data.columns])
    
    # Fill missing numerical values
    num_cols = ['Age', 'CabinNum', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                'TotalSpend', 'GroupMeanAge', 'GroupTotalSpend']
    for col in num_cols:
        if col in data.columns:
            median_val = data[col].median()
            data[col].fillna(median_val, inplace=True)
    
    # Fill missing categorical values
    cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'AgeGroup']
    for col in cat_cols:
        if col in data.columns:
            mode_val = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
            data[col].fillna(mode_val, inplace=True)
    
    # Convert boolean to int
    data['CryoSleep'] = data['CryoSleep'].astype(int)
    data['VIP'] = data['VIP'].astype(int)
    
    # Handle category dtype
    cat_dtype_cols = data.select_dtypes(include=['category']).columns.tolist()
    for col in cat_dtype_cols:
        data[col] = data[col].astype(str)
    
    # Encode object columns
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
    
    # Final safety: ensure all columns are numeric 
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        if data[col].dtype.name == 'category':
            data[col] = data[col].cat.codes
    
    # No need to convert to float32 for LightGBM, keep original numeric types
    if is_train:
        return data, le_dict
    else:
        return data

#3. Preprocessing 
y = train['Transported'].astype(int)
train_feat = train.drop('Transported', axis=1)
X_train, le_dict = feature_engineering(train_feat, is_train=True)
X_test = feature_engineering(test, is_train=False, le_dict=le_dict)

print("Training feature dimensions:", X_train.shape)
print("Test feature dimensions:", X_test.shape)

#4. LightGBM model 
# Using tuned hyperparameters
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# 5-fold cross-validation evaluation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(lgb_model, X_train, y, cv=cv, scoring='accuracy')
print(f"LightGBM 5-fold CV accuracy: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

# Train on full data
lgb_model.fit(X_train, y)

#5.Predict and generate submission file
pred = lgb_model.predict(X_test).astype(bool)
sample_sub['Transported'] = pred
sample_sub.to_csv('lgb_submission.csv', index=False)
print("Submission file generated: lgb_submission.csv")