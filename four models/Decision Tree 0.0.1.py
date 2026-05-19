import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. Load Dataset
# ==========================================
# Ensure train.csv and test.csv are in the same directory
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    print("Datasets loaded successfully.")
except FileNotFoundError:
    print("Error: train.csv or test.csv not found.")

# ==========================================
# 2. Feature Engineering & Preprocessing
# ==========================================
def preprocess_spaceship_titanic(df):
    """
    Custom preprocessing function for Decision Tree model.
    Includes feature splitting, missing value imputation, and encoding.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Split 'Cabin' into Deck, Num, and Side
    # Format is Deck/Num/Side (e.g., B/0/P)
    df[['Deck', 'Cabin_Num', 'Side']] = df['Cabin'].str.split('/', expand=True)

    # Group spending features into a single total expenditure
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpending'] = df[spending_cols].sum(axis=1)

    # Impute Missing Values
    # Categorical: Fill with mode or 'Unknown'
    df['HomePlanet'] = df['HomePlanet'].fillna(df['HomePlanet'].mode()[0])
    df['CryoSleep'] = df['CryoSleep'].fillna(False)
    df['Destination'] = df['Destination'].fillna(df['Destination'].mode()[0])
    df['VIP'] = df['VIP'].fillna(False)
    df['Deck'] = df['Deck'].fillna('Unknown')
    df['Side'] = df['Side'].fillna('Unknown')

    # Numerical: Fill with median (more robust to outliers than mean)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    for col in spending_cols:
        df[col] = df[col].fillna(0) # Assume no spending if data is missing

    # Encoding Categorical Variables
    le = LabelEncoder()
    categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    for col in categorical_features:
        df[col] = le.fit_transform(df[col].astype(str))

    # Drop columns that are not useful for tree splitting
    # 'Name' and 'PassengerId' are unique identifiers
    # 'Cabin_Num' is often too high-cardinality for a simple tree
    cols_to_drop = ['PassengerId', 'Cabin', 'Name', 'Cabin_Num']
    return df.drop(columns=cols_to_drop)

# Process both datasets
X = preprocess_spaceship_titanic(train_df.drop('Transported', axis=1))
y = train_df['Transported'].astype(int)
X_test_final = preprocess_spaceship_titanic(test_df)

# ==========================================
# 3. Model Selection & Hyperparameter Tuning
# ==========================================
# Split training data for internal validation (80% Train, 20% Val)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for Grid Search (Course Requirement 4.4.2)
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 8, 12, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'ccp_alpha': [0.0, 0.001, 0.01] # Complexity pruning to prevent overfitting
}

print("\nStarting Hyperparameter Tuning...")
start_time = time.time() # Start timer for Computational Cost analysis

dt_model = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

end_time = time.time()
training_duration = end_time - start_time

# Select the best model
best_dt = grid_search.best_estimator_

# ==========================================
# 4. Evaluation & Analysis
# ==========================================
y_pred = best_dt.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred)

print(f"\n--- Decision Tree Performance ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Training Time: {training_duration:.4f} seconds")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Feature Importance Visualization (Course Requirement 4.5.3)
plt.figure(figsize=(10, 6))
importances = pd.Series(best_dt.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh', color='skyblue')
plt.title('Top 10 Feature Importances (Decision Tree)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# ==========================================
# 5. Generate Kaggle Submission
# ==========================================
test_predictions = best_dt.predict(X_test_final)
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Transported': test_predictions.astype(bool)
})
submission.to_csv('submission_decision_tree.csv', index=False)
print("\nSubmission file saved as 'submission_decision_tree.csv'.")