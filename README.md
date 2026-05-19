# Spaceship Titanic – LightGBM Solution

**Team Name:** Curly Baboon @MLW  
**Course:** AI3023 Machine Learning Workshop  

## 1. Project Description
This repository contains our solution for the Kaggle competition **Spaceship Titanic**.  
The goal is to predict whether a passenger was transported to an alternate dimension based on personal records (e.g., home planet, age, cryosleep status, onboard spending).  

Competition link: [Spaceship Titanic on Kaggle](https://www.kaggle.com/competitions/spaceship-titanic)

## 2. Requirements
- Python 3.8 or higher
- Install the required packages using pip:

```bash
pip install -r requirements.txt


```markdown
## 3. Data Preparation
1. Download `train.csv`, `test.csv`, and `sample_submission.csv` from the Kaggle competition page.
2. Place these three files in **the same directory** as the script `lightgbm_model.py`.
## 4. How to Run
Open a terminal (or command prompt) in the directory containing the script and data files, then execute:

```bash
python lightgbm_model.py

The script will:

Load and preprocess the data

Perform feature engineering (extract features from PassengerId, Cabin, Name, etc.)

Train a LightGBM classifier with 5‑fold cross‑validation

Output a submission file lgb_submission.csv ready for Kaggle upload


```markdown
## 5. Code Structure
- `lightgbm_model.py` – Main script containing:
  - Data loading
  - Feature engineering function (`feature_engineering`)
  - Missing value imputation (median for numerical, mode for categorical)
  - Label encoding for categorical variables
  - LightGBM model training (with hyperparameters tuned)
  - 5‑fold stratified cross‑validation
  - Prediction and submission file generation
- `requirements.txt` – List of required Python packages
- `README.md` – This file
## 6. Key Features Engineered
| Feature | Description |
|---------|-------------|
| `GroupSize` | Number of passengers in the same travelling group (from `PassengerId`) |
| `Deck`, `CabinNum`, `Side` | Extracted from `Cabin` (e.g., "B/0/P" → Deck=B, CabinNum=0, Side=P) |
| `CabinParity` | Cabin number parity (odd/even) |
| `TotalSpend` | Sum of all five luxury amenity spending columns |
| `CryoSpendConflict` | Indicator: CryoSleep == True but TotalSpend > 0 |
| `AgeGroup` | Binned age: Child, Young, Adult, Senior, Elder |
| `IsChild`, `IsSenior` | Binary flags for age <18 or >60 |
| `Used_*` | Binary flag for each spending column (spent >0) |
| `Log_*` | Log‑transformed spending: log1p(amount) |
| `GroupMeanAge` | Average age of the passenger's group |
| `GroupTotalSpend` | Total spending of the group |
| `GroupVIPRatio` | Proportion of VIP members in the group |
| `Title` | Extracted from `Name` (Mr, Mrs, Miss, Master, Rare) |

## 7. Model & Hyperparameters
We use **LightGBM** with the following tuned parameters:

```python
LGBMClassifier(
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



```markdown
## 8. Performance
- **5‑fold stratified cross‑validation accuracy:** 0.805 (±0.008)
- **Kaggle public leaderboard score:** 0.805

## 9. Authors
Curly Baboon @MLW – Group members:  
- Yang Chubing  
- Luo Yitong  
- Xu Xianliang  
- Sun Duoyang  

## 10. License
This project is for educational purposes only.