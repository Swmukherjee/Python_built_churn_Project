import os, zipfile, requests

# 1. Create folders
folders = [
    "customer-churn-prediction/data",
    "customer-churn-prediction/notebooks",
    "customer-churn-prediction/scripts",
    "customer-churn-prediction/sql",
    "customer-churn-prediction/dashboard"
]
for f in folders:
    os.makedirs(f, exist_ok=True)

# 2. Download dataset (Telco Customer Churn public dataset)
url = "https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/Telco-Customer-Churn.csv"
dataset_path = "customer-churn-prediction/data/customer_churn.csv"
with open(dataset_path, "wb") as f:
    f.write(requests.get(url).content)

# 3. README.md
readme_content = """# Customer Churn Prediction & Analysis

This project analyses telecom customer churn using Python, SQL, Power BI, and Excel.

## Features
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Churn prediction models (Logistic Regression, Random Forest)
- Visual dashboards in Power BI and Excel

## Folder Structure
- data/ : dataset
- notebooks/ : Jupyter notebook for the full workflow
- scripts/ : Python scripts for modular tasks
- sql/ : SQL queries for churn KPIs
- dashboard/ : Power BI (.pbix) and Excel (.xlsx) dashboards

## How to Run
1. Install dependencies:
   pip install -r requirements.txt
2. Open churn_analysis.ipynb and run cells in order.
3. Explore dashboards in Power BI or Excel for visualization insights.
"""
open("customer-churn-prediction/README.md", "w").write(readme_content)

# 4. requirements.txt
reqs = """pandas
numpy
scikit-learn
matplotlib
seaborn
"""
open("customer-churn-prediction/requirements.txt", "w").write(reqs)

# 5. Example Python scripts

# Data cleaning
open("customer-churn-prediction/scripts/data_cleaning.py", "w").write(
"""import pandas as pd

def clean_data(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.fillna({'TotalCharges': df['TotalCharges'].median()}, inplace=True)
    return df
"""
)

# EDA
open("customer-churn-prediction/scripts/eda.py", "w").write(
"""import seaborn as sns
import matplotlib.pyplot as plt

def churn_distribution(df):
    sns.countplot(x='Churn', data=df)
    plt.title("Churn Distribution")
    plt.show()
"""
)

# Model training
open("customer-churn-prediction/scripts/model_training.py", "w").write(
"""from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_model(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
"""
)

# 6. SQL queries
open("customer-churn-prediction/sql/churn_analysis.sql", "w").write(
"""-- Churn rate by contract type
SELECT Contract,
       AVG(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churn_rate
FROM customers
GROUP BY Contract
ORDER BY churn_rate DESC;
"""
)

# 7. Dashboard placeholders (empty files for now)
open("customer-churn-prediction/dashboard/churn_dashboard.pbix", "wb").close()
open("customer-churn-prediction/dashboard/churn_dashboard.xlsx", "wb").close()
open("customer-churn-prediction/dashboard/dashboard_preview.pdf", "wb").close()

# 8. Zip the project
zip_path = "customer-churn-prediction.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk("customer-churn-prediction"):
        for file in files:
            z.write(os.path.join(root, file))

print(f"✅ Project zip created: {zip_path}")
