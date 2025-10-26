import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

# Paths for trained models
CATEGORY_MODEL_PATH = os.path.join("models", "expense_classifier.pkl")
SPENDING_MODEL_PATH = os.path.join("models", "spending_predictor.pkl")

# --------------------- CSV Handling --------------------- #

def load_expenses(csv_path='expenses.csv'):
    """Load expenses CSV, return empty DataFrame if not exists."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=["Date", "Amount", "Description", "Category"])
    return df

def save_expenses(df, csv_path='expenses.csv'):
    """Save the entire DataFrame to CSV."""
    df.to_csv(csv_path, index=False)

def save_expense(date, amount, description, category='Other', csv_path='expenses.csv'):
    """Append a single expense to CSV."""
    new_row = pd.DataFrame([{
        "Date": date,
        "Amount": amount,
        "Description": description,
        "Category": category
    }])
    if os.path.exists(csv_path) and os.stat(csv_path).st_size > 0:
        new_row.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        new_row.to_csv(csv_path, index=False)

# --------------------- Category Prediction --------------------- #

def predict_category(description):
    """Predict expense category using the trained model, fallback to 'Unknown'."""
    if not os.path.exists(CATEGORY_MODEL_PATH):
        return "Unknown"
    with open(CATEGORY_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    try:
        return model.predict([description])[0]
    except Exception:
        return "Unknown"

def train_category_model(csv_path='expenses.csv'):
    """Train a category classification model from CSV descriptions."""
    df = load_expenses(csv_path)
    if 'Description' not in df.columns or 'Category' not in df.columns:
        return "No suitable columns to train."
    
    df = df.dropna(subset=['Description', 'Category'])
    if df.empty:
        return "No data to train on."

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(df['Description'].astype(str), df['Category'].astype(str))

    # Ensure models folder exists
    os.makedirs(os.path.dirname(CATEGORY_MODEL_PATH) or '.', exist_ok=True)
    with open(CATEGORY_MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)

    return "✅ Category prediction model trained successfully."

# --------------------- Spending Prediction --------------------- #

def train_spending_model(csv_path='expenses.csv'):
    """Train a Linear Regression model to predict next month spending."""
    df = load_expenses(csv_path)
    if 'Date' not in df.columns or 'Amount' not in df.columns:
        return "No suitable columns for spending model."

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df = df.dropna(subset=['Date', 'Amount'])
    if df.empty:
        return "No valid data for training spending model."

    monthly = df.groupby(df['Date'].dt.to_period('M'))['Amount'].sum().reset_index()
    monthly['Index'] = range(len(monthly))

    X = monthly[['Index']].values
    y = monthly['Amount'].values

    model = LinearRegression()
    model.fit(X, y)

    os.makedirs(os.path.dirname(SPENDING_MODEL_PATH) or '.', exist_ok=True)
    with open(SPENDING_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    return "✅ Spending prediction model trained successfully."

def predict_next_spending(csv_path='expenses.csv'):
    """Predict next month's spending using trained Linear Regression model."""
    if not os.path.exists(SPENDING_MODEL_PATH):
        return 0.0

    with open(SPENDING_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    df = load_expenses(csv_path)
    if 'Date' not in df.columns or 'Amount' not in df.columns:
        return 0.0

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df = df.dropna(subset=['Date', 'Amount'])
    if df.empty:
        return 0.0

    monthly = df.groupby(df['Date'].dt.to_period('M'))['Amount'].sum().reset_index()
    next_index = len(monthly)

    try:
        prediction = model.predict(np.array([[next_index]]))[0]
        return round(float(prediction), 2)
    except Exception as e:
        print("Prediction error:", e)
        return 0.0

# --------------------- Expense Stats --------------------- #

def get_expense_stats(csv_path='expenses.csv'):
    """Return dict of current month, last month, avg monthly, and total expenses."""
    if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
        return {
            'current_month_total': 0,
            'last_month_total': 0,
            'avg_monthly': 0,
            'total_expenses': 0
        }

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    if 'date' not in df.columns or 'amount' not in df.columns:
        return {
            'current_month_total': 0,
            'last_month_total': 0,
            'avg_monthly': 0,
            'total_expenses': 0
        }

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df.dropna(subset=['date', 'amount'])

    now = datetime.now()
    current_month = now.month
    current_year = now.year

    current_month_exp = df[(df['date'].dt.month == current_month) & (df['date'].dt.year == current_year)]
    last_month = current_month - 1 if current_month > 1 else 12
    last_month_year = current_year if current_month > 1 else current_year - 1
    last_month_exp = df[(df['date'].dt.month == last_month) & (df['date'].dt.year == last_month_year)]

    df['month'] = df['date'].dt.to_period('M')
    avg_monthly = df.groupby('month')['amount'].sum().mean()

    return {
        'current_month_total': float(current_month_exp['amount'].sum()),
        'last_month_total': float(last_month_exp['amount'].sum()),
        'avg_monthly': float(avg_monthly) if not np.isnan(avg_monthly) else 0.0,
        'total_expenses': float(df['amount'].sum())
    }
