from flask import Flask, render_template, request, redirect
import os
import pandas as pd
from utils import save_expense, get_expense_stats, train_category_model, train_spending_model, predict_next_spending

app = Flask(__name__)
CSV_FILE = 'expenses.csv'

# --------------------- Routes --------------------- #

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST' and request.form.get('predict'):
        prediction = predict_next_spending(CSV_FILE)

    stats = get_expense_stats(CSV_FILE)
    return render_template('index.html', prediction=prediction, **stats)

@app.route('/add', methods=['POST'])
def add():
    """Add new expense from form submission."""
    date = request.form['date']
    amount = float(request.form['amount'])
    description = request.form['description']
    save_expense(date, amount, description)
    return redirect('/')

@app.route('/view')
def view():
    """Display all expenses in a table."""
    if os.path.exists(CSV_FILE) and os.stat(CSV_FILE).st_size > 0:
        df = pd.read_csv(CSV_FILE)
        # Optional: sort by date descending
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(by='Date', ascending=False)
        table_html = df.to_html(classes='data', index=False)
    else:
        table_html = "<p>No expense records found.</p>"

    return render_template('view.html', tables=[table_html])

@app.route('/train_category')
def train_category():
    """Train category prediction model."""
    train_category_model(CSV_FILE)
    return redirect('/')

@app.route('/train_spending')
def train_spending():
    """Train monthly spending prediction model."""
    train_spending_model(CSV_FILE)
    return redirect('/')

@app.route('/predict_spending')
def predict_spending_route():
    """Predict next month's spending."""
    prediction = predict_next_spending(CSV_FILE)
    stats = get_expense_stats(CSV_FILE)
    return render_template('index.html', prediction=prediction, **stats)

# --------------------- Main --------------------- #
if __name__ == '__main__':
    app.run(debug=True)
