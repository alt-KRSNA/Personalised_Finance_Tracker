# seed_data.py
import pandas as pd
import random
from faker import Faker

fake = Faker()

categories = ['Travel', 'Food', 'Party', 'Shopping', 'Health', 'Entertainment']

data = []
for _ in range(100):
    date = fake.date_this_decade()
    amount = round(random.uniform(100, 5000), 2)
    description = fake.city()
    category = random.choice(categories)
    data.append([date, amount, description, category])

df = pd.DataFrame(data, columns=['Date', 'Amount', 'Description', 'Category'])
df.to_csv('expenses.csv', index=False)
print("CSV file 'expenses.csv' created successfully!")
