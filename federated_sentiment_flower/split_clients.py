import pandas as pd
from sklearn.model_selection import train_test_split
import os


# Load dataset
df = pd.read_csv('twitter_training.csv') # columns: text, label


# Shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)


# Split dataset into 3 parts
clients = 3
size = len(df) // clients


for i in range(clients):
    start = i * size
    end = (i + 1) * size if i < clients - 1 else len(df)
    subset = df.iloc[start:end]
    subset.to_csv(f'client_data_{i+1}.csv', index=False)
    print(f'Client {i+1} data saved with {len(subset)} rows')