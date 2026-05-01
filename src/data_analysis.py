import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Create images directory if it doesn't exist
if not os.path.exists('../images'):
    os.makedirs('../images')

# 2. Load and clean data
try:
    df = pd.read_csv('../data/letter-recognition.csv')
    df.columns = df.columns.str.strip()
except FileNotFoundError:
    print("Error: Ensure letter-recognition.csv is in the '../data/' folder.")
    exit()

# 3. Visualization 1: Letter Distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='letter', order=sorted(df['letter'].unique()), palette='coolwarm')
plt.title("Frequency Distribution of Target Letters")
plt.xlabel("Letter Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig('../images/letter_distribution.png')
print("Saved: letter_distribution.png")

# 4. Visualization 2: Feature Variance (xbar)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='letter', y='xbar', order=sorted(df['letter'].unique()), palette='Set3')
plt.title("Variance of xbar feature across Letters")
plt.xlabel("Letter Class")
plt.ylabel("Mean x of on pixels (xbar)")
plt.tight_layout()
plt.savefig('../images/xbar_variance.png')
print("Saved: xbar_variance.png")
