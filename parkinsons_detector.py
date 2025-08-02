# Parkinson's Detection using only Pandas & Numpy
# Created by: [Your Name]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('parkinsons_data.csv')

# Remove 'name' column if it exists
if 'name' in data.columns:
    data = data.drop(columns=['name'])

# ------------------------------
# Rule-based prediction function
# ------------------------------
def rule_based_prediction(row):
    jitter = row['MDVP:Jitter(%)']
    shimmer = row['MDVP:Shimmer']
    nhr = row['NHR']

    # Custom rules based on observation
    if jitter > 0.006 and shimmer > 0.035 and nhr > 0.02:
        return 1  # Likely Parkinson's
    else:
        return 0  # Likely Healthy

# Apply rule to each row
data['prediction'] = data.apply(rule_based_prediction, axis=1)

# ------------------------------
# Accuracy check (if 'status' exists)
# ------------------------------
if 'status' in data.columns:
    correct = np.sum(data['status'] == data['prediction'])
    total = len(data)
    accuracy = (correct / total) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")

    # Manual confusion matrix
    tp = np.sum((data['status'] == 1) & (data['prediction'] == 1))
    tn = np.sum((data['status'] == 0) & (data['prediction'] == 0))
    fp = np.sum((data['status'] == 0) & (data['prediction'] == 1))
    fn = np.sum((data['status'] == 1) & (data['prediction'] == 0))

    print("\nConfusion Matrix:")
    print(f"True Positive:  {tp}")
    print(f"True Negative:  {tn}")
    print(f"False Positive: {fp}")
    print(f"False Negative: {fn}")

# ------------------------------
# Show some sample predictions
# ------------------------------
print("\nSample Predictions:")
print(data[['MDVP:Jitter(%)', 'MDVP:Shimmer', 'NHR', 'prediction']].head())

# ------------------------------
# Histogram visualization
# ------------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(data['MDVP:Jitter(%)'], bins=15, color='skyblue', edgecolor='black')
plt.title('Jitter (%) Distribution')
plt.xlabel('Jitter (%)')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.hist(data['MDVP:Shimmer'], bins=15, color='salmon', edgecolor='black')
plt.title('Shimmer Distribution')
plt.xlabel('Shimmer')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# ------------------------------
# Scatter plot visualization
# ------------------------------
colors = ['green' if p == 0 else 'red' for p in data['prediction']]

plt.figure(figsize=(6, 6))
plt.scatter(data['MDVP:Jitter(%)'], data['MDVP:Shimmer'], c=colors)
plt.xlabel('Jitter (%)')
plt.ylabel('Shimmer')
plt.title("Parkinson's Prediction (Red = Parkinson's, Green = Healthy)")
plt.grid(True)
plt.show()
