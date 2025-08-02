import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('parkinsons_data.csv')
if 'name' in data.columns:
    data = data.drop(columns=['name'])

def rule_based_prediction(row):
    if row['MDVP:Jitter(%)'] > 0.006 and row['MDVP:Shimmer'] > 0.035 and row['NHR'] > 0.02:
        return 1
    return 0

data['prediction'] = data.apply(rule_based_prediction, axis=1)

if 'status' in data.columns:
    accuracy = (data['status'] == data['prediction']).mean() * 100
    print(f"\nAccuracy: {accuracy:.2f}%")

    tp = np.sum((data['status'] == 1) & (data['prediction'] == 1))
    tn = np.sum((data['status'] == 0) & (data['prediction'] == 0))
    fp = np.sum((data['status'] == 0) & (data['prediction'] == 1))
    fn = np.sum((data['status'] == 1) & (data['prediction'] == 0))

    print("\nConfusion Matrix:")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

print("\nSample Predictions:")
print(data[['MDVP:Jitter(%)', 'MDVP:Shimmer', 'NHR', 'prediction']].head())

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

colors = ['green' if p == 0 else 'red' for p in data['prediction']]
plt.figure(figsize=(6, 6))
plt.scatter(data['MDVP:Jitter(%)'], data['MDVP:Shimmer'], c=colors)
plt.xlabel('Jitter (%)')
plt.ylabel('Shimmer')
plt.title("Parkinson's Prediction (Red = Parkinson's, Green = Healthy)")
plt.grid(True)
plt.show()
