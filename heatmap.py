import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

df_merged = pd.read_csv('data/merged_air_traffic_data.csv')

# Optional: If your dataset has 'YearMonth' or other non-numeric columns, drop them for correlation
numeric_df = df_merged.select_dtypes(include=['float64', 'int64'])

# Generate correlation matrix
corr_matrix = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=0.5)
plt.title('Correlation Heatmap: Air Traffic, Weather, and Prices', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
