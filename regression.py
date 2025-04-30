import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import joblib

# Save the model

df_merged = pd.read_csv('data/merged_air_traffic_data.csv')


# Define features and target
features = ['AWND', 'TAVG', 'PRCP', 'Gasoline_Price']  # You can expand this!
target = 'Total_Traffic'

# ---- Checking seasonality of each feature ----
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt  # Already imported above

df_seasonality = df_merged.copy()
df_seasonality['YearMonth'] = pd.to_datetime(df_seasonality['YearMonth'])
df_seasonality.set_index('YearMonth', inplace=True)

for feature in features:
    try:
        decomposition = seasonal_decompose(df_seasonality[feature].dropna(), period=12, model='additive')
        fig = decomposition.plot()
        fig.suptitle(f'Seasonal Decomposition of {feature}')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Seasonal decomposition failed for {feature}: {e}")

# ---- Checking seasonality of Total_Air_Traffic ----
try:
    decomposition = seasonal_decompose(df_seasonality[target].dropna(), period=12, model='additive')
    fig = decomposition.plot()
    fig.suptitle('Seasonal Decomposition of Total Air Traffic')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Seasonal decomposition failed for Total Air Traffic: {e}")

# --- Part 1: Simple Regression Lines (visual for each predictor) ---

for feature in features:
    plt.figure(figsize=(8, 5))
    sns.regplot(x=df_merged[feature], y=df_merged[target], scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title(f'{target} vs {feature}')
    plt.xlabel(feature)
    plt.ylabel('Total Traffic')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Part 2: Multiple Linear Regression (quantitative) ---

# Prepare data
X = df_merged[features]
y = df_merged[target]

# Clean data: replace inf with NaN and drop rows with missing values
X = X.replace([np.inf, -np.inf], np.nan)
y = y.replace([np.inf, -np.inf], np.nan)
df_complete = pd.concat([X, y], axis=1).dropna()
X_clean = df_complete[features]
y_clean = df_complete[target]

# Add constant for statsmodels
X_sm = sm.add_constant(X_clean)

# Fit the model using cleaned data
model = sm.OLS(y_clean, X_sm).fit()

# Print the summary
print(model.summary())
plt.figure(figsize=(12, 5))
plt.plot(df_merged['YearMonth'], df_merged['AWND'], marker='o', label='Wind Speed (AWND)')
plt.plot(df_merged['YearMonth'], df_merged['Total_Traffic'] / 1e6, marker='s', label='Total Traffic (millions)')
plt.title('Wind Speed and Total Air Traffic Over Time')
plt.xlabel('YearMonth')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Added code for precipitation plotting
plt.figure(figsize=(12, 5))
plt.plot(df_merged['YearMonth'], df_merged['PRCP'], marker='o', label='Precipitation (PRCP)')
plt.plot(df_merged['YearMonth'], df_merged['Total_Traffic'] / 1e6, marker='s', label='Total Traffic (millions)')
plt.title('Precipitation and Total Air Traffic Over Time')
plt.xlabel('YearMonth')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

joblib.dump(model, 'air_traffic_model.pkl')