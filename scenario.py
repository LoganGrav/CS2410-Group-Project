import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import joblib

# Load the model and data
model = joblib.load('air_traffic_model.pkl')
df_merged = pd.read_csv('data/merged_air_traffic_data.csv')
features = ['AWND', 'TAVG', 'PRCP', 'Gasoline_Price']

# Step 1: Create base case (average values)
base_case = df_merged[features].mean().to_frame().T

# Step 2: Create a range of gasoline prices (e.g., from min to max)
gas_price_range = np.linspace(df_merged['Gasoline_Price'].min(),
                              df_merged['Gasoline_Price'].max(),
                              100)  # 100 points for smooth curve

# Step 3: Create a DataFrame for all scenarios
scenario_df = pd.DataFrame({
    'Gasoline_Price': gas_price_range
})

# Keep other features constant (base case values)
for feature in features:
    if feature != 'Gasoline_Price':
        scenario_df[feature] = base_case.iloc[0][feature]
        
# Step 4: Add constant term for statsmodels
scenario_sm = sm.add_constant(scenario_df, has_constant='add')

# Step 5: Predict traffic
scenario_df['Predicted_Traffic'] = model.predict(scenario_sm)


# Step 6: Plot the result
plt.figure(figsize=(10, 6))
plt.plot(scenario_df['Gasoline_Price'], scenario_df['Predicted_Traffic'] / 1e6, color='red')
plt.xlabel('Gasoline Price (USD)')
plt.ylabel('Predicted Total Traffic (millions)')
plt.title('Continuous Scenario: Impact of Gasoline Price on Air Traffic')
plt.grid(True)
plt.tight_layout()
plt.show()
