import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV files
traffic_df = pd.read_csv("formatted_traffic.csv")
inflation_df = pd.read_csv("Inflation.csv", header=None, names=["inflation_value"])
gdp_df = pd.read_csv("gdp.csv")


def get_quarter_date(row):
    quarter_str = row['quarter'][-2:]
    mapping = {'Q1': '01-01', 'Q2': '04-01', 'Q3': '07-01', 'Q4': '10-01'}
    return pd.to_datetime(f"{row['year']}-{mapping[quarter_str]}")

traffic_df['date'] = traffic_df.apply(get_quarter_date, axis=1)


start_date = pd.to_datetime("2006-01-01")  # Modify this if inflation data starts earlier
inflation_df['date'] = [start_date + pd.DateOffset(months=3 * i) for i in range(len(inflation_df))]


gdp_df['date'] = pd.to_datetime(gdp_df['observation_date'])


# Merge Traffic vs Inflation (keeping all inflation points)
merged_inflation = pd.merge(traffic_df, inflation_df, on="date", how="outer").sort_values("date")

# Merge Traffic vs GDP
merged_gdp = pd.merge(traffic_df, gdp_df, on="date", how="outer").sort_values("date")

fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot Traffic vs Inflation
sns.lineplot(ax=axes[0], x='date', y='traffic_value', data=merged_inflation, marker='o', label='Traffic')
sns.lineplot(ax=axes[0], x='date', y='inflation_value', data=merged_inflation, marker='o', label='Inflation')
axes[0].set_title("Traffic vs Inflation (Full Dataset)")
axes[0].set_ylabel("Value")
axes[0].legend()

# Plot Traffic vs GDP
sns.lineplot(ax=axes[1], x='date', y='traffic_value', data=merged_gdp, marker='o', label='Traffic')
sns.lineplot(ax=axes[1], x='date', y='CANQGSP', data=merged_gdp, marker='o', label='GDP')
axes[1].set_title("Traffic vs GDP (Full Dataset)")
axes[1].set_ylabel("Value")
axes[1].legend()

plt.xlabel("Date")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

merged_df = pd.merge(traffic_df, inflation_df, on="date", how="outer")
merged_df = pd.merge(merged_df, gdp_df, on="date", how="outer")

# Compute correlation between traffic and inflation
traffic_inflation_corr = merged_df["traffic_value"].corr(merged_df["inflation_value"])

# Compute correlation between traffic and GDP
traffic_gdp_corr = merged_df["traffic_value"].corr(merged_df["CANQGSP"])

# Print results
print(f"Correlation between Air Traffic and Inflation: {traffic_inflation_corr:.4f}")
print(f"Correlation between Air Traffic and GDP: {traffic_gdp_corr:.4f}")

