import pandas as pd

df_weather = pd.read_csv('data/2006-2016.csv')
df_crude = pd.read_csv('data\Cushing_OK_WTI_Spot_Price_FOB.csv')
df_air = pd.read_csv('data\Los_Angeles_International_Airport_-_Passenger_Traffic_By_Terminal.csv')
df_gasoline = pd.read_csv('data\Los_Angeles_Regular_All_Formulations_Retail_Gasoline_Prices.csv')

import pandas as pd

# --- Step 1: Clean df_crude ---
df_crude.rename(columns={
    'Month': 'YearMonth',
    'Cushing OK WTI Spot Price FOB Dollars per Barrel': 'Crude_Oil_Price'
}, inplace=True)
df_crude['YearMonth'] = pd.to_datetime(df_crude['YearMonth'])

# --- Step 2: Clean df_gasoline ---
df_gasoline.rename(columns={
    'Month': 'YearMonth',
    'Los Angeles Regular All Formulations Retail Gasoline Prices Dollars per Gallon': 'Gasoline_Price'
}, inplace=True)
df_gasoline['YearMonth'] = pd.to_datetime(df_gasoline['YearMonth'])

# --- Step 3: Clean and aggregate df_air ---
df_air['ReportPeriod'] = pd.to_datetime(df_air['ReportPeriod'])
df_air['YearMonth'] = df_air['ReportPeriod'].dt.to_period('M').dt.to_timestamp()

# Group by YearMonth and Domestic_International
df_air_grouped = df_air.groupby(['YearMonth', 'Domestic_International'])['Passenger_Count'].sum().reset_index()

# Pivot to get Domestic and International in separate columns
df_air_pivot = df_air_grouped.pivot(index='YearMonth', columns='Domestic_International', values='Passenger_Count').reset_index()

# Rename columns
df_air_pivot.columns.name = None
df_air_pivot.rename(columns={
    'Domestic': 'Domestic_Passengers',
    'International': 'International_Passengers'
}, inplace=True)

# Fill missing values with 0 (in case there are months with no traffic for one type)
df_air_pivot.fillna(0, inplace=True)

# Create Total_Traffic
df_air_pivot['Total_Traffic'] = df_air_pivot['Domestic_Passengers'] + df_air_pivot['International_Passengers']

# --- Step 4: Clean df_weather ---
df_weather['YearMonth'] = pd.to_datetime(df_weather['YearMonth'])  # already monthly

# --- Step 5: Merge all datasets ---
df_merged = df_weather.merge(df_crude, on='YearMonth', how='inner')
df_merged = df_merged.merge(df_gasoline, on='YearMonth', how='inner')
df_merged = df_merged.merge(df_air_pivot, on='YearMonth', how='inner')

# --- Step 6: Check the result ---
print(df_merged.info())
print(df_merged.head())

df_merged.to_csv('data/merged_air_traffic_data.csv', index=False)