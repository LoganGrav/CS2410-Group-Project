import pandas as pd

df_weather1 = pd.read_csv('data/2006-2010.csv')
df_weather2 = pd.read_csv('data/2011-2016.csv')

all_columns = ['STATION', 'NAME', 'DATE', 'AWND', 'DAPR', 'MDPR', 'PRCP', 'SNOW', 'SNWD', 'TAVG', 'TMAX', 'TMIN', 'TOBS']

for df in [df_weather1, df_weather2]:
    for col in all_columns:
        if col not in df.columns:
            df[col] = None  # Add missing columns as None (will become NaN)

# Ensure 'DATE' column exists in both datasets
if 'DATE' not in df_weather1.columns or 'DATE' not in df_weather2.columns:
    raise ValueError("One of the input weather files is missing the 'DATE' column.")

# Step 2: Concatenate datasets
df_weather = pd.concat([df_weather1, df_weather2], ignore_index=True)

# Step 3: Convert DATE to datetime and create YearMonth
df_weather['DATE'] = pd.to_datetime(df_weather['DATE'])
df_weather['YearMonth'] = df_weather['DATE'].dt.to_period('M').dt.to_timestamp()

# Step 4: Aggregate weather data by month
agg_functions = {
    'AWND': 'mean',
    'DAPR': 'sum',
    'MDPR': 'sum',
    'PRCP': 'sum',
    'SNOW': 'sum',
    'SNWD': 'mean',
    'TAVG': 'mean',
    'TMAX': 'mean',
    'TMIN': 'mean',
    'TOBS': 'mean'
}

df_weather_monthly = df_weather.groupby('YearMonth').agg(agg_functions).reset_index()

print(df_weather_monthly.head())
df_weather_monthly.to_csv('data/2006-2016.csv', index=False)