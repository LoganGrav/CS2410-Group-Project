# this program models various LAX air traffic datasets

# 'passenger_per_terminal.csv' contains passengers per terminal in LAX for different years
# https://catalog.data.gov/dataset/los-angeles-international-airport-passenger-traffic-by-terminal

# 'unemployment_rates.csv' contains data about the unemployment rate of Los Angeles, Long Beach, and Anaheim per month
# https://fred.stlouisfed.org/series/LOSA106UR 

# IGNORE THE FOLLOWING WHEN WORKING SOLELY WITH Pandas OR MatPlotLib FUNCTIONS:
# import sys
# caution: path[0] is reserved for this script path
# sys.path.insert(1, r"C:\Users\joshy\Desktop\Foundations of DS\projects\Project 2")
# import the module containing methods 'my_statsfuncts.py'
# import my_statsfuncts as msf

import pandas as pd
import matplotlib.pyplot as plt


# STEP 1: Get the passenger count per year
df_pass = pd.read_csv('passenger_per_terminal.csv')

# convert 'ReportPeriod' to DateTime format
df_pass['ReportPeriod'] = pd.to_datetime(df_pass['ReportPeriod']) 
df_pass['Year'] = df_pass['ReportPeriod'].dt.year

yearly_count = df_pass.groupby('Year').size() # gets the size of each column by 'Year'
print(yearly_count)

# group instances by year and aggregate passenger count, as a series
yearly_passengers = df_pass.groupby('Year')['Passenger_Count'].mean()
yearly_passengers = pd.DataFrame(yearly_passengers) # convert to dataframe



# STEP 2: Get the unemployment rate per year
df_unemp = pd.read_csv('unemployment_rates.csv')

# convert 'observation_date' to DateTime format
df_unemp['observation_date'] = pd.to_datetime(df_unemp['observation_date'])
df_unemp['Year'] = df_unemp['observation_date'].dt.year # extract the year

# group instances by year and take the mean
yearly_unemployment_rates = df_unemp.groupby('Year')['LOSA106UR'].mean()
yearly_unemployment_rates = pd.DataFrame(yearly_unemployment_rates) # make into a dataframe



# STEP 3: Combine both dataframes by their commonalities (a.k.a. intersection or inner)
df_combin = yearly_passengers.merge(yearly_unemployment_rates, left_index=True, right_index=True, how='inner', sort=True)
df_combin = df_combin.rename(columns={ # reformat the dataframe with CLEAR labels
    'Passenger_Count': 'Passengers',
    'LOSA106UR': 'Unemployment_Rate'
})
print(df_combin)

# STEP 4: Take Pearson's Coefficient of the new dataframe
pearson_corr = df_combin['Passengers'].corr(df_combin['Unemployment_Rate'])
print(f'Pearson\'s Correlation between Passengers and Unemployment Rate: {pearson_corr:.4f}')

# FINAL STEP: Plotting
fig, ax1 = plt.subplots()

# Plot scaled passenger count
ax1.plot(df_combin.index, df_combin['Passengers'], color='blue', marker='o', label='LAX Passengers')
ax1.set_xlabel("Year")
ax1.set_ylabel("No. of Passengers (in tens of millions)", color='black')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True)

# Plot unemployment rate on a secondary axis
ax2 = ax1.twinx()
ax2.plot(df_combin.index, df_combin['Unemployment_Rate'], color='red', marker='o', linestyle='--', label='Unemployment Rate')
ax2.set_ylabel("Unemployment Rate (in %)", color='black')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(0, 15)

plt.xticks(df_combin.index)
plt.title('LAX Passenger Traffic and Unemployment Rate From 2006 to 2023')
fig.tight_layout()
plt.show()

# For more info on the datasets:
# df_passengers.info()
# df_passengers.describe()
# df_employmentData.info()
# df_employmentData.describe()

print("")