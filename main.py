# 📦 Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# 📁 Create output directory for charts
output_dir = 'visuals' #visuals
os.makedirs(output_dir, exist_ok=True)

# 📁 Load Dataset
if 'owid-covid-data.csv':
    df = pd.read_csv('owid-covid-data.csv')  # Ensure this file is in the same directory
else:
    print('File not found!')


# def write_image(image_data, output_dir):
#     with open(output_dir, 'wb') as f:
#         f.write(image_data)

# Reload global dataset and extract latest data
latest_global = pd.read_csv('owid-covid-data.csv')
latest_global = latest_global[['iso_code', 'location', 'date', 'total_cases']]

# FIXED datetime parsing
latest_global['date'] = pd.to_datetime(latest_global['date'], dayfirst=True, errors='coerce')

# Filter to latest date only
latest_date = latest_global['date'].max()
latest_global = latest_global[latest_global['date'] == latest_date]

# 🎯 Select Relevant Columns
df = df[['iso_code', 'continent', 'location', 'date', 
         'total_cases', 'new_cases', 'total_deaths', 'new_deaths', 
         'total_vaccinations', 'people_vaccinated']]

# 🧹 Data Cleaning
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['date', 'location'], inplace=True)
df.fillna(0, inplace=True)

# 🌍 Select Countries to Analyze
countries = ['Kenya', 'India', 'United States', 'Rwanda', 'Brazil', 'Germany', 'South Africa', 'China']

df = df[df['location'].isin(countries)]

# 📈 Total Cases Over Time
plt.figure(figsize=(10, 6))
for country in countries:
    data = df[df['location'] == country]
    plt.plot(data['date'], data['total_cases'], label=country)
plt.title('Total COVID-19 Cases Over Time')
plt.xlabel('Date')
plt.ylabel('Total Cases')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'total_cases_over_time.png'))
plt.close()

# 📉 Total Deaths Over Time
plt.figure(figsize=(10, 6))
for country in countries:
    data = df[df['location'] == country]
    plt.plot(data['date'], data['total_deaths'], label=country)
plt.title('Total COVID-19 Deaths Over Time')
plt.xlabel('Date')
plt.ylabel('Total Deaths')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'total_deaths_over_time.png'))
plt.close()

# 📊 Daily New Cases
plt.figure(figsize=(10, 6))
for country in countries:
    data = df[df['location'] == country]
    plt.plot(data['date'], data['new_cases'], label=country)
plt.title('Daily New COVID-19 Cases')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'new_cases_over_time.png'))
plt.close()

# ⚰️ Death Rate Over Time
df['death_rate'] = df['total_deaths'] / df['total_cases']
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df['death_rate'] = df['death_rate'].fillna(0)

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='date', y='death_rate', hue='location')
plt.title('COVID-19 Death Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Death Rate')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'death_rate_over_time.png'))
plt.close()

# 💉 Cumulative Vaccinations Over Time
plt.figure(figsize=(10, 6))
for country in countries:
    data = df[df['location'] == country]
    plt.plot(data['date'], data['total_vaccinations'], label=country)
plt.title('Total Vaccinations Over Time')
plt.xlabel('Date')
plt.ylabel('Total Vaccinations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'total_vaccinations_over_time.png'))
plt.close()

# 🧩 Pie Charts for Vaccination Rate
latest = df[df['date'] == df['date'].max()]
for country in countries:
    country_data = latest[latest['location'] == country]
    if not country_data.empty:
        vaccinated = float(country_data['people_vaccinated'].values[0])
        vaccinated = max(0, vaccinated)
        total_population_estimate = vaccinated / 0.7 if vaccinated > 0 else 1  # assume 70% if no population data
        unvaccinated = total_population_estimate - vaccinated

        plt.figure()
        plt.pie([vaccinated, unvaccinated],
                labels=['Vaccinated', 'Unvaccinated'],
                autopct='%1.1f%%', colors=['green', 'lightgrey'])
        plt.title(f'Vaccination Rate in {country}')
        plt.axis('equal')
        plt.savefig(os.path.join(output_dir, f'vaccination_rate_{country}.png'))
        plt.close()
        print(f"Vaccination pie chart saved for {country}")
    else:
        print(f"No recent data available for {country}")

# 🌍 Choropleth Map (Total Cases Latest)
latest_global = pd.read_csv('owid-covid-data.csv')
latest_global = latest_global[['iso_code', 'location', 'date', 'total_cases']]
latest_global['date'] = pd.to_datetime(latest_global['date'], dayfirst=True, errors='coerce')
latest_date = latest_global['date'].max()
latest_global = latest_global[latest_global['date'] == latest_date]

fig = px.choropleth(latest_global,
                    locations='iso_code',
                    color='total_cases',
                    hover_name='location',
                    title='Total COVID-19 Cases by Country',
                    color_continuous_scale='Reds')

#fig.write_image(os.path.join(output_dir, 'choropleth_map.png'))
#fig.write_image(os.path.join(output_dir, 'choropleth_map.png'))
fig.show()

# 📋 Summary of Key Insights (Write in Markdown if using Jupyter Notebook)

# 1. The US had the highest number of total cases and vaccinations.
# 2. India's case spike occurred around mid-2021, with a visible second wave.
# 3. Kenya maintained lower total cases but had slower vaccination rollout.
# 4. Death rates varied significantly over time and between countries.
# 5. Vaccination rollout improved sharply in late 2021 for all countries.

# 📌 END OF NOTEBOOK
