# 📦 Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
import os
from datetime import datetime

# 📁 Configuration
class Config:
    DATA_FILE = 'owid-covid-data.csv'
    OUTPUT_DIR = 'charts'
    COUNTRIES = ['Kenya', 'India', 'United States', 'Rwanda', 
                 'Brazil', 'Germany', 'South Africa', 'China']
    COLUMNS = ['iso_code', 'continent', 'location', 'date', 
               'total_cases', 'new_cases', 'total_deaths', 'new_deaths',
               'total_vaccinations', 'people_vaccinated', 'population']
    PLOT_STYLE = 'seaborn-v0_8'
    PLOT_SIZE = (12, 7)

# Set style and create output directory
try:
    plt.style.use(Config.PLOT_STYLE)
except OSError:
    print(f"Style '{Config.PLOT_STYLE}' not found. Using default style.")
    print(f"Available styles: {plt.style.available}")
    plt.style.use('ggplot')

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load and validate COVID-19 dataset"""
    try:
        if not os.path.exists(Config.DATA_FILE):
            raise FileNotFoundError(f"Data file '{Config.DATA_FILE}' not found")
            
        df = pd.read_csv(Config.DATA_FILE)
        print(f"✅ Data loaded successfully with {len(df)} records")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None

def clean_data(df):
    """Clean and preprocess the dataset"""
    if df is None:
        return None
        
    try:
        # Create a copy of the dataframe to avoid SettingWithCopyWarning
        df_clean = df.copy()
        
        # Select relevant columns (only those that exist in the dataframe)
        available_cols = [col for col in Config.COLUMNS if col in df_clean.columns]
        missing_cols = set(Config.COLUMNS) - set(available_cols)
        if missing_cols:
            print(f"⚠️ Missing columns: {missing_cols}")
            
        df_clean = df_clean[available_cols]
        
        # Convert and clean date
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        df_clean = df_clean[df_clean['date'].notna()]
        
        # Handle missing values
        numeric_cols = df_clean.select_dtypes(include=np.number).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
        
        # Filter countries
        df_clean = df_clean[df_clean['location'].isin(Config.COUNTRIES)]
        
        # Calculate derived metrics
        df_clean['death_rate'] = np.where(
            df_clean['total_cases'] > 0,
            df_clean['total_deaths'] / df_clean['total_cases'],
            0
        )
        
        if 'population' in df_clean.columns and 'people_vaccinated' in df_clean.columns:
            df_clean['vaccination_rate'] = df_clean['people_vaccinated'] / df_clean['population']
        
        return df_clean
    except Exception as e:
        print(f"❌ Error cleaning data: {str(e)}")
        return None

def plot_time_series(df, column, title, ylabel, output_file):
    """Plot a time series for a specific column."""
    try:
        plt.figure(figsize=Config.PLOT_SIZE)
        for country in Config.COUNTRIES:
            country_data = df[df['location'] == country]
            plt.plot(country_data['date'], country_data[column], label=country)
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        output_path = os.path.join(Config.OUTPUT_DIR, output_file)
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"✅ Saved plot: {output_path}")
    except Exception as e:
        print(f"❌ Error generating plot '{title}': {str(e)}")

def generate_vaccination_pie_charts(df):
    """Generate pie charts for vaccination rates by country."""
    try:
        latest_data = df[df['date'] == df['date'].max()]
        
        for country in Config.COUNTRIES:
            country_data = latest_data[latest_data['location'] == country]
            if not country_data.empty and 'vaccination_rate' in country_data.columns:
                vaccination_rate = country_data['vaccination_rate'].values[0] * 100
                labels = ['Vaccinated', 'Not Vaccinated']
                sizes = [vaccination_rate, 100 - vaccination_rate]
                plt.figure(figsize=(8, 8))
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, 
                        colors=['#4CAF50', '#FFC107'], wedgeprops={'edgecolor': 'white'})
                plt.title(f'Vaccination Rate in {country}\n({latest_data["date"].dt.strftime("%Y-%m-%d").values[0]})',
                         pad=20)
                output_path = os.path.join(Config.OUTPUT_DIR, f'{country}_vaccination_pie_chart.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ Saved pie chart: {output_path}")
    except Exception as e:
        print(f"❌ Error generating vaccination pie charts: {str(e)}")

def generate_geopandas_choropleth(df):
    """Generate static choropleth map using GeoPandas"""
    try:
        # Try to load world map data from local file first
        try:
            world = gpd.read_file('ne_110m_admin_0_countries.shp')
            print("✅ Loaded world map data from local file")
        except:
            # Fallback to online source if local file not found
            try:
                world = gpd.read_file('https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip')
                print("✅ Loaded world map data from online source")
            except Exception as e:
                print(f"❌ Error loading world map data: {str(e)}")
                print("Please download the Natural Earth data manually from:")
                print("https://www.naturalearthdata.com/downloads/110m-cultural-vectors/")
                print("And place 'ne_110m_admin_0_countries.shp' in your project directory")
                return
        
        # Standardize column names
        world = world.rename(columns={'ISO_A3': 'iso_code'})
        
        # Convert to a projected CRS (World Mercator) for accurate centroid calculations
        world = world.to_crs('EPSG:3395')
        
        # Prepare the COVID data
        latest_data = df[df['date'] == df['date'].max()]
        
        # Merge with world map data
        merged = world.merge(
            latest_data,
            on='iso_code',
            how='left'
        )
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        merged.plot(
            column='total_cases',
            cmap='OrRd',
            linewidth=0.8,
            ax=ax,
            edgecolor='0.8',
            legend=True,
            legend_kwds={'label': "Total Cases", 'orientation': "horizontal"},
            missing_kwds={'color': 'lightgrey'}
        )
        
        # Customize the plot
        ax.set_title('Global COVID-19 Cases', fontsize=20, pad=20)
        ax.set_axis_off()
        
        # Add country labels for our selected countries
        for country in Config.COUNTRIES:
            country_data = latest_data[latest_data['location'] == country]
            if not country_data.empty:
                iso_code = country_data['iso_code'].values[0]
                country_geo = merged[merged['iso_code'] == iso_code]
                if not country_geo.empty and not country_geo.geometry.isna().any():
                    centroid = country_geo.geometry.centroid.iloc[0]
                    ax.text(
                        centroid.x,
                        centroid.y,
                        f"{country}\n{int(country_data['total_cases'].values[0]):,}",
                        fontsize=8,
                        ha='center',
                        va='center',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                    )
        
        # Save the plot
        output_path = os.path.join(Config.OUTPUT_DIR, 'global_cases_geopandas.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved GeoPandas choropleth: {output_path}")
        
    except Exception as e:
        print(f"❌ Error generating GeoPandas choropleth: {str(e)}")
        
def generate_plotly_choropleth(df):
    """Generate interactive choropleth with Plotly"""
    try:
        # Prepare data
        latest_global = df[['iso_code', 'location', 'date', 'total_cases', 'total_deaths', 'total_vaccinations']].copy()
        latest_date = latest_global['date'].max()
        latest_global = latest_global[latest_global['date'] == latest_date]
        
        # Create choropleth map
        fig = px.choropleth(
            latest_global,
            locations='iso_code',
            color='total_cases',
            hover_name='location',
            hover_data={
                'iso_code': False,
                'total_cases': ':,',
                'total_deaths': ':,',
                'total_vaccinations': ':,'
            },
            title=f'Global COVID-19 Cases as of {latest_date.strftime("%Y-%m-%d")}',
            color_continuous_scale='Viridis',
            projection='natural earth',
            labels={'total_cases': 'Total Cases'}
        )
        
        # Customize hover template
        fig.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>" +
                          "Total Cases: %{customdata[0]:,}<br>" +
                          "Total Deaths: %{customdata[1]:,}<br>" +
                          "Total Vaccinations: %{customdata[2]:,}"
        )
        
        # Update layout
        fig.update_layout(
            margin={"r":0,"t":40,"l":0,"b":0},
            coloraxis_colorbar={
                'title': 'Total Cases',
                'thickness': 20,
                'len': 0.5
            }
        )
        
        # Save as HTML
        output_path = os.path.join(Config.OUTPUT_DIR, 'global_cases_plotly.html')
        fig.write_html(output_path, include_plotlyjs='cdn')
        print(f"✅ Saved Plotly choropleth: {output_path}")
        
    except Exception as e:
        print(f"❌ Error generating Plotly choropleth: {str(e)}")

def generate_report(df):
    """Generate summary statistics and insights"""
    if df is None:
        return
        
    try:
        latest_date = df['date'].max()
        
        def print_separator():
            """Print a separator line."""
            print("\n" + "=" * 50)
        
        print_separator()
        print("COVID-19 DATA ANALYSIS REPORT")
        print(f"Last Updated: {latest_date.strftime('%Y-%m-%d')}".center(50))
        print("=" * 50 + "\n")
        
        # Summary statistics
        latest_data = df[df['date'] == latest_date]
        print("Summary Statistics:")
        print(latest_data[['location', 'total_cases', 'total_deaths', 'total_vaccinations']]
              .set_index('location')
              .map(lambda x: f"{int(x):,}" if pd.notnull(x) else "N/A"))
        
        # Key insights
        print("\nKey Insights:")
        insights = [
            "1. The United States consistently shows the highest total cases and vaccination numbers",
            "2. Developing nations like Kenya and Rwanda show slower vaccination rollouts",
            "3. Brazil and India experienced significant waves of infections at different times",
            "4. Death rates vary significantly between countries, suggesting differences in healthcare capacity",
            "5. Vaccination rates in developed nations (US, Germany) outpace developing nations"
        ]
        
        for insight in insights:
            print(f"- {insight}")
            
        print("\n" + "=" * 50)
        
    except Exception as e:
        print(f"❌ Failed to generate report: {str(e)}")

# Main execution
if __name__ == "__main__":
    
    print("\nStarting COVID-19 Data Analysis...\n")
    
    # Load and process data
    df = load_data()
    df_clean = clean_data(df)
    
    if df_clean is not None:
        # Generate visualizations
        plot_time_series(df_clean, 'total_cases', 'Total COVID-19 Cases Over Time', 'Total Cases', 'total_cases.png')
        plot_time_series(df_clean, 'total_deaths', 'Total COVID-19 Deaths Over Time', 'Total Deaths', 'total_deaths.png')
        plot_time_series(df_clean, 'new_cases', 'Daily New COVID-19 Cases', 'New Cases', 'new_cases.png')
        plot_time_series(df_clean, 'death_rate', 'COVID-19 Death Rate Over Time', 'Death Rate', 'death_rate.png')
        
        if 'total_vaccinations' in df_clean.columns:
            plot_time_series(df_clean, 'total_vaccinations', 'Total Vaccinations Over Time', 'Vaccinations', 'vaccinations.png')
            generate_vaccination_pie_charts(df_clean)
        
        # Generate both types of choropleth maps
        generate_geopandas_choropleth(df_clean)
        generate_plotly_choropleth(df_clean)
        
        generate_report(df_clean)
    
    print("\nAnalysis complete! \nAll outputs saved to the 'charts' directory.")