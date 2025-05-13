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
from matplotlib import ticker
from matplotlib import rcParams
import textwrap

# 📁 Configuration
class Config:
    DATA_FILE = 'owid-covid-data.csv'
    OUTPUT_DIR = 'output'
    COUNTRIES = ['Kenya', 'India', 'United States', 'Rwanda', 
                 'Brazil', 'Germany', 'South Africa', 'China']
    COLUMNS = ['iso_code', 'continent', 'location', 'date', 
               'total_cases', 'new_cases', 'total_deaths', 'new_deaths',
               'total_vaccinations', 'people_vaccinated', 'population',
               'icu_patients', 'hosp_patients', 'reproduction_rate']
    PLOT_STYLE = 'seaborn-v0_8-darkgrid'
    PLOT_SIZE = (14, 8)
    COLOR_PALETTE = {
        'cases': '#FF6B6B',
        'deaths': '#4ECDC4',
        'vaccinations': '#45B7D1',
        'recovery': '#A5D8A2',
        'active': '#FFD166',
        'background': '#F7F7F7',
        'text': '#333333'
    }
    FONT = 'Segoe UI'
    DPI = 300
    DARK_MODE = False

# Set style and create output directory
try:
    plt.style.use(Config.PLOT_STYLE)
    plt.rcParams['font.family'] = Config.FONT
    plt.rcParams['axes.titlepad'] = 20
    plt.rcParams['axes.labelpad'] = 10
except OSError:
    print(f"Style '{Config.PLOT_STYLE}' not found. Using default style.")
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
    """Clean and preprocess the dataset with enhanced features"""
    if df is None:
        return None

    try:
        return _extracted_from_clean_data_8(df)
    except Exception as e:
        print(f"❌ Error cleaning data: {str(e)}")
        return None


# TODO Rename this here and in `clean_data`
def _extracted_from_clean_data_8(df):
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df_clean = df.copy()

    # Select relevant columns (only those that exist in the dataframe)
    available_cols = [col for col in Config.COLUMNS if col in df_clean.columns]
    if missing_cols := set(Config.COLUMNS) - set(available_cols):
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

    # Calculate derived metrics with enhanced calculations
    df_clean['death_rate'] = np.where(
        df_clean['total_cases'] > 0,
        df_clean['total_deaths'] / df_clean['total_cases'],
        0
    )

    # 7-day moving averages for smoothing
    df_clean.sort_values(['location', 'date'], inplace=True)
    df_clean['new_cases_7day_avg'] = df_clean.groupby('location')['new_cases'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    df_clean['new_deaths_7day_avg'] = df_clean.groupby('location')['new_deaths'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )

    if 'population' in df_clean.columns:
        df_clean['cases_per_million'] = df_clean['total_cases'] / (df_clean['population'] / 1e6)
        df_clean['deaths_per_million'] = df_clean['total_deaths'] / (df_clean['population'] / 1e6)

        if 'people_vaccinated' in df_clean.columns:
            df_clean['vaccination_rate'] = df_clean['people_vaccinated'] / df_clean['population']
            df_clean['vaccinations_per_hundred'] = df_clean['vaccination_rate'] * 100

    return df_clean

def plot_time_series(df, column, title, ylabel, output_file, log_scale=False):
    """Enhanced time series plotting with professional styling"""
    try:
        plt.figure(figsize=Config.PLOT_SIZE, facecolor=Config.COLOR_PALETTE['background'])
        
        # Create a custom color palette for countries
        country_palette = sns.color_palette("husl", len(Config.COUNTRIES))
        
        for i, country in enumerate(Config.COUNTRIES):
            country_data = df[df['location'] == country]
            if not country_data.empty:
                plt.plot(country_data['date'], country_data[column], 
                         label=country, 
                         linewidth=2.5,
                         color=country_palette[i])
        
        # Formatting
        plt.title(title, fontsize=16, fontweight='bold', color=Config.COLOR_PALETTE['text'], pad=20)
        plt.xlabel('Date', fontsize=12, color=Config.COLOR_PALETTE['text'])
        plt.ylabel(ylabel, fontsize=12, color=Config.COLOR_PALETTE['text'])
        
        # Format y-axis with thousands separator
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        
        if log_scale:
            ax.set_yscale('log')
            plt.title(f"{title} (Log Scale)", fontsize=16, fontweight='bold')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Custom legend
        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        for text in legend.get_texts():
            text.set_color(Config.COLOR_PALETTE['text'])
        
        # Grid and spines
        ax.grid(True, linestyle='--', alpha=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor(Config.COLOR_PALETTE['text'])
        
        plt.tight_layout()
        
        # Save with transparent background if dark mode
        output_path = os.path.join(Config.OUTPUT_DIR, output_file)
        plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight', 
                    facecolor=plt.gcf().get_facecolor(), transparent=Config.DARK_MODE)
        plt.close()
        print(f"✅ Saved plot: {output_path}")
    except Exception as e:
        print(f"❌ Error generating plot '{title}': {str(e)}")

def generate_vaccination_pie_charts(df):
    """Enhanced pie charts with better styling and annotations"""
    try:
        latest_data = df[df['date'] == df['date'].max()]
        
        for country in Config.COUNTRIES:
            country_data = latest_data[latest_data['location'] == country]
            if not country_data.empty and 'vaccination_rate' in country_data.columns:
                vaccination_rate = country_data['vaccination_rate'].values[0] * 100
                labels = ['Vaccinated', 'Not Vaccinated']
                sizes = [vaccination_rate, 100 - vaccination_rate]
                colors = [Config.COLOR_PALETTE['vaccinations'], '#E0E0E0']
                
                # Create figure with constrained layout
                fig, ax = plt.subplots(figsize=(8, 8), facecolor=Config.COLOR_PALETTE['background'])
                
                # Create pie chart with enhanced properties
                wedges, texts, autotexts = ax.pie(
                    sizes, 
                    labels=labels, 
                    autopct=lambda p: f'{p:.1f}%\n({int(p/100 * country_data["population"].values[0]/1e6):.1f}M)',
                    startangle=90,
                    colors=colors,
                    wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
                    textprops={'color': Config.COLOR_PALETTE['text'], 'fontsize': 10}
                )
                
                # Set title with more information
                date_str = latest_data['date'].dt.strftime("%Y-%m-%d").values[0]
                title = f'Vaccination Coverage in {country}\n({date_str})\n'
                title += f"Population: {int(country_data['population'].values[0]/1e6):.1f}M"
                ax.set_title(title, fontsize=12, fontweight='bold', 
                            color=Config.COLOR_PALETTE['text'], pad=20)
                
                # Equal aspect ratio ensures pie is drawn as a circle
                ax.axis('equal')  
                
                # Add a circle at the center to make it a donut chart
                centre_circle = plt.Circle((0,0),0.70,fc=Config.COLOR_PALETTE['background'])
                ax.add_artist(centre_circle)
                
                # Add annotation in the center
                ax.text(0, 0, f"{vaccination_rate:.1f}%", 
                       ha='center', va='center', fontsize=24, 
                       fontweight='bold', color=Config.COLOR_PALETTE['vaccinations'])
                
                output_path = os.path.join(Config.OUTPUT_DIR, f'{country}_vaccination_pie_chart.png')
                plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight',
                            facecolor=fig.get_facecolor(), transparent=Config.DARK_MODE)
                plt.close()
                print(f"✅ Saved pie chart: {output_path}")
    except Exception as e:
        print(f"❌ Error generating vaccination pie charts: {str(e)}")

def generate_geopandas_choropleth(df):
    """Enhanced static choropleth map with better styling"""
    try:
        try:
            world = gpd.read_file('ne_110m_admin_0_countries.shp')
            print("✅ Loaded world map data from local file")
        except:
            world = gpd.read_file('https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip')
            print("✅ Loaded world map data from online source")
        
        world = world.rename(columns={'ISO_A3': 'iso_code'})
        world = world.to_crs('EPSG:3395')
        latest_data = df[df['date'] == df['date'].max()]
        
        merged = world.merge(latest_data, on='iso_code', how='left')
        
        # Create figure with constrained layout
        fig, ax = plt.subplots(1, 1, figsize=(24, 14), facecolor=Config.COLOR_PALETTE['background'])
        
        # Plot the choropleth with improved styling
        merged.plot(
            column='total_cases',
            cmap='YlOrRd',
            linewidth=0.8,
            ax=ax,
            edgecolor='#444444',
            legend=True,
            legend_kwds={
                'label': "Total COVID-19 Cases",
                'orientation': "horizontal",
                'shrink': 0.5,
                'pad': 0.02,
                'aspect': 40,
                'format': lambda x, pos: f"{int(x/1e6):,}M"
            },
            missing_kwds={
                'color': 'lightgrey',
                'edgecolor': 'white',
                'hatch': '///',
                'label': 'No data'
            }
        )
        
        # Set title with styling
        ax.set_title('🌍 Global COVID-19 Cases Distribution', 
                    fontsize=24, pad=30, weight='bold',
                    color=Config.COLOR_PALETTE['text'])
        ax.set_axis_off()
        
        # Add custom annotations for our countries of interest
        for country in Config.COUNTRIES:
            country_data = latest_data[latest_data['location'] == country]
            if not country_data.empty:
                iso_code = country_data['iso_code'].values[0]
                country_geo = merged[merged['iso_code'] == iso_code]
                if not country_geo.empty and not country_geo.geometry.isna().any():
                    centroid = country_geo.geometry.centroid.iloc[0]
                    ax.annotate(
                        text=f"{country}\n{int(country_data['total_cases'].values[0]/1e6):.1f}M",
                        xy=(centroid.x, centroid.y),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha='center',
                        va='bottom',
                        fontsize=10,
                        color='black',
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor='white',
                            edgecolor='black',
                            alpha=0.8
                        ),
                        arrowprops=dict(
                            arrowstyle="->",
                            connectionstyle="arc3,rad=0.1",
                            color='black'
                        )
                    )
        
        # Add data source and timestamp
        fig.text(
            0.5, 0.01,
            f"Data source: Our World in Data | Generated on {datetime.now().strftime('%Y-%m-%d')}",
            ha='center',
            fontsize=10,
            color=Config.COLOR_PALETTE['text']
        )
        
        output_path = os.path.join(Config.OUTPUT_DIR, 'global_cases_geopandas.png')
        plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), transparent=Config.DARK_MODE)
        plt.close()
        print(f"✅ Saved enhanced GeoPandas choropleth: {output_path}")
        
    except Exception as e:
        print(f"❌ Error generating GeoPandas choropleth: {str(e)}")

def generate_plotly_choropleth(df):
    """Enhanced interactive choropleth with Plotly"""
    try:
        # Prepare data
        latest_global = df[['iso_code', 'location', 'date', 'total_cases', 'total_deaths', 'total_vaccinations']].copy()
        latest_date = latest_global['date'].max()
        latest_global = latest_global[latest_global['date'] == latest_date]
        
        # Create enhanced choropleth map
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
            title=f'<b>Global COVID-19 Cases as of {latest_date.strftime("%B %d, %Y")}</b>',
            color_continuous_scale='OrRd',
            projection='natural earth',
            labels={'total_cases': 'Total Cases'},
            range_color=[0, latest_global['total_cases'].quantile(0.9)],
            height=700
        )
        
        # Customize hover template
        fig.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>" +
                         "<b>Total Cases:</b> %{customdata[0]:,}<br>" +
                         "<b>Total Deaths:</b> %{customdata[1]:,}<br>" +
                         "<b>Total Vaccinations:</b> %{customdata[2]:,}<extra></extra>"
        )
        
        # Update layout with professional styling
        fig.update_layout(
            margin={"r":0,"t":80,"l":0,"b":0},
            coloraxis_colorbar={
                'title': 'Total Cases',
                'thickness': 20,
                'len': 0.5,
                'tickprefix': '',
                'ticksuffix': '',
                'tickformat': ',.0f'
            },
            title_font=dict(size=24, family=Config.FONT),
            font=dict(family=Config.FONT),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            annotations=[
                dict(
                    x=0.5,
                    y=-0.1,
                    xref='paper',
                    yref='paper',
                    text='Source: Our World in Data | Created with Plotly',
                    showarrow=False,
                    font=dict(size=10)
        )]
        )
        
        # Save as HTML with additional config
        output_path = os.path.join(Config.OUTPUT_DIR, 'global_cases_plotly.html')
        fig.write_html(
            output_path,
            include_plotlyjs='cdn',
            config={
                'displayModeBar': True,
                'scrollZoom': True,
                'responsive': True
            }
        )
        print(f"✅ Saved Plotly choropleth: {output_path}")
        
    except Exception as e:
        print(f"❌ Error generating Plotly choropleth: {str(e)}")

def generate_comparison_bar_charts(df):
    """Generate comparison bar charts for key metrics"""
    try:
        latest_data = df[df['date'] == df['date'].max()]
        
        metrics = [
            ('total_cases', 'Total Cases', 'cases'),
            ('total_deaths', 'Total Deaths', 'deaths'),
            ('total_vaccinations', 'Total Vaccinations', 'vaccinations')
        ]
        
        for metric, title, color_key in metrics:
            if metric in latest_data.columns:
                plt.figure(figsize=Config.PLOT_SIZE, facecolor=Config.COLOR_PALETTE['background'])
                
                # Sort data for better visualization
                sorted_data = latest_data.sort_values(metric, ascending=False)
                
                # Create bar plot with enhanced styling
                bars = plt.bar(
                    sorted_data['location'],
                    sorted_data[metric],
                    color=Config.COLOR_PALETTE[color_key],
                    edgecolor='white',
                    linewidth=1
                )
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width()/2.,
                        height,
                        f'{int(height/1e6):.1f}M' if height > 1e6 else f'{int(height):,}',
                        ha='center',
                        va='bottom',
                        fontsize=10,
                        color=Config.COLOR_PALETTE['text']
                    )
                
                # Formatting
                plt.title(f'{title} by Country', 
                         fontsize=16, fontweight='bold', 
                         color=Config.COLOR_PALETTE['text'], pad=20)
                plt.xlabel('Country', fontsize=12, color=Config.COLOR_PALETTE['text'])
                plt.ylabel(title, fontsize=12, color=Config.COLOR_PALETTE['text'])
                
                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45, ha='right')
                
                # Format y-axis with thousands separator
                ax = plt.gca()
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
                
                # Grid and spines
                ax.grid(True, linestyle='--', alpha=0.7, axis='y')
                for spine in ax.spines.values():
                    spine.set_edgecolor(Config.COLOR_PALETTE['text'])
                
                plt.tight_layout()
                
                output_path = os.path.join(Config.OUTPUT_DIR, f'{metric}_comparison.png')
                plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight',
                            facecolor=plt.gcf().get_facecolor(), transparent=Config.DARK_MODE)
                plt.close()
                print(f"✅ Saved comparison chart: {output_path}")
    except Exception as e:
        print(f"❌ Error generating comparison charts: {str(e)}")

def generate_report(df):
    """Generate professional PDF report with analysis and visualizations"""
    from matplotlib.backends.backend_pdf import PdfPages

    if df is None:
        return

    try:
        latest_date = df['date'].max()
        report_path = os.path.join(Config.OUTPUT_DIR, 'covid19_analysis_report.pdf')

        with PdfPages(report_path) as pdf:
            _extracted_from_generate_report_14(latest_date, pdf, df)
        print(f"✅ Saved professional report: {report_path}")

    except Exception as e:
        print(f"❌ Failed to generate report: {str(e)}")


# TODO Rename this here and in `generate_report`
def _extracted_from_generate_report_14(latest_date, pdf, df):
    # Title page
    plt.figure(figsize=(11, 8.5), facecolor=Config.COLOR_PALETTE['background'])
    plt.text(0.5, 0.7, 'COVID-19 Data Analysis Report', 
             ha='center', va='center', fontsize=24, fontweight='bold',
             color=Config.COLOR_PALETTE['text'])
    plt.text(0.5, 0.6, f'Analysis Date: {datetime.now().strftime("%B %d, %Y")}', 
             ha='center', va='center', fontsize=16,
             color=Config.COLOR_PALETTE['text'])
    plt.text(0.5, 0.5, f'Data Updated: {latest_date.strftime("%B %d, %Y")}', 
             ha='center', va='center', fontsize=16,
             color=Config.COLOR_PALETTE['text'])
    plt.text(0.5, 0.3, 'Countries Included:', 
             ha='center', va='center', fontsize=14,
             color=Config.COLOR_PALETTE['text'])

    # List countries in two columns
    country_list = "\n".join([f"• {country}" for country in Config.COUNTRIES[:4]])
    plt.text(0.35, 0.2, country_list, 
             ha='left', va='center', fontsize=12,
             color=Config.COLOR_PALETTE['text'])

    country_list = "\n".join([f"• {country}" for country in Config.COUNTRIES[4:]])
    plt.text(0.65, 0.2, country_list, 
             ha='left', va='center', fontsize=12,
             color=Config.COLOR_PALETTE['text'])

    _extracted_from_generate_report_39(pdf, 'Summary Statistics')
    # Create summary table
    latest_data = df[df['date'] == latest_date]
    summary_df = latest_data[['location', 'total_cases', 'total_deaths', 'total_vaccinations']]
    summary_df = summary_df.set_index('location')
    summary_df.columns = ['Total Cases', 'Total Deaths', 'Total Vaccinations']

    # Format numbers in the table
    formatted_df = summary_df.map(lambda x: f"{int(x):,}" if pd.notnull(x) else "N/A")

    # Add table to report
    plt.table(
        cellText=formatted_df.values,
        rowLabels=formatted_df.index,
        colLabels=formatted_df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0.1, 0.1, 0.8, 0.7]
    )

    plt.text(0.1, 0.05, 'Key Metrics as of ' + latest_date.strftime("%B %d, %Y"), 
             fontsize=12, color=Config.COLOR_PALETTE['text'])

    _extracted_from_generate_report_39(pdf, 'Key Insights')
    insights = [
        "1. The United States consistently shows the highest total cases and vaccination numbers",
        "2. Developing nations like Kenya and Rwanda show slower vaccination rollouts",
        "3. Brazil and India experienced significant waves of infections at different times",
        "4. Death rates vary significantly between countries, suggesting differences in healthcare capacity",
        "5. Vaccination rates in developed nations (US, Germany) outpace developing nations",
        "6. Case trajectories show different pandemic waves across countries",
        "7. Vaccination campaigns began at different times with varying effectiveness",
        "8. Some countries show better case-to-death ratios indicating healthcare system strength"
    ]

    for i, insight in enumerate(insights):
        plt.text(0.1, 0.8 - i*0.08, insight, 
                 fontsize=12, ha='left', va='center',
                 color=Config.COLOR_PALETTE['text'])

    _extracted_from_generate_report_114(pdf)
    # Add visualizations to the report
    visualization_files = [
        'total_cases.png',
        'total_deaths.png',
        'global_cases_geopandas.png'
    ]

    for vis_file in visualization_files:
        vis_path = os.path.join(Config.OUTPUT_DIR, vis_file)
        if os.path.exists(vis_path):
            img = plt.imread(vis_path)
            plt.figure(figsize=(11, 8.5))
            plt.imshow(img)
            _extracted_from_generate_report_114(pdf)
    # Final page with data source
    plt.figure(figsize=(11, 8.5), facecolor=Config.COLOR_PALETTE['background'])
    plt.text(0.5, 0.7, 'Data Sources', 
             ha='center', va='center', fontsize=20, fontweight='bold',
             color=Config.COLOR_PALETTE['text'])
    plt.text(0.5, 0.6, 'Primary Data Source: Our World in Data COVID-19 Dataset', 
             ha='center', va='center', fontsize=14,
             color=Config.COLOR_PALETTE['text'])
    plt.text(0.5, 0.5, 'Geographic Data: Natural Earth', 
             ha='center', va='center', fontsize=14,
             color=Config.COLOR_PALETTE['text'])
    plt.text(0.5, 0.4, f'Report generated on {datetime.now().strftime("%B %d, %Y %H:%M")}', 
             ha='center', va='center', fontsize=12,
             color=Config.COLOR_PALETTE['text'])
    _extracted_from_generate_report_114(pdf)


# TODO Rename this here and in `generate_report`
def _extracted_from_generate_report_39(pdf, arg1):
    _extracted_from_generate_report_114(pdf)
    # Summary statistics page
    plt.figure(figsize=(11, 8.5), facecolor=Config.COLOR_PALETTE['background'])
    plt.text(
        0.1,
        0.9,
        arg1,
        fontsize=20,
        fontweight='bold',
        color=Config.COLOR_PALETTE['text'],
    )


# TODO Rename this here and in `generate_report`
def _extracted_from_generate_report_114(pdf):
    plt.axis('off')
    pdf.savefig(bbox_inches='tight')
    plt.close()

# Main execution
if __name__ == "__main__":
    print("\nStarting Enhanced COVID-19 Data Analysis...\n")
    
    # Load and process data
    df = load_data()
    df_clean = clean_data(df)
    
    if df_clean is not None:
        # Generate enhanced visualizations
        plot_time_series(df_clean, 'total_cases', 'Total COVID-19 Cases Over Time', 'Total Cases', 'total_cases.png')
        plot_time_series(df_clean, 'total_deaths', 'Total COVID-19 Deaths Over Time', 'Total Deaths', 'total_deaths.png')
        plot_time_series(df_clean, 'new_cases_7day_avg', '7-Day Average of New COVID-19 Cases', 'New Cases (7-day avg)', 'new_cases_7day_avg.png')
        plot_time_series(df_clean, 'death_rate', 'COVID-19 Case Fatality Rate Over Time', 'Death Rate', 'death_rate.png')
        
        if 'total_vaccinations' in df_clean.columns:
            plot_time_series(df_clean, 'total_vaccinations', 'Total Vaccinations Over Time', 'Vaccinations', 'vaccinations.png')
            generate_vaccination_pie_charts(df_clean)
        
        # Generate comparison charts
        generate_comparison_bar_charts(df_clean)
        
        # Generate both types of choropleth maps
        generate_geopandas_choropleth(df_clean)
        generate_plotly_choropleth(df_clean)
        
        # Generate professional PDF report
        generate_report(df_clean)
    
    print("\nAnalysis complete! \nAll outputs saved to the 'output' directory.")