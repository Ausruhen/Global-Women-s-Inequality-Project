import numpy as np
import pandas as pd 

#Graphing
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

import warnings
import streamlit as st
warnings.filterwarnings('ignore')
color_pal = sns.color_palette("bright")

st.title("Streamlit Personal Project Conversion")

#Data loading 
@st.cache_data
def load_data():
    #Gdp by year dataframe and transformations
    gdp_by_year_func = pd.read_csv("Data/GDP_by_year.csv")    
    gdp_by_year_func = gdp_by_year_func.drop(columns =["Indicator Name", "Indicator Code"])
    
    #Code region income dataframe and Transformations
    code_region_income_func = pd.read_csv("Data/codes_region_income.csv")
    code_region_income_func = code_region_income_func.drop(columns = ["Unnamed: 5", "SpecialNotes", "TableName"])

    #Gii dataframe and Transformations  
    gii_func = pd.read_csv("Data/Gender Inequality Index.csv")
    gii_func = gii_func.drop(columns = ["indexCode", "index","dimension", "indicatorCode", "note"])
    gii_func = gii_func[gii_func["countryIsoCode"].apply(len) == 3]

    #population dataframes, used for calculating gdp per capita
    pop_func = pd.read_csv("data/population_world_un.csv",header=16)
    
    return code_region_income_func, gdp_by_year_func, gii_func, pop_func
    
code_region_income, gdp_by_year, gii, pop = load_data()

st.markdown("""
## Obtaining the Data Explanation
This data was found in the World Bank Group website (WBG). They are an organization that provides knowledge and info to fight "poverty and build shared prosperity in developing countries". Linked below at the end of the cell will be both their website and the page where you can download the raw data. Once downloading the raw data we get a zip file that contains three csv files. Of the two csv's of relevance one contains the GDP of nations from 1960-2023 and the other contains WBG classifications of Region, IncomeGroup and special notes about the country. For our purposes I renamed the second csv "code_region_income", with the other one being renamed "GDP_by_year".  I added one more dataset from the united nations to look at gender inequality with the goal being to see if I can use GDP to predict various statistics about gender inequality. Lastly I added the third and final dataset of population statistics around the world to use gdp_per capita as a predictor in place of GDP. 

As of 10/15/24 the only data sets used are from World Bank Group: [Organization](https://data.imf.org/?sk=4c514d48-b6ba-49ed-8ab9-52b0c1a0179b&sid=1409151240976), [Data](https://data.worldbank.org/indicator/NY.GDP.MKTP.CD)

As of 11/9/24 We've added another data set being the UN GII index found here: [Organization](https://hdr.undp.org/data-center/thematic-composite-indices/gender-inequality-index#/indicies/GII%5D), [Data](https://hdr.undp.org/data-center/thematic-composite-indices/gender-inequality-index#/indicies/GII%5D)

As of 11/24/24 Were adding the third and final set for population statistics around the world (From the UN): [Data](https://population.un.org/wpp/Download/Standard/MostUsed/)
""")

st.text("Dataframe of countries Code, Region, and Income Level")
st.dataframe(code_region_income)

st.text("Dataframe with countries GDP by year and Regional GDP (1960-2023)")
st.dataframe(gdp_by_year)

st.markdown("""
## Cleaning the data/Making it Tidy
Lets first load in our data and then see what might need to change. As a note before we do any pandas operations, I edited the original csv file gdp_by_year to delete the first 5 rows because it did not contain anything useful to this project. By deleting the first 5 rows, the csv file looks like a normal csv file with headers that represent each column.
""")

st.code("""code_region_income = pd.read_csv("Data/codes_region_income.csv")
gdp_by_year = pd.read_csv("Data/GDP_by_year.csv")

special_notes = code_region_income["SpecialNotes"]
code_region_income = code_region_income.drop(columns = ["Unnamed: 5", "SpecialNotes", "TableName"])
code_region_income.head(5)""")

st.dataframe(code_region_income.head(5))

st.code("""#Lets drop some unncessary columns, Was unncessary because all values weren't unique
gdp_by_year = gdp_by_year.drop(columns =["Indicator Name", "Indicator Code"])
gdp_by_year.head()""")

st.dataframe(gdp_by_year.head())

st.text("""Next step should be to convert this from "wide" data into "long" data similar to how we originally did this in the slides in the third lecture, and then to make sure that each data type for each column makes sense as to what it should be. In this step I am willingly choosing to leave the Null values as is because I definitely don't want to replace them with 0's (going to mess up averages), and they are fine as is.""")

st.code("""gdp_long = gdp_by_year.melt(id_vars = ["Country Name", "Country Code"],
                            var_name = "year", 
                            value_vars = gdp_by_year.columns[2:gdp_by_year.shape[1]],
                            value_name = "GDP")""")

@st.cache_data
def melting_and_sorting():   #Not containing the merges/groupby later because we need gdp_long_sorted for future function calls 
    gdp_long_func = gdp_by_year.melt(id_vars = ["Country Name", "Country Code"],
                                var_name = "year", 
                                value_vars = gdp_by_year.columns[2:gdp_by_year.shape[1]],
                                value_name = "GDP")
    gdp_long_func["year"] = gdp_long_func["year"].astype('int')
    gdp_long_sorted_func = gdp_long_func.sort_values(["Country Name", 'year'])
    return gdp_long_sorted_func #sorted by name and year

gdp_long_sorted = melting_and_sorting()

st.dataframe(gdp_long_sorted)


st.markdown("""## Basic Lineplot Section
Now that we have the gdp_long formatted correctly lets, join in the regions from the other table and do some plotting! 
As a note for this streamlit conversion, we will now be deviating from the original project as creating these interactive plots
covers all the lineplots previously made in the original project. If you want to see the original project plots and interpretations
refer to the original project. 
""")

@st.cache_data
def merging_and_grouping(metric): 
    joined_code_income_func = gdp_long_sorted.merge(code_region_income, on = "Country Code", how = "left")
    sum_gdp_per_region_func = joined_code_income_func.groupby(['Region','year'])['GDP'].agg(metric).reset_index()
    return sum_gdp_per_region_func

sum_gdp_per_region = merging_and_grouping('sum')

#Graph options for line_plotting section
region_checks = st.multiselect("Please Pick Regions (Default is everything): ", sum_gdp_per_region.Region.unique()) #returns an empty list if not checked
year_double_slider = st.slider('Select a range of values', 1960, 2023, (1960, 2022), step = 1) #returns a list/tuple can index with [0] and [1] for selected values
metric_select = st.selectbox("Please pick a metric (Default is Sum): ", ["sum", "mean", "median", "max", "min"])


if metric_select == None:
    metric_select = "mean"

@st.cache_data
def line_plotting_section(region_checks_func, year_double_slider_func, metric):   
    if len(region_checks_func) == 0:
        plt.figure(figsize = (9,7))
        metric_gdp_per_region = merging_and_grouping(metric)
        metric_checked_years = metric_gdp_per_region[(metric_gdp_per_region["year"] >= year_double_slider_func[0]) & (metric_gdp_per_region["year"] <= year_double_slider_func[1])]
        plot1 = sns.lineplot(data = metric_checked_years, x = "year", y = "GDP", hue = "Region", palette = color_pal)
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5)) #going to completely honest I used AI assistance to do this. It makes 5 ticks along the axis
        plt.xlabel("Year")
        plt.title(f"{metric} of Regional Gdps Per Year")
        plt.savefig(f"{metric} of Regional Gdps Per Year", dpi=300, bbox_inches='tight')
        st.pyplot(plot1.get_figure())
    else:
        metric_gdp_per_region = merging_and_grouping(metric)
        metric_checked = metric_gdp_per_region[metric_gdp_per_region["Region"].isin(region_checks_func)]
        metric_checked_years = metric_checked[(metric_checked["year"] >= year_double_slider_func[0]) & (metric_checked["year"] <= year_double_slider_func[1])]
        plt.figure(figsize = (9,7))
        plot1 = sns.lineplot(data = metric_checked_years, x = "year", y = "GDP", hue = "Region", palette = color_pal)
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5)) #going to completely honest I used AI assistance to do this. It makes 5 ticks along the x axis
        plt.xlabel("Year")
        plt.title(f"{metric} of Regional Gdps Per Year")
        plt.savefig(f"{metric} of Regional Gdps Per Year", dpi=300, bbox_inches='tight')
        st.pyplot(plot1.get_figure())


line_plotting_section(region_checks, year_double_slider,metric_select)


st.markdown("""## Adding The Second/Third Data Set
Now lets join in the Gender Inequality Index provided by the UN. The GII attempts to measure gender Inequality by a number of factors, with the ones im going to be focusing in on being adolescent birth rate, female labor force%, maternal mortality rates, % of women with secondary education, and % of women in parliament. The goal is see how well GDP can relate with Gender Inequality, eventually doing the predictions/inference section using GDP to predict factors of the GII. But lets read this in, format it, and check the data types of what were working with.

Note for streamlit viewers we're going to display the cleaned dataframe thats ready for use where we format GII index values to be in columns along with year, country code, and gdp and gdp per capita
""")

indicators = np.unique(gii['indicator'])
adol_gii = gii[gii['indicator'] == indicators[0]]
lfpf = gii[gii['indicator'] == indicators[3]]
maternal = gii[gii['indicator'] == indicators[5]]
second_f = gii[gii['indicator'] == indicators[6]]
parla = gii[gii['indicator'] == indicators[8]]

@st.cache_data
def all_score_gdp_frame():
    gdp_long_sorted_func = gdp_long_sorted.rename(columns = {"Country Code": "countryIsoCode"})
    adol_gii_func = adol_gii.rename(columns = {"value": "adol_value"})
    lfpf_func = lfpf.rename(columns = {"value": "lf female%"})
    maternal_func = maternal.rename(columns = {"value": "maternal mortality"})
    second_f_func = second_f.rename(columns = {"value": "second education female"})
    parla_func = parla.rename(columns = {"value": "female parliament%"})
    
    all_score = adol_gii_func.merge(lfpf_func, how = 'left', on = ["countryIsoCode", "year"]).merge(maternal_func, how = 'left', on = ["countryIsoCode", "year"])
    all_score = all_score[["countryIsoCode", "year", "adol_value", "lf female%", "maternal mortality"]].merge(second_f_func, how = 'left', on = ["countryIsoCode", "year"]).merge(parla_func, how = 'left', on = ["countryIsoCode", "year"])
    all_score = all_score[["countryIsoCode", "year", "adol_value", "lf female%", "maternal mortality", "second education female", "female parliament%"]]
    
    all_score_gdp = all_score.merge(gdp_long_sorted_func, how = 'left', on = ["countryIsoCode", "year"])
    all_score_gdp = all_score_gdp[["countryIsoCode", "year", "adol_value", "lf female%", "maternal mortality", "second education female", "female parliament%", "GDP"]]
    
    countries = pop[pop["Type"] == "Country/Area"]
    countries = countries[["ISO3 Alpha-code", "Year", "Total Population, as of 1 January (thousands)", "Male Population, as of 1 July (thousands)", "Female Population, as of 1 July (thousands)"]]
    countries = countries.rename(columns = {"ISO3 Alpha-code": "countryIsoCode", "Year":"year"})
    countries["year"] = countries["year"].astype('int')
    
    for column in countries.columns[2:]:
        countries[column] = countries[column].str.replace(" ", "")
        countries[column] = countries[column].astype(int) * 1000
    
    countries.columns = countries.columns.str.lower().str.replace(r',.*$', "", regex = True)
    
    all_score_gdp.columns = all_score_gdp.columns.str.lower()
    
    all_score_gdp_pop = all_score_gdp.merge(countries, how = 'left', on = ["countryisocode", "year"])
    all_score_gdp_pop["gdp_capita"] = all_score_gdp_pop["gdp"]/all_score_gdp_pop["total population"] 
    
    return all_score_gdp_pop

all_score_gdp_pop = all_score_gdp_frame()

st.dataframe(all_score_gdp_pop)

st.markdown("""
## Plotting Second Data set
Lets start with some boxplots to get a general sense of distributions before we move on to a heatmap of the correlation coefficients, with a pairplot to finish things off! 
""")

frames = [adol_gii, lfpf, maternal, second_f, parla]
gii_options = ["Adolescent Birth Rates", "Female Labor Force %", "Maternal Mortality (Deaths per 100,000 births)", "Females With Secondary Education % (Age >25)", "Females in Parliament %"]

gii_silection = st.selectbox("Please Pick a GII Parameter", gii_options)

@st.cache_data
def boxplots():
    selected_index = gii_options.index(gii_silection)
    selected_frame = frames[selected_index]
    plot2 = plt.figure(figsize= (15,8))
    sns.boxplot(data = selected_frame, x = "value", color = sns.color_palette('pastel')[selected_index])
    plt.xlabel(f"{selected_frame.indicator.iloc[0]}")
    return plot2 
plot2 = boxplots()    
st.pyplot(plot2)

capita_vals = all_score_gdp_pop[all_score_gdp_pop.columns[2:]].drop(columns = ["total population", "male population", "female population"]) 

@st.cache_data
def compute_heatmap():
    capita_corr = capita_vals.corr()  
    
    # Create figure before plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data=capita_corr, annot=True, cmap='mako', ax=ax)
    ax.set_title("Heatmap of R-values between GII Index values and GDP")

    return fig  # ✅ Return the figure, not the plot

@st.cache_data
def compute_pairplot():
    recent = all_score_gdp_pop[all_score_gdp_pop["year"] == 2020]
    recent = recent[["adol_value", "lf female%", "maternal mortality", "second education female", "female parliament%", "gdp",'gdp_capita']]
    
    plot = sns.pairplot(data=recent, corner=True)
    
    return plot.fig  # ✅ Return the figure associated with pairplot

if "heatmap_fig" not in st.session_state:
    st.session_state.heatmap_fig = compute_heatmap()

if "pairplot_fig" not in st.session_state:
    st.session_state.pairplot_fig = compute_pairplot()


st.pyplot(st.session_state.heatmap_fig)
st.pyplot(st.session_state.pairplot_fig)

test_select = st.selectbox("this is just testing: ", ["1", "2", "3"])

st.markdown("""
    # Inference Section

So for this inference section heres my question: Does gdp affect GII values uniformly across regions? This question came to me when discussing my project with my friends and I was suprised by the differences in responses to what we expected gdp to do across regions. We unaniousmly agreed that higher gdp meant better statistics but a couple said that it was going to be roughly the same effect across regions while some of us (including me) thought that it was going to be significantly different. So this leads me here so lets see if we can answer this question!
"""
           )



















