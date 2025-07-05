
from altair import Data
from plotly.graph_objs import Font
import streamlit as st
import pandas as pd
import plotly.express as px

from numpy import tile

# Loading Data
@st.cache_data
def load_data():
    df = pd.read_csv('Data/WHRA-cleaned_2024.csv')
    return df

st.set_page_config(

    page_title="World Happiness Report Analysis",
    layout="wide",
    initial_sidebar_state="expanded"

)


st.markdown(" <div style='font-size:70px; font-weight: bold; text-align: center;'> WORLD&#x1F30D; HAPPINESS REPORT ANALYSIS &#x1F60A;</div>", unsafe_allow_html=True)
st.markdown(" <div style='font-size:40px; text-align: center;'> Chinmay Mishra </div>", unsafe_allow_html=True)

st.caption("LinkedIn: https://www.linkedin.com/in/chinmay-mishra-dev/ ")

st.markdown("---")

frame = load_data()

# Developing sidebar.

st.sidebar.info("""Collapse the sidebar while viewing graphs.""")


st.sidebar.header("Filters for Bar Chart")
selected_year_bar = st.sidebar.selectbox(
    
    "Select a Year",
    options = sorted(frame['year'].unique(), reverse=True)

)

st.sidebar.markdown('---')

st.sidebar.header("Filters for Heat Map")
selected_country_heatMap = st.sidebar.selectbox(

    "Select a Country",
    options = sorted(frame['Country name'].unique())

)



# Data Manipulation for Graphs and Tables

DataForBar = frame[frame['year'] == selected_year_bar]
DataForHeatMap = frame[frame['Country name'] == selected_country_heatMap]



# Plots and Figures

def DataTable():

    st.header(':orange[ORIGINAL DATA REPORT TABLE]')
    st.markdown(f"""
    
        The data below is a recorded set of :green[ladder scores] (or happiness scores) along with other important metrics that define the movement of the happiness index across the general population of a country.
        Many of these metrics show strong correlation, which we shall be exploring shortly.
        Surprisingly, based on my findings, the most commonly assumed metric for deriving a general sense of a strong emotion like happiness — within social hierarchical environments — turned out not to be the general sense of generosity among a country's population.
        Instead, GDP and related economic factors appeared to drive most countries' happiness metrics, influencing the shared emotional state of their populations more significantly than acts of generosity.
        
        Hence, the goal of this analysis - naturally turned towards the exploration of such factors and qestions attached to it. 
        
        <ul>
            <li>"How closely are two factors, one emotion, and other being a statistical terminology previously assumed unrelated are actually closely related ?"</li>
            <li>"Does this mean that economic factors are also closely related to defining the movement of an emotional metric like happiness amoung the general population ?"</li>
            <li>"If so, then is GDP a singular factor affecting the movement of the happiness index and other factors involved?"</li>
        </ul>

        Therefore, lets explore these questions together in detail. :smile:
    
    """, unsafe_allow_html=True)
    st.caption("Data Source: https://www.kaggle.com/datasets/jainaru/world-happiness-report-2024-yearly-updated/data")
    st.dataframe(frame, use_container_width=True)

def top10bar():

    top10 = DataForBar.sort_values(by="Life Ladder", ascending = False).head(10) #type:ignore
    fig = px.bar(

        top10,
        x="Life Ladder",
        y="Country name",
        orientation="h",
        color="Life Ladder",
        color_continuous_scale="viridis",
        height=600

    )

    fig.update_layout(yaxis=dict(autorange='reversed'))

    return fig

def corr_matrix_individual():

    numeric_frame = DataForHeatMap.drop(columns='year').select_dtypes(include=['float64', 'int64'])

    correlation_matrix = numeric_frame.corr()

    fig = px.imshow(
        correlation_matrix,
        text_auto='.2f', #type: ignore
        color_continuous_scale="RdBu_r",
        title=f"Country: {selected_country_heatMap}",
        width=1000,
        height=1000

    )

    fig.update_layout(
        font = dict(
            size=16,
        ),
        xaxis = dict(
            tickfont = dict(
                size = 14,
            ),
        ),
        yaxis = dict(
            tickfont = dict(
                size = 14,
            ),
        ),
        title = {

            "x" : 0.5,
            "xanchor" : "center"

        },
        title_font_size = 25,
    )

    return fig

def corr_matrix_global():

    numeric_frame = frame.drop(columns='year').select_dtypes(include=['float64', 'int64'])

    correlation_matrix = numeric_frame.corr()

    fig = px.imshow(
        correlation_matrix,
        text_auto='.2f', #type: ignore
        color_continuous_scale="RdBu_r",
        title=f"Global",
        width=1000,
        height=1000

    )

    fig.update_layout(
        font = dict(
            size=16,
        ),
        xaxis = dict(
            tickfont = dict(
                size = 14,
            ),
        ),
        yaxis = dict(
            tickfont = dict(
                size = 14,
            ),
        ),
        title = {

            "x" : 0.5,
            "xanchor" : "center"

        },
        title_font_size = 25,
    )

    return fig

def bellCurve(frame):
    
    selected_factor = st.selectbox("Select a Factor", frame.columns)

    # Gloabal Stats
    mean = frame[selected_factor].mean()
    std = frame[selected_factor].std()

    



DataTable()

st.markdown("---")
st.header(f":orange[TOP 10 HAPPIEST COUNTIRES IN {selected_year_bar}]")

st.markdown(f"""

    To understand the global happiness index, we must first identify the countries that consistently rank at the top.
    This not only provides a clear starting point for our analysis, but also helps us isolate the strong factors contributing to the high happiness index values in these nations.
    Once we have successfully separated and understood these positive drivers among the happiest countries, we can establish a strong baseline for comparison.
    Using this baseline, we can then evaluate countries with lower scores, identifying the specific areas where improvements could significantly enhance their general happiness index.

""")

st.info("Plotting Top 10 Countries helps segrigate these countries to be baseline while comparing to other countries perfomring low on happiness score.")

col1 = st.columns(1)[0]
with col1:
    st.plotly_chart(top10bar(), use_container_width=True)

st.markdown('---')

st.header(':orange[CORRELATIONAL ANALYSIS]')
st.markdown(f"""

    The correlational analysis presented below offers a comparative view between the selected country (Canada) and the global perspective. Each heatmap captures the strength and direction of relationships between key factors influencing the overall happiness index.
    On the left, the country-specific heatmap focuses on Country's internal dynamics. A :red[positive correlation (red shades)] indicates that as one factor increases, so does the other, while a :blue[negative correlation (blue shades)] implies an inverse relationship. For example, in Canada, Positive Affect shows a strong positive correlation with :green[Life Ladder (Happiness)], highlighting the role of emotional well-being in national happiness. Similarly, Log GDP per capita and Social Support are closely tied to happiness, affirming the importance of economic stability and social safety nets.
    On the right, the global heatmap provides a broader lens. Globally, similar patterns emerge — GDP, Social Support, and Healthy Life Expectancy are consistently strong contributors to higher happiness scores. However, subtle variations between the country-specific and global maps reveal the unique social, economic, and emotional fabric of each nation.
    Through this side-by-side analysis, it becomes evident how internal country dynamics compare against the global averages, offering deeper insights into what factors drive happiness locally and universally.

""")

st.info("""
    
        The idea behind comparing the two correlational heatmaps, is to identify the factors that have positive correlation, and negative. 
        After isolating these factors, we can then compare the global factors with individual factors of a given country. This can help us commpare different scenarios
        and other predictive analysis of other factors(Not present in datatable) like cultural landscape, religious differences, or in some cases geographical locations.
        These factors have high chances of influencing happiness score, or have similarly high chances on being total exceptions. Quite a paradox! Hence, the correlational
        heat maps to gather more facts.
    
""")


col2, col3 = st.columns(2)

with col2:
    st.plotly_chart(corr_matrix_individual(), use_container_width=True)

with col3:
    st.plotly_chart(corr_matrix_global(), use_container_width=True)

col4 = st.columns(1)[0]

with col4:
    
    stats_frame = frame.drop(columns = ['year']).describe().drop(index = ['count'])
    
    st.subheader("DESCRIPTIVE STATISTICS BLOCK")
    st.info("""
        
        Making a descriptive statistical table gives us a clearn, and clean vision into the big picture idea of data movements. It can help us track the
        worlds average of each factors, which helps us detect the global patterns, and flag unstable factors at the very beginning of our analysis. This gives
        deep insight into the global standard deviations, mean, minimum, first quartile, median, third quartile, and maximum for each factor or our descriptive table.
        The formula used for calculating standard deviation is 'Sample Standard Deviation', as the data represents a sample of the global population.

    """)
    st.dataframe(stats_frame)

    st.latex(r'''\text{Mean} = \frac{\sum{x_i}}{n}''')
    st.latex(r'''\text{Sample Standard Deviation} = \sqrt{ \frac{ \sum (x_i - \bar{x})^2 }{n - 1} }''')

    st.markdown("---")
    bellCurve(frame = stats_frame)


    st.subheader("STATISTICAL KEY FINDINGS")
    st.markdown(f"""
    
        We take standard deviation values which helps us achieve volatility of a factor. In our case, :green[low deviation] means the volatility range is lower meaning a global
        stable factor, :red[high deviation] means the volatility is high meaning global unstable factor. This helps in clean seggrigation of factors that should be observed,
        and others to be ignored.

        Therefore, here are the factors required for our analysis:

           - Social Support: :green[Stable ({ round( stats_frame['Social support']['std'], 4) })]
           - Freedom: :green[Stable ( { round( stats_frame['Freedom to make life choices']['std'], 4) })]
           - Generosity: :green[Stable ( {round( stats_frame['Generosity']['std'], 4) })]
           - Perceptions of curruption: :green[Stable ( { round( stats_frame['Perceptions of corruption']['std'], 4 ) } )]
           

    """) #type: ignore

    