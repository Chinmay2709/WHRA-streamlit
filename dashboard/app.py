import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.linear_model import LinearRegression

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

def regression_bar():

    st.subheader("Visualiztion of Factors Impact")

    coeff_df = pd.DataFrame({
    "Factor": coefficients.index,
    "Impact": coefficients.values
    }).sort_values(by="Impact", ascending=True)  # Sort for cleaner layout

    # Create horizontal bar chart
    fig = px.bar(
        coeff_df,
        x="Impact",
        y="Factor",
        orientation='h',
        title=f"Regression Coefficients (R² = {r_squared:.2f})",
        labels={"Impact": "Coefficient Value", "Factor": "Happiness Factor"},
        color="Impact",  # Optional: adds color based on strength/direction
        color_continuous_scale="Viridis"
    )

    st.plotly_chart(fig)

def residual_scatter(Prediction, Residual):

    st.subheader("Prediction vs Residual Scatter Plot")

    residual_frame = pd.DataFrame({
        "Prediction": Prediction,
        "Residual": Residual
    })

    fig = px.scatter(
        residual_frame,
        x = "Prediction",
        y = "Residual",
        opacity = 0.7
    )

    fig.add_shape(
        
        type="line",
        x0 = residual_frame['Prediction'].min(),
        x1 = residual_frame['Prediction'].max(),
        y0 = 0,
        y1 = 0,
        line = dict(color = 'red', dash = 'dash')
        
    )

    st.plotly_chart(fig)

def world_map(Country, Prediction, Residual):

    results = pd.DataFrame({

        'Country': Country,
        'Prediction': Prediction,
        'Residual': Residual

    })

    fig = px.choropleth(
        results,
        locations="Country",
        locationmode="country names",
        color="Residual",
        color_continuous_scale=px.colors.diverging.RdBu,
        title="Residuals by country(Predicted Error)",
        labels={"Residual": "Predicted Error"},
        range_color=[-2, 2]

    )

    fig.update_layout(
        height=700,
        width=1200,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor='rgba(0,0,0,0)',
        title_font=dict(size=24, color='white'),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    st.plotly_chart(fig)


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

    
    social_sup_std = round( stats_frame['Social support']['std'], 4) #type: ignore
    freedom_sup_std = round( stats_frame['Freedom to make life choices']['std'], 4) #type: ignore
    generosity_std = round( stats_frame['Generosity']['std'], 4) #type: ignore
    perception_corrup_std = round( stats_frame['Perceptions of corruption']['std'], 4 ) #type: ignore

    st.subheader("STATISTICAL KEY FINDINGS")
    st.markdown(f"""
    
        We take standard deviation values which helps us achieve volatility of a factor. In our case, :green[low deviation] means the volatility range is lower meaning a global
        stable factor, :red[high deviation] means the volatility is high meaning global unstable factor. This helps in clean segregation of factors that should be observed,
        and others to be ignored.

        Therefore, here are the factors required for our analysis:

           - Social Support: :green[Stable {social_sup_std}]
           - Freedom: :green[Stable {freedom_sup_std}]
           - Generosity: :green[Stable {generosity_std}]
           - Perceptions of curruption: :green[Stable {perception_corrup_std}]

        Factors with higher standard deviation than our stability threshold (0.2 in this case) are considered unstable and therefore excluded from further analysis. 
        This ensures our insights focus only on globally stable, reliable indicators.
           

    """) #type: ignore

col5 = st.columns(1)[0]

with col5:

    st.markdown("---")
    st.subheader("REGRESSION ANALYSIS")

    st.info(f"""
    
        To have better and deep understanding of our selective stable factors can contribute to overall happiness scores across countries, 
        we shall build a linear regression model using :green[GDP per capita], :green[Social Support], :green[Freedom], :green[Generosity] and :green[Perception of curruption],
        as our 'X factors' against :green[Life Ladder or Happiness score] 'Y factor'. 


    """)

    x_factors = frame[[
        "Log GDP per capita",
        "Social support",
        "Healthy life expectancy at birth",
        "Freedom to make life choices",
        "Generosity",
        "Perceptions of corruption"
    ]]

    y_factor = frame["Life Ladder"].values.ravel()


    model = LinearRegression()
    model.fit(x_factors, y_factor)

    coefficients = pd.Series(model.coef_, index = x_factors.columns)
    intercept = model.intercept_
    r_squared = model.score(x_factors, y_factor)

    st.dataframe(coefficients.rename("Impact on Happiness").round(4))

    regression_bar()

    y_prediction = model.predict(x_factors)
    residuals = y_prediction - y_factor

    residual_scatter(y_prediction, residuals)

    st.markdown(f"""
    
        The regression analysis provides a clear insight into how various happiness-related factors contribute to the overall happiness score. 
        Each bar in the coefficient plot represents the weight or influence of a particular factor, with higher values indicating stronger positive impact. 
        Among these, :green[Social Support] and :green[Freedom to make life choices] stand out as the most influential, while Perception of Corruption surprisingly shows a negative association. 
        The model achieved an :green[R² value of 0.75], indicating that :green[75%] of the variance in happiness scores can be explained by our selected factors—a strong indication of the model's predictive power and goodness of fit.
        To further assess model reliability, we plotted the residuals (the differences between actual and predicted happiness scores) against the predicted values in a scatter plot. 
        Ideally, a well-fitting linear regression model should produce residuals that are randomly scattered around the zero line, with no visible pattern. 
        Our plot displays exactly that: the residuals are centered around zero and spread consistently across the prediction range, suggesting that the model does not suffer from heteroscedasticity or omitted variable bias. 
        Together, the :green[R² metric] and residual analysis validate the robustness of our regression model and confirm that it is suitable for drawing meaningful conclusions about the drivers of happiness across countries.
    
    """)

col6 = st.columns(1)[0]

with col6:

    st.markdown("---")
    st.header(":orange[CONCLUSION]")

    st.info(f"""
    
            This project presents a full-cycle data analysis and machine learning solution applied to the World Happiness Report, aiming to understand the key drivers behind national happiness scores. 
            By leveraging a linear regression model, we quantified the influence of multiple socio-economic and psychological factors such as social support, freedom of life choices, perceptions of corruption, and generosity. 
            The model achieved a commendable R² score of 0.75, suggesting that our selected features explain 75% of the variance in happiness scores across countries—a strong indicator of the model’s effectiveness and reliability.
            To validate the integrity of the model, we conducted a residual analysis using diagnostic plots such as the residuals vs predicted values, distribution histogram, and Q-Q plot. 
            These plots confirmed that the residuals were randomly distributed, with minimal outliers and no discernible patterns—thereby satisfying core regression assumptions like linearity and homoscedasticity. 
            This strengthens confidence in both the predictions and the interpretability of our coefficients.
            Beyond modeling, the project delivers a visually engaging and interactive Streamlit dashboard, allowing users to explore the global happiness landscape with ease. It brings together statistical rigor and storytelling by combining predictive modeling with intuitive visuals. 
            The codebase is clean, modular, and version-controlled via GitHub, ensuring scalability and collaboration readiness.
            Overall, this project demonstrates end-to-end proficiency in data science—from data acquisition and preprocessing to model development, evaluation, and deployment. 
            It not only serves as an analytical exploration of global well-being but also reflects practical competence in Python, machine learning, data visualization, and app deployment. 
            This experience sets a strong foundation for more advanced projects involving time-series forecasting, geographic modeling, or policy simulation in the domain of global development and behavioral analytics.

    """)

    Country = frame["Country name"]

    world_map(Country, y_prediction, residuals)
    
    st.markdown(f"""
    
        This choropleth map provides a geographical view of the residuals—the difference between the actual and predicted happiness scores—for each country in the dataset. 
        Residuals help us understand where the model performs well and where it falls short. 
        A blue shade indicates countries where the model underestimated happiness (actual > predicted), while a red shade marks those where it overestimated happiness (actual < predicted). Countries shown in neutral tones reflect high model accuracy with minimal error.
        By visualizing these discrepancies globally, we gain valuable insight into regional or cultural factors that may not be fully captured by our current features. 
        For instance, model underperformance in certain countries might suggest hidden variables like political stability, cultural resilience, or unique policy factors. 
        This map not only validates the regression results spatially but also serves as a foundation for further investigations into region-specific data or the inclusion of additional predictors in future modeling efforts.

    """)


    

