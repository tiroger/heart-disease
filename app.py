#############
# LIBRARIES #
#############

import pandas as pd
import numpy as np
import plotly.express as px

import streamlit as st

import pickle as pkl

import warnings
warnings.filterwarnings("ignore")

#########################################
# LOADING MODEL AND ENCODER/TRANSFORMER #
#########################################

st.set_page_config(page_title='Healthy Heart', layout='centered', page_icon='./Images/cardiology-red-icon-256.png')

loaded_extraTree_model = pkl.load(open('./Model/final_model.pickle', 'rb'))
loaded_encoder = pkl.load(open('./Model/feature_encoder.pickle', 'rb'))
OldPeak_tranformer = pkl.load(open('./Model/OldPeak_tranformer.pickle', 'rb'))
Cholesterol_transformer = pkl.load(open('./Model/Cholesterol_transformer.pickle', 'rb'))

# Front end elements
html_temp = """ 
    <div style ="background-color:white; padding:5px"> 
    <h1 style ="color:black; text-align:center">Caring for your Heart</h1>
    <img src="https://cpb-eu-w2.wpmucdn.com/blogs.brighton.ac.uk/dist/f/6375/files/2019/12/website-pic-2.gif" alt="Stylized heart" style="width:100%;height:auto;"> 
    </div>
    <h3>Heart Disease</h3>
    <p> 
    Cardiovascular disease (CVD) is the leading cause of deaths globally. An estimated 18 million people died from CVD in 2019, accounting for about 32% of all deaths. Of those, 85% were due to heart attacks and strokes and about 30% occured in people under the age of 70.
    </p>
    <h3>Risk factors for CVD</h3>
    <p>
    The most important behavioral risk factors for heart disease and stroke are unhealthy diet, physical inactivity, tobacco use and excessive consumption of alcohol. The effects of these risk factors may manifest in individuals as high blood pressure, elevated blood glucose and lipids, and obesity. These “intermediate risks factors” can be measured in primary care facilities and indicate an increased risk of heart attack, stroke, heart failure and other complications.
    </p>
    <p>
    There are also a number of underlying determinants of CVD. These are a reflection of the major forces driving social, economic and cultural change --globalization, urbanization and population aging. Other determinants of CVD include poverty, stress and hereditary factors.
    </p>
    <i> For more information on CVD, please visit <a href=https://www.cdc.gov/heartdisease/facts.htm>www.cdc.gov/heartdisease/facts</a></i>
    """

html_temp_2 = """
    <h3>Predicting Heart Failure</h3>
    <p>
    Heart failure is a common event caused by CVD and this prediction model contains 11 features that can be used to predict the risk of heart disease.
    </p>
    <p>
    People with CVD or those who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or other comorbidities) need early detection and management wherein a machine learning model can be of great help.
    </p>
    """

html_chart = """
    <h3>Cardiovascular Disease Mortality Trends </h3>
"""

st.markdown(html_temp, unsafe_allow_html = True) 

html_space = """

"""

########################################################
# FUNCTIONS FOR PROCESSING DATA AND MAKING PREDICTIONS #
########################################################

def process_data(user_input_df):
    # user_input_df = pd.DataFrame(user_input) # Creating a dataframe containing user inputs

    cat_cols = user_input_df.select_dtypes(include='object').columns
    user_input_df[cat_cols] = user_input_df[cat_cols].astype('category')

    user_input_df['FastingBS'] = np.where(user_input_df['FastingBS']=='No', 0, 1)
    user_input_df['FastingBS'] = user_input_df['FastingBS'].astype('category') # Converting to type category for transforming
    
    # Encoding categorical features
    categorical_columns = user_input_df.select_dtypes('category').columns
    user_input_df[categorical_columns] = loaded_encoder.transform(user_input_df[categorical_columns])

    # Transforming data
    transformed_OldPeak = OldPeak_tranformer.transform(user_input_df[['Oldpeak']])
    transformed_Cholesterol = Cholesterol_transformer.transform(user_input_df[['Cholesterol']])
    transformed_RestingBP = np.log(user_input_df[['RestingBP']])
    user_input_df['Oldpeak'] = transformed_OldPeak
    user_input_df['Cholesterol'] = transformed_Cholesterol
    user_input_df['RestingBP'] = transformed_RestingBP

    return user_input_df

def make_prediction(user_input_df):
    prediction = loaded_extraTree_model.predict(user_input_df)
    prediction_probability = loaded_extraTree_model.predict_proba(user_input_df).max()

    results = prediction, prediction_probability
    return results



################
# PLOTLY CHART #
################

st.markdown(html_space, unsafe_allow_html = True)

# st.markdown(html_chart, unsafe_allow_html = True)

cvd_deaths = pd.read_csv('Data/cardiovascular-disease-death-rates.csv')
cvd_deaths.rename(columns={'Deaths - Cardiovascular diseases - Sex: Both - Age: Age-standardized (Rate)': 'CVD Deaths'}, inplace=True)

grouped = cvd_deaths.groupby(['Year', 'Entity']).max().reset_index()
grouped['CVD Deaths'] = round(grouped['CVD Deaths'], 0)
all_entities = grouped.Entity.unique()

entity_to_show = st.multiselect('Select Country or Entity', all_entities, default=['World', 'China', 'Japan', 'United States'])


fig = px.line(grouped.loc[grouped['Entity'].isin(entity_to_show)], x="Year", y="CVD Deaths", color='Entity', title="Death rate from cardiovascular disease, 1990 to 2017 <br><sup>The annual number of deaths from cardiovascular disease per 100,000 people.</sup>", markers=True)
fig.for_each_trace(lambda trace: fig.add_annotation(
    x=trace.x[-1], y=trace.y[-1], text='  '+trace.name, 
    font_color=trace.line.color,
    ax=10, ay=10, xanchor="left", showarrow=False))

# fig.update_yaxes(title='y', visible=False, showticklabels=True)    
fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            # showgrid=True,
            zeroline=False,
            showline=True,
            gridcolor = 'rgb(235, 236, 240)',
            showticklabels=True,
            title='',
            autorange=True
        ),
        autosize=True,
        hovermode="x unified",
        # margin=dict(
        #     autoexpand=True,
        #     l=100,
        #     r=20,
        #     t=110,
        # ),
        showlegend=False,
#         legend=dict(
#         # orientation="h",
#         yanchor="bottom",
#         y=0.9,
#         xanchor="left",
#         x=0.7
# ),
        plot_bgcolor='rgba(0,0,0,0)'
    )
# fig.show()




st.plotly_chart(fig, use_container_width = True)

st.markdown(html_temp_2, unsafe_allow_html = True) 

#############
# STREAMLIT #
#############

# Creating boxes where users can enter data required to make prediction
age = int(st.slider("Age (Years)", min_value=1, value=32, max_value=121))
sex = st.radio('Select Gender (M: Male, F: Female)', ('M', 'F'), index=0)
ChestPainType = st.selectbox('Chest Pain Type (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic)', ("TA","ATA","NAP","ASY"), index=3) 
restingBP = int(st.slider('Resting Blood Pressure (mm Hg)', min_value=10, value=120, max_value=240))
RestingECG = st.selectbox('Resting Electrocardiographic Results [Normal: Normal, ST: Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: Showing probable or definite left ventricular hypertrophy by Estes criteria]', ("Normal", "ST", "LVH"), index=0)
cholesterol = int(st.slider('Serum Cholestoral (mg/dl)', min_value=1, value=170, max_value=500))
fastingBS = st.radio('Fasting Blood Sugar (Y: If FastingBS > 120 mg/dl, N: Otherwise)', ['Y','N'], index=1)
MaxHR = int(st.slider('Maximum Heart Rate Achieved', min_value=60, value=190, max_value = 240))
ExerciseAngina = st.radio('Exercise Induced Angina (Y: Yes, N: No)',['Y','N'], index=1)
Oldpeak = int(st.slider('Oldpeak--ST (Numeric value measured in depression)', min_value=-10, value=0, max_value = 10))
stSlope = st.selectbox('Heart Rate Slope--the slope of the peak exercise ST segment (Up: Upsloping, Flat: Flat, Down: Downsloping)', ("Up","Flat","Down"), index=0)


user_input = {
                'Age': [age], 
                'Sex': [sex],
                'ChestPainType': [ChestPainType], 
                'RestingBP': [restingBP], 
                'Cholesterol': [cholesterol],
                'FastingBS': [fastingBS],
                'RestingECG': [RestingECG],
                'MaxHR': [MaxHR],
                'ExerciseAngina': [ExerciseAngina],
                'Oldpeak': [Oldpeak],
                'ST_Slope': [stSlope],
}
user_input_df = pd.DataFrame(user_input) # Creating a dataframe containing user inputs
# st.dataframe(user_input_df)


processed_data = process_data(user_input_df)
results = make_prediction(processed_data)



if st.button("Predict"):    
    if results[0] == 0:
        st.success(f'You are not at risk for heart disease --Prediction Confidence: {"{:.0%}".format(results[1])}.')
        st.image('https://h2hcardiaccenter.com/blog/wp-content/uploads/2018/07/shutterstock_556072003-1160x650-1024x574.jpg')
    else:
        st.error(f'You ARE at risk for heart disease! --Prediction Confidence: {"{:.0%}".format(results[1])}.')
        st.image('http://4.bp.blogspot.com/-sZAA_0WxS2c/UQFFWsL76LI/AAAAAAAAAC0/9RxHrWV6aWo/s1600/heart+disease.jpg')


st.sidebar.subheader("About")
st.sidebar.info('This web app is a tool to help you determine whether you are at risk of developing cardiovascular disease.')

st.sidebar.subheader("How to use")
st.sidebar.info('Enter your information in each of the fields and click the "Predict" button.')

st.sidebar.subheader("Disclaimer")
st.sidebar.info('This web app does not provide medical advice. The information, including but not limited to, text, graphics, images and other material contained here are for informational purposes only. No material on this site is intended to be a substitute for professional medical advice, diagnosis or treatment.')

st.sidebar.caption('Data Sources: Cardiovascular deaths data published by Global Burden of Disease Collaborative Network. Global Burden of Disease Study 2017 (GBD 2017) Results. Seattle, United States: Institute for Health Metrics and Evaluation (IHME), 2018.')

link = '[Roger Lefort](http://www.thecyclingscientist.com/)'
st.sidebar.caption('By ' + link, unsafe_allow_html=True)
