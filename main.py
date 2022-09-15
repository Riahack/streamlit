import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
#from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()

st.markdown(
    """
    <style>
    .main {
    background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache
def get_data(filename):
    data = pd.read_csv(filename)
    return data

with header:
    st.title("Runway Landing")

with dataset:

    st.header("Runway data sources")
    st.text("Many data sources formed our source: Metar, NOTAMS.")
    data = get_data('hack5.csv')
    st.write(data.head())
    st.subheader('Brake coeff with destination airports')
    bcoeff = pd.DataFrame(data['braking_coefficient'].value_counts()).head(50)
    st.bar_chart(bcoeff)

with features:

    st.header("Main runway features")
    st.markdown('* **first feature:** I created this feature because of this... I calculated it using this logic..')
    st.markdown('* **second feature:** I created this feature because of this... I calculated it using this logic..')
    #runway_data = pd.read_csv("hack4.csv")


with modelTraining:
    st.header("Model Training")
    st.subheader('Time to train the model!')
    st.text('Here you get to choose the hyperparameters of the model and see how the performance changes!')
    sel_col, disp_col = st.columns(2)
    max_depth = sel_col.slider('What should be the max_depth of the model?', min_value=10, max_value=100, value=20, step=10)
    n_estimators = sel_col.selectbox('How many trees should there be?', options=[100,200,300,'No limit'], index = 0)
    sel_col.text('Here is a list of features in my data:')
    sel_col.write(data.columns)
    input_feature = sel_col.text_input('Which feature should be used as the input feature?','braking_coefficient')
    if n_estimators == 'No limit':
        regr = XGBRegressor(max_depth=max_depth)
    else:
        regr = XGBRegressor(max_depth=max_depth, n_estimators=n_estimators)


    X = data[[input_feature]]
    #y = data[['destination_actual_icao']]

    #regr.fit(X, y)
    #prediction = regr.predict(y)
    disp_col.text('Mean absolute error of the model is:')
    #disp_col.write(mean_absolute_error(y, prediction))

    disp_col.text('Mean squared error of the model is:')
    #disp_col.write(mean_squared_error(y, prediction))

    disp_col.text('R squared score of the model is:')
    #disp_col.write(r2_score(y, prediction))
    


