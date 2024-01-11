import streamlit as st 
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# make containers
header = st.container()
datasets = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("kashti ki app")
    st.text("In this project we will work on kashti data")

with datasets:
    st.header("kashti doob gai")
    st.text("We will work with titanic dataset")
    # import data
    df = sns.load_dataset('titanic')
    df = df.dropna()
    st.write(df.head(10))
    
    st.subheader("sambha, are oooh sambha, kitny admi thy?")
    st.bar_chart(df['sex'].value_counts())

    #other plot
    st.subheader("class k hisaab se farq")
    st.bar_chart(df['class'].value_counts())



#barplot
    st.bar_chart(df['age'])

with features:
    st.header("These are our app features")
    st.text("Awein bht saary features add krty hain")
    st.markdown("1. **Feature 1:** This will tell us pata nahi")


# with model_training:
#     st.header("kashti walon kia bna?-model training")
#     st.text("In this project we will work on kashti data")
    # making columns
    input, display = st.columns(2)


# Assuming you want two columns
features, display = st.columns(2)

# Use the 'features' column
features.markdown("1. **Feature 1:** This will tell us pata nahi")
max_depth = input.slider("How many people do you know?", min_value=100, value=20, step=5)


# Use the 'display' column
display.bar_chart(df['class'].value_counts())

# n_estimators
n_estimators = input.selectbox("How many tress should be in RF?", options=[50,100,200,300,'NO limit'])

#adding list of features
input.write(df.columns)

#input features from user
input_features = input.text_input('Which feature we should use?')

#machine learning model

model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

#yahan aik condition lgaien gy
if n_estimators == 'No Limit':
    model = RandomForestRegressor(max_depth=max_depth)
else:
    model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
#define X and Y
X = df[[input_features]]
Y = df[['fare']]

#fit our model
model.fit(X, Y)
pred = model.predict(Y)

#Display metrices
display.subheader("Mean absolute error of the model is")
display.write(mean_absolute_error(y, pred))

display.subheader("Mean squared error of the model is")
display.write(mean_squared_error(y, pred))

display.subheader("R square score of the model is")
display.write(r2_score(y, pred))