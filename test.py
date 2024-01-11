import streamlit as st
import seaborn as sns

st.header("This app is brought to you by #Irshad Ahmed")
st.text("kia apko maza aa raha h?")
st.header("This is my first app")

df = sns.load_dataset('iris')
st.write(df.head(4))

st.bar_chart(df['sepal_length'])
st.line_chart(df['sepal_length'])

from .model_methods import predict  # If model_methods.py is in the same directory
# Correct import statement
from model_methods import predict
print("Before import statement")
from model_methods import predict
print("After import statement")
