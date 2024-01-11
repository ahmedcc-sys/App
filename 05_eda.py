import numpy as np 
import pandas as pd 
import streamlit as st 
import seaborn as sns
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

#Webap ka title
st.markdown('''
# **Exploratory Data analysis Web Application**
This app is developed by Ahmed called **EDA**
            ''')

#how to upload file from pc

with st.sidebar.header(" Upload your dataset (.csv)"):
    uploaded_file = st.sidebar.file_uploader("Upload your file", type=['csv'])

    df = sns.load_dataset('titanic')
    st.sidebar.markdown("[Example CSV file](https://github.com/ahmedcc-sys/Markdown-practice)")


#profiling report for pandas
if uploaded_file is not None:
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**Input DF**')
    st.write(df)
    st.write('---')
    st.header('**Profiling report with pandas**')
    st_profile_report(pr)
else:
    def load_csv():
     st.info('Awaiting for CSV file, upload kr b do bhai?')
    if st.button('Press to use example data'):
    #Example dataset
     a = pd.DataFrame(np.random.rand(100, 5), columns=['age', 'banana', 'codanics', 'Deutchland', 'Ear'])
    
    # Return the DataFrame
    return a
    df = load_data()
    pr = ProfileReport(df, explorative=True)
    st.header('**Input DF**')
    st.write(df)
    st.write('---')
    st.header('**Profiling report with pandas**')
    st_profile_report(pr)