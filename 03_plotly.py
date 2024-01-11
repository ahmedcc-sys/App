#import libraries
import streamlit as st 
import plotly.express as px
import pandas as pd 

#import dataset
st.title("plotly and streamlit ko mila k app bnana")
df = px.data.gapminder()
st.write(df)
st.write(df.head())
st.write(df.columns)

#summary stat
st.write(df.describe())


#data management
year_option = df['year'].unique().tolist()

year = st.selectbox("which year should we plot?", year_option, 0)
# df = df[df['year']== year]

#plotting

import plotly.express as px

# Example data
data = px.data.gapminder()

# Scatter plot with log scale on the x-axis
fig = px.scatter(data, x="gdpPercap", y="lifeExp", color="continent",
                  log_x=True, size_max=55, range_x=[100, 100000], range_y=[20, 90], 
                  animation_frame = 'year', animation_group='country')

# Show the plot
fig.show()
