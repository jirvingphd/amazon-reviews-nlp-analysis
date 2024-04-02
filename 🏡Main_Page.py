import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from PIL import Image
import custom_functions as fn
import json
import joblib
import pandas as pd

import streamlit.components.v1 as components
import plotly.express as px
import plotly.io as pio

pio.templates.default = 'streamlit'

# Changing the Layout
st.set_page_config(page_title="🏡Amazon Reviews NLP Dash")

## Load in the data
with open("config/filepaths.json") as f:
    FPATHS = json.load(f)

@st.cache_data    
def load_df(fpath):
    return joblib.load(fpath)

@st.cache_data
def load_metadata(fpath):
    return pd.read_json(fpath)

df = load_df(FPATHS['data']['processed-nlp']['processed-reviews-with-target_joblib'])
meta_df = load_metadata(FPATHS['data']['app']['product-metadata_json'])
product = meta_df.iloc[0]

## Title /header
st.header("Amazon Customer Reviews Analysis")
st.image(FPATHS['images']['banner_png'], width=700, use_column_width='always')

st.sidebar.markdown("> ☝️*Select page.*")

st.markdown("> This project utilizes the Amazon Reviews Dataset for a deep dive into customer sentiment surrounding a specific product.")

st.subheader("Product Information")

with st.expander("Product Information", expanded=True):
    col1, col2 = st.container(border=True).columns(2)
    col1.markdown(fn.display_metadata(meta_df))
    col2.image(product['Product Image'], width=300)

## Distribution of reviews
colors = {
    1: "red",
    2: "orange",
    3: "yellow",
    4: 'limegreen',
    5: 'green'
}
muted_colors = fn.mute_colors_by_key(colors, keys_to_mute=None, saturation_adj=.7, lightness_adj=3)
df = df.sort_values('year', ascending=False)

st.markdown("#### Distribution of Star-Ratings for Selected Product")

fig = px.histogram(df, 'overall', color='overall', width=600, color_discrete_map=muted_colors)
fig.update_layout(dragmode=False)
st.plotly_chart(fig, use_container_width=True)
st.markdown("> **Miracle Noodles have a large number of both 1 star and 5-star reviews.**")

st.divider()

st.markdown("#### Change in Average Ratings By Year")

avg_by_year = fn.get_average_rating_by_year(df)
fig = px.scatter(avg_by_year, trendline='ols', width=800, height=400)
fig.update_layout(dragmode=False)
st.plotly_chart(fig)

st.markdown("> **However, overall the average customer rating has decreased over time.**")

st.divider()

st.markdown("#### Trend in Proportion of Star Ratings over Time")

counts_by_year = fn.get_rating_percent_by_year(df)
stars_to_plot = st.multiselect('Ratings (Stars) to Include', options=list(counts_by_year.columns), default=[1, 5])
melted_counts_by_year = fn.get_rating_percent_by_year(df, melted=True)
melted_counts_by_year = melted_counts_by_year[melted_counts_by_year['Stars'].isin(stars_to_plot)]

fig = px.scatter(melted_counts_by_year, x='year', y='%', color='Stars', color_discrete_map=muted_colors, trendline='ols')
fig.update_layout(dragmode=False)
st.plotly_chart(fig, use_container_width=True)
st.markdown("> **The proportion of 1-star reviews has increased over time while the proportion of 5-star reviews decreased.**")

st.divider()

st.markdown("We will use Natural Language Processing and Machine Learning to provide insights into what customers do or do not like about Miracle Noodles")
st.divider()

st.markdown("##### ***👈 Select the app page via the sidebar. (Click on the `>` to show sidebar.)***")
st.divider()

st.subheader("References")
st.markdown("""> **Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering** R. He, J. McAuley *WWW*, 2016 [pdf](https://cseweb.ucsd.edu/~jmcauley/pdfs/www16a.pdf)

> **Image-based recommendations on styles and substitutes** J. McAuley, C. Targett, J. Shi, A. van den Hengel *SIGIR*, 2015 [pdf](https://cseweb.ucsd.edu/~jmcauley/pdfs/sigir15.pdf)

""")

with st.expander("Show Full Amazon Dataset Information", expanded=False):
    with open('data/Amazon Product Reviews.md') as f:
        st.markdown(f.read())

with open("app-assets/author-info.md") as f:
    author_info = f.read()

with st.sidebar.container(border=True):
    st.subheader("Author Information")
    st.markdown(author_info, unsafe_allow_html=True)
