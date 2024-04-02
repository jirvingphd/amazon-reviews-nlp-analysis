import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os
import joblib
# import tensorflow as tf
from PIL import Image
import custom_functions as fn
import plotly.express as px
import plotly.io as pio
pio.templates.default='streamlit'
# Changing the Layout
st.set_page_config( layout="wide", 
                   page_icon='📈',
                   page_title="Scattertext Explorer")

    
# st.sidebar.subheader("Author Information")
# with open("app-assets/author-info.html") as f:
#     author_info = f.read()
# with st.sidebar.container():
#     components.html(author_info)#"""
    
# Get Fpaths
@st.cache_data
def get_app_fpaths(fpath='config/filepaths.json'):
	import json
	with open(fpath ) as f:
		return json.load(f)



##Load in the data
import json
with open("config/filepaths.json") as f:
    FPATHS = json.load(f)
    

    

@st.cache_data    
def load_df(fpath):
    import joblib
    return joblib.load(fpath)

@st.cache_data
def load_metadata(fpath):
    import pandas as pd
    return pd.read_json(fpath)

@st.cache_data
def get_rating_percent_by_year(df,**kwargs):
    return fn.get_rating_percent_by_year(df,**kwargs)

@st.cache_data
def get_average_rating_by_year(df, **kwargs):
    return fn.get_average_rating_by_year(df,**kwargs)

df = load_df(FPATHS['data']['processed-nlp']['processed-reviews-with-target_joblib'])
meta_df = load_metadata(FPATHS['data']['app']['product-metadata_json'])
product= meta_df.iloc[0]





## scattertext
@st.cache_data
def load_scattertext(fpath):
	with open(fpath) as f:
		explorer = f.read()
		return explorer

st.header("ScatterText Explorer:")
# if show_scattertext:


    # checkbox_scatter = st.checkbox("Show Scattertext Explorer",value=True)
# if checkbox_scatter:
with st.container(border=True):
    st.markdown("The Scattertext visualization below plots the frequency of word usage for Low reviews (x-axis) and High Reviews (y-axis).")
    st.markdown("Click on a word below to see example reviews from both groups.")
    
with st.spinner("Loading explorer..."):
    html_to_show = load_scattertext(FPATHS['eda']['scattertext-by-group_html'])
    with st.container(border=True):
        components.html(html_to_show, width=1200, height=800, scrolling=True)
    # st.divider()
st.markdown('### ☝️ Scroll up to return to scatterplot')


with open("app-assets/author-info.md") as f:
    author_info = f.read()
    
with st.sidebar.container(border=True):
    st.subheader("Author Information")

    st.markdown(author_info, unsafe_allow_html=True)
    