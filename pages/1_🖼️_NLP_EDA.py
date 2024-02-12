import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from PIL import Image
import custom_functions as fn
import plotly.express as px
import plotly.io as pio
pio.templates.default='streamlit'
# Changing the Layout
st.set_page_config( #layout="wide", 
                   page_icon="⭐️Amazon Reviews NLP EDA")



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
    

    
st.sidebar.subheader("Author Information")
with open("app-assets/author-info.html") as f:
    author_info = f.read()
with st.sidebar.container():
    components.html(author_info)#"""
    
    

    #     <ul>
    #     <li>Analysis by James M. Irving, Ph.D.</li>
    #     <li><a href="https://github.com/jirvingphd/amazon-reviews-nlp-analysis">📁Project Repository</a></li>
    # <li><a href="https://www.linkedin.com/in/james-irving-phd" rel="nofollow noreferrer">
    #     <img src="https://i.stack.imgur.com/gVE0j.png" alt="linkedin"> LinkedIn
    # </a> </li>
    # <li><a href="https://github.com/jirvingphd" rel="nofollow noreferrer">
    #     <img src="https://i.stack.imgur.com/tskMh.png" alt="github"> Github
    # </a></li>
    # </ul>
    # """)
    
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



## Title /header
# st.header("Exploratory Data Analysis of Amazon Reviews ")
# st.divider()
st.header("Amazon Customer Reviews Analysis")

st.image(FPATHS['images']['banner_png'],width=700,use_column_width='always')
st.divider()
## Product metasata
st.markdown("##### ***👈 Select the Display Options to enable/disable app components.***")
st.divider()
# st.subheader("Exploratory Analysis ")

# Setting sidebar controls ahead of time
### TEXT SELECTION OPTIONS FOR SIDEBAR
## Select which text column/preprocessing
st.sidebar.header("Display Options")
st.sidebar.markdown(">*Select visualizations for main pane.*")
show_product= st.sidebar.checkbox("Show Product Information", value=True)
show_review_graphs = st.sidebar.checkbox("Show rating distibutions.", value=True)
show_yearly =st.sidebar.checkbox('Show yearly trends in reviews', value=True)
show_wordclouds = st.sidebar.checkbox('Show Word Couds', value=True)
show_scattertext = st.sidebar.checkbox("Show ScatterText Visual", value=False)
st.sidebar.divider()

st.sidebar.header("Text Preprocessing Options")
st.sidebar.markdown(">*Select form of text for NLP EDA visuals.*")

text_col_map  ={"Original Text":'review-text-full',
                "Tokenized Text (no stopwords)":'tokens',
            'Lemmatzied Text':'lemmas'
            }
text_preprocessing_selection  =  st.sidebar.radio("Select Tokenization",options=list(text_col_map.keys()),# ['Original','Lemmas','Cleaned Tokens'],
                                            index=0)
text_col_selection = text_col_map[text_preprocessing_selection]
## Select # of words/ngrams
ngram_map = {'Unigrams/Tokens (1 Word)':1,
            'Bigrams (2 words)':2,
            'Trigrams (3 words)':3,
            'Quadgrams (4 words)':4}
ngram_selection = st.sidebar.radio("Select ngrams", options=list(ngram_map.keys()), #['Single Words','Bigrams','Trigrams','Quadgrams'],
                        index=1)
ngram_n = ngram_map[ngram_selection]
# Select custom stopwords
add_stopwords_str = st.sidebar.text_input("Enter list of words to exclude:",value='five,one,star')
stopwords_list = fn.get_stopwords_from_string(add_stopwords_str)


st.sidebar.divider()


st.sidebar.subheader("Dev Options")
dev_show_fpaths = st.sidebar.checkbox('[Dev] Show FPATHS?',value=False)
dev_show_frame = st.sidebar.checkbox("[Dev] Show frame?",value=False)

if dev_show_fpaths:
    FPATHS
    
if dev_show_frame:
    st.dataframe(df.head())

st.subheader("Product Information")

if show_product==True:


    # st.markdown(f'Product Title: ***{product["Title (Raw)"]}***')
    # st.divider()
    col1,col2 = st.columns(2)

    # @st.cache_data
    def display_metadata(meta_df,iloc=0):
        # product = meta_df.iloc[iloc]
        # md = "#### Product Being Reviewed"
        md = ""
        md += f'\n- Product Title:\n***\"{product["Title (Raw)"]}\"***'
        # md += f"<p><img src='{product['Product Image']}' width=300px></p>"
        md += f'\n- Brand: {product["Brand"]}'
        md += f"\n- Price: {product['Price']}"
        md += f"\n- Ranked {product['Rank']} (2018)"

        md += f"\n- Categories:\n    - "
        md += "; ".join(product['Categories'])
        # md += f"\n- Categories:{', '.join(product['Categories'])}"
        
        
        return md

    col1.markdown(display_metadata(meta_df))
    col2.image(product['Product Image'],width=300)
else:
    col1,col2 =st.columns(2)
    col1.empty()
    col2.empty()

st.divider()

# st.image(FPATHS['images']['selected-product_jpg'])

## Distrubtion of reviews
# label: color
colors = {
    1: "red",
    2: "orange",
    3: "yellow",
    4:'limegreen',
    5:'green'}
muted_colors = fn.mute_colors_by_key(colors,keys_to_mute=None, saturation_adj=.7, lightness_adj=3)
df = df.sort_values('year', ascending=False)


st.markdown("#### Distribution of Star-Ratings for Selected Product")

if show_review_graphs==True:
    # show_histogram = st.checkbox("Show overall ratings distribution.")

    # if show_histogram:

        ## Plot histogram
    st.plotly_chart(px.histogram(df, 'overall', color='overall',width=600,
                                # title='# of Reviews per Star Rating',
                                color_discrete_map=muted_colors))

else:
    st.empty()

st.divider()

st.markdown("#### Change in Average Ratings By Year")
if show_yearly==True:

    ## Plot average scatter with trendline by year
    avg_by_year = get_average_rating_by_year(df)
    st.plotly_chart(px.scatter(avg_by_year, trendline='ols', width=800, height=400,
                            # title='Average Rating over Time'
                            ))


    st.divider()
    st.markdown("#### Trend in Proportion of Star Ratings over Time")
    # Plot counts by year
    counts_by_year=  get_rating_percent_by_year(df)
    stars_to_plot = st.multiselect('Ratings (Stars) to Include', options=list(counts_by_year.columns),
                                default=[1,5])
    # counts_by_year = counts_by_year.reset_index(drop=False)
    melted_counts_by_year = get_rating_percent_by_year(df, melted=True)
    melted_counts_by_year = melted_counts_by_year[melted_counts_by_year['Stars'].isin(stars_to_plot)]

    st.plotly_chart(px.scatter(melted_counts_by_year, x='year', y='%', color='Stars',
                            color_discrete_map=muted_colors,# title='Trend in Proportion of Star Ratings over Time',
                            trendline='ols'))
else:
    st.empty()

st.divider()
# st.plotly_chart(px.histogram(df, 'overall', color='overall',title='# of Reviews per Star Rating',animation_frame='year'))


## word clouds
st.header("NLP EDA")
st.divider()
st.subheader("Word Clouds")



# Get groups dict
@st.cache_data
def fn_get_groups_freqs_wordclouds(df,ngrams=ngram_n, as_freqs=True, 
                                        group_col='target-rating', text_col = text_col_selection,
                                        stopwords=stopwords_list):
    kwargs = locals()
    group_texts = fn.get_groups_freqs_wordclouds(**kwargs) #testing stopwords
    return group_texts
## MENU FOR WORDCLOUDS
if show_wordclouds:
    st.markdown("👈 Change Text Preprocessing Options on the sidebar.")



    # wc_col1, wc_col2 = st.columns(2)
    # text_preprocessing_selection  =  wc_col1.radio("Select Text Processing",options=list(text_col_map.keys()),# ['Original','Lemmas','Cleaned Tokens'],
    #                                         index=0)
    # text_col_selection = text_col_map[text_preprocessing_selection]

    # ## Select # of words/ngrams
    # ngram_map = {'Single Words':1,
    #             'Bigrams':2,
    #             'Trigrams':3,
    #             'Quadgrams':4}
    # ngram_selection = wc_cofl2.radio("Select ngrams", options=list(ngram_map.keys()), #['Single Words','Bigrams','Trigrams','Quadgrams'],
    #                         index=0)
    # ngram_n = ngram_map[ngram_selection]

    # Select custom stopwords
    # add_stopwords_str = wc_col1.text_input("Enter list of words to exclude:",value='five,one,star')
    # stopwords_list = fn.get_stopwords_from_string(add_stopwords_str)

    
    group_texts = fn_get_groups_freqs_wordclouds(df,ngrams=ngram_n, as_freqs=True,group_col='target-rating', text_col = text_col_selection,
                                            stopwords=stopwords_list )
    # preview_group_freqs(group_texts)
    
    col1, col2 = st.columns(2)
    min_font_size = col1.number_input("Minumum Font Size",min_value=4, max_value=50,value=6, step=1)
    max_words = col2.number_input('Maximum # of Words', min_value=10, max_value=1000, value=200, step=5)
    
    fig  = fn.make_wordclouds_from_freqs(group_texts,stopwords=stopwords_list,min_font_size=min_font_size, max_words=max_words)
    
    st.pyplot(fig)
else:
    st.empty()
 
st.divider()


## Add creating ngrams
st.subheader('N-Grams')

# ngrams = st.radio('n-grams', [2,3,4],horizontal=True,index=1)
top_n = st.select_slider('Compare Top # Ngrams',[10,15,20,25],value=15)
## Compare n-grams
ngrams_df = fn.show_ngrams(df,top_n, ngram_n,text_col_selection,stopwords_list=stopwords_list)
fig = fn.plotly_group_ngrams_df(ngrams_df,show=False, title=f"Top {top_n} Most Common ngrams")
st.plotly_chart(fig)

# ## scattertext
# @st.cache_data
# def load_scattertext(fpath):
# 	with open(fpath) as f:
# 		explorer = f.read()
# 		return explorer

# st.subheader("ScatterText:")
# if show_scattertext:


#     # checkbox_scatter = st.checkbox("Show Scattertext Explorer",value=True)
#     # if checkbox_scatter:
#     with st.spinner("Loading explorer..."):
#         html_to_show = load_scattertext(FPATHS['eda']['scattertext-by-group_html'])
#         components.html(html_to_show, width=1200, height=800, scrolling=True)
# else:
#     st.empty()