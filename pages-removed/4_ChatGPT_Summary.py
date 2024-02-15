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

import os


### ChatBot Imports
import streamlit as st 
if st.__version__ <"1.31.0":
    streaming=False
else:
    streaming=True

import time,os
# from streamlit_chat

## LLM Classes 
from langchain_openai.chat_models import ChatOpenAI
# from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

    
st.sidebar.subheader("Author Information")
with open("app-assets/author-info.html") as f:
    author_info = f.read()
with st.sidebar.container():
    components.html(author_info)#"""
    
# ## Memory Modules
# from langchain.chains.conversation.memory import (ConversationBufferMemory, 
#                                                   ConversationSummaryBufferMemory,
#                                                   ConversationBufferWindowMemory,
#                                                   ConversationSummaryMemory)
# Template for changing conversation chain's "flavor"
# from langchain.prompts.prompt import PromptTemplate


########### END OF IMPORTS

# # Changing the Layout
# st.set_page_config( #layout="wide", 
#                    page_icon="⭐️Amazon Reviews NLP Dash")



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



## Title /header
# st.header("Exploratory Data Analysis of Amazon Reviews ")
# st.divider()
st.header("ChatGPT-Summary")

# st.image(FPATHS['images']['banner_png'],width=700,use_column_width='always')
st.divider()
# ## Product metasata
# st.markdown("##### ***👈 Select the Display Options to enable/disable app components.***")
# st.divider()
# # st.subheader("Exploratory Analysis ")
####### CHATBOT


# Create required session_state containers
if 'messages' not in st.session_state:
    st.session_state.messages=[]
    
if 'API_KEY' not in st.session_state:
    st.session_state['API_KEY'] = os.getenv('OPENAI_API_KEY') # Could have user paste in via sidebar

if 'conversation' not in st.session_state:
    st.session_state['conversation'] = None


def reset():
    if 'messages' in st.session_state:
        st.session_state.messages=[]

    if 'conversation' in st.session_state:
        st.session_state['conversation'] = None


# # st.set_page_config(page_title="ChatGPT Clone", page_icon=':robot:')
# # st.header("Hey, I'm your Chat GPT")
# st.header("How can I assist you today?")



### ChatGPTM Options

## Special form of ngrams for chatgpt
from wordcloud import STOPWORDS
chatgpt_stopwords = [*STOPWORDS, 'angel','hair','miracle','noodle','shirataki','pasta']
    
def format_ngrams_for_chat(top_n_group_ngrams):
    
    string_table = []
    
    for group_name in top_n_group_ngrams.columns.get_level_values(0).unique():
        print(group_name)
        group_df = top_n_group_ngrams[group_name].copy()
        group_df['Rating Group'] = group_name 
        group_df = group_df.set_index("Rating Group")
        string_table.append(group_df)
        # string_table.append((group_df.values))
    return pd.concat(string_table)



## Define chatbot personalities/flavors
st.header("Ask ChatGPT for summary.")
# flavor_options = {
#     "Summary(General)": "You are a helpful assistant data scientist who uses ngrams from product reviews to summarize that customers do and do not like.",
#     "Summary(Recommendations)": "You are a helpful assistant data scientist who uses ngrams from product reviews to provide actionable recommendations for how to improve the product.",
#     "Customer (Low Carb/Gluten Free)": "You are a typical consumer who follows a low carb diet and has gluten sensitivity. You know what things you like in your food products.",
#     "Customer (General)":  "You are a typical consumer who does not follow a special diet and enjoys eating gluten-containing foods. You know what things you like in your food products.",
# }
flavor_options = {
    "Summary(General)": {'prompt':"You are a helpful assistant data scientist who uses ngrams from product reviews to summarize that customers do and do not like.",
                         'placeholder': ""},
    "Summary(Recommendations)": {'prompt':"You are a helpful assistant data scientist who uses ngrams from product reviews to provide actionable recommendations for how to improve the product.",
                                 'placeholder':''},
    "Customer (Low Carb/Gluten Free)": {'prompt':"You are a typical consumer who follows a low carb diet and has gluten sensitivity. You know what things you like in your food products.",
                                        "placeholder":''},
    "Customer (General)":  {'prompt':"You are a typical consumer who does not follow a special diet and enjoys eating gluten-containing foods. You know what things you like in your food products.",
                            "placeholder":""},
}

# def on_change_flavor():
#     st.session_state.text = st.session_state.text.upper()

# # st.text_area("Enter text", key="text")
# # st.button("Upper Text", on_click=on_upper_clicked)


@st.cache_resource
def load_chatgpt(temp,flavor_name):
    top_n_group_ngrams = fn.show_ngrams(df, top_n=25,ngrams=4, text_col_selection='review-text-full',
                                     stopwords_list=chatgpt_stopwords)
    md_table = format_ngrams_for_chat(top_n_group_ngrams)
    table_message = f"Heres a table of the most common ngrams from Low Rating reviews and high rating reviews. ```{md_table}```" 

    # Clear message history and specify flavor
    st.session_state.session_messages = [
    SystemMessage(content=flavor_options[flavor_name]['prompt']),
    SystemMessage(content=table_message)
    ]
    return  ChatOpenAI(temperature=temp,api_key=os.environ['OPENAI_API_KEY'])

if "placeholder" not in st.session_state:

    flavor_name = 'Summary(General)'
    
    
def set_placeholder(flavor_name):
        text = flavor_options[flavor_name]['placeholder']
        st.session_state.placeholder = text
        
if "placeholder" not in st.session_state:
    flavor_name ='Summary(General)'
    set_placeholder(flavor_name)
    
     

    # get_text()
    # st.session_state.text = st.session_state.text.upper()
     
col1,col2=st.columns(2)
flavor_name = col1.selectbox("Which type of chatbot?", key='flavor',options=list(flavor_options.keys()), index=1)#,on_change=set_placeholder)
# temp = col2.slider("Select model temperature:",min_value=0.0, max_value=2.0, value=0.1)
temp=0.1
reset_chat = st.sidebar.button("Clear history?")
if reset_chat:
    chat = load_chatgpt()
    # del st.session_state.session_messages


chat = load_chatgpt(temp,flavor_name)
if reset_chat:
    chat = load_chatgpt(temp,flavor_name)

def load_answer(query):#, model_name="gpt-3.5-turbo-instruct"):
    st.session_state.session_messages.append(HumanMessage(content=query))
    # Get answer and append to session state
    ai_answer = chat.invoke(st.session_state.session_messages)
    st.session_state.session_messages.append(AIMessage(content=ai_answer.content))
    return ai_answer.content
    # return show_history()
# Please give me a summary list of what customers liked  and did not like about the product."



def get_text():
    input_text = st.text_area("You: ", key='input', value =st.session_state.placeholder)#"Please give me a summary of what customers liked  and did not like about this product.")
    return input_text



def get_text():
    input_text = st.text_area("You: ", key='input', value ="Please give me a summary of what customers liked  and did not like about this product. How can I make it a better product?")
    return input_text

user_input=get_text()
submit = st.button('Generate')  

if submit:
    st.subheader(f"Answer - {flavor_name.title()}:")

    with st.container():
        response = load_answer(user_input)
    st.markdown(response)
else:
    st.empty()
