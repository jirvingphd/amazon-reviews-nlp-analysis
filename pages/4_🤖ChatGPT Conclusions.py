import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from PIL import Image
# import custom_functions as fn
import plotly.express as px
import plotly.io as pio
pio.templates.default='streamlit'

import os

import time,os
# from streamlit_chat

## LLM Classes 
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI


from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
set_llm_cache(InMemoryCache())



##Load in the data
import json
with open("config/filepaths.json") as f:
    FPATHS = json.load(f)
    
    
st.sidebar.subheader("Author Information")
with open("app-assets/author-info.html") as f:
    author_info = f.read()
with st.sidebar.container():
    components.html(author_info)#"""


# Get Fpaths
@st.cache_data
def get_app_fpaths(fpath='config/filepaths.json'):
	import json
	with open(fpath ) as f:
		return json.load(f)
    

@st.cache_data    
def load_df(fpath):
    import joblib
    return joblib.load(fpath)

@st.cache_data
def load_metadata(fpath):
    import pandas as pd
    return pd.read_json(fpath)


df = load_df(FPATHS['data']['processed-nlp']['processed-reviews-with-target_joblib'])
meta_df = load_metadata(FPATHS['data']['app']['product-metadata_json'])
product= meta_df.iloc[0]

@st.cache_data
def load_summaries(fpath):
    import json
    with open(fpath) as f:
        summaries = json.load(f)
    return summaries

summaries = load_summaries(FPATHS['results']['review-summary-01_json'])

st.header("Summaries & Recommendations")
st.markdown('We leveraged pre-trained summarization models from HuggingFace transformers to summarize all low and all high reviews. These summaries will be the contextual information that ChatGPT will use to provide the final conclusions.')

st.subheader("Summarized Low & High Reviews")
st.write("(made with HuggingFace transformers)")

col1, col2 =st.columns([.3,.7])
col1.image('images/hf-logo.png', width=100)
if col1.checkbox("Show model details",value=False) == True:
    col2.markdown("##### HuggingFace Model Details")
    col2.write(summaries['model-info'])
    

# col1,col2 = st.columns(2)

st.subheader("Low Reviews")
st.markdown(">"+summaries['summary-low'])
st.subheader("High Reviews")
st.markdown(">" + summaries['summary-high'])
st.divider()
# st.write("These summaries will be used as context for answering questions with ChatGPT below:")
# st.divider()

task_options = {"Summary":'provide a summary list of what customers did and did not like about the product in a list or table.',
                   'Recommendation':'provide a list of 3-5 actionable business recommendations on how to improve the product.'}
    


if 'messages' not in st.session_state:
    st.session_state.messages = []
    
if 'chat' not in st.session_state:
    st.session_state.chat = chat = ChatOpenAI(temperature=0)
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.environ['OPENAI_API_KEY']
    
def get_prompt_answer(summaries,selected_task,query):
    
    template_assistant = "You are a helpful assistant data scientist who uses NLP analysis to {task}. {context}."

    star_one = summaries['summary-low']
    star_five = summaries['summary-high']
    context = f"Here is a summary of 1-star reviews: {star_one}.\n\n Here is a summary of 5-star reviews{star_five}"
    
    
    # task_options = {"summarize":'summarize what customers did and did not like about the product.',
    #                'recommend':'provide a list of 3-5 actionable business recommendations on how to improve the product.'}
    
    
    # source: https://python.langchain.com/docs/integrations/chat/openai
    system_message_prompt = SystemMessagePromptTemplate.from_template(template_assistant)
    human_template = "{query}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )                       

    chat = st.session_state.chat#ChatOpenAI(temperature=0)
    prompt = chat_prompt.format_prompt(query=query, 
                                  context=context, task=task_options[selected_task]).to_messages()
    response = chat.invoke(prompt)
    
    st.session_state.messages.append(prompt)
    return response.content

st.header("Conclusions with ChatGPT")


st.image('images/OpenAI_Logo.svg')

def response_gen(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(.05)		

col1,col2 = st.columns(2)
selected_task = col1.selectbox("ChatGPT Task",options=list(task_options.keys()), index=0)
col2.write("\n\n\n    ")
ask_chatgpt = col2.button("Ask ChatGPT")

# col1,col2 = st.columns(2)
# check_summarize =  col1.checkbox("Summarize Findings")
# check_recommend =  col2.checkbox("Provide recommendations")
with st.container():
    if ask_chatgpt:
        st.subheader(f"ChatGPT {selected_task}")
        
        if 'summar' in selected_task.lower():
            # selected_task = 'Summarize'
            query = "What are the results of your analysis?"
        else:
            query = "How can we improve our product?"
            # col2.write("\n\n")

        resp = get_prompt_answer(summaries, selected_task,query)
        # st.write(st.session_state.messages)
        st.session_state.messages.append( AIMessage(content=resp))
        # st.write(resp)
        st.write_stream(response_gen(resp))
    else:
        st.empty()
        
        
    
# if check_recommend:
#     st.subheader("ChatGPT Recommendations")
#     selected_task = 'Recommend'
#     query = "How can we improve our product?"
#     # col2.write("\n\n")
#     # get_answer = col2.button("Submit")
#     resp = get_prompt_answer(summaries, selected_task,query)
#     st.session_state.messages.append( AIMessage(content=resp))
#     # st.write(resp)
#     st.write_stream(response_gen(resp))

# else:
#     st.empty()
    
    

# query = st.text_area("Your Question Here:", value="What are the results of your analysis?")
# col2.write("\n\n")
# get_answer = col2.button("Submit")

# if get_answer:
#     resp = get_prompt_answer(summaries, selected_task,query)
#     st.session_state.messages.append( AIMessage(content=resp))
#     st.write(resp)