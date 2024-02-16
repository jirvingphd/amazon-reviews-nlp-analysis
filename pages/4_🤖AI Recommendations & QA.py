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
import streamlit as st 
if st.__version__ <"1.31.0":
    streaming=False
else:
    streaming=True

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
import custom_functions as fn
# set_llm_cache(InMemoryCache())



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



#### COMMENTED OUT TO WORK ON NEW QA PART OF APP
task_options = {"Summary of Customer Sentiment":'provide a summary list of what 1-star reviews did not like and  what did 5-star reviews liked.',
                   'Product Recommendations':'provide a list of 3-5 actionable business recommendations on how to improve the product.',
                   'Marketing Recommendations':'provide a list of 2-5 recommendations for the marketing team to on how to better set customer expectations before purchasing the product or to better target the customers who will enjoy it.'}
    

if 'conversation' not in st.session_state:
    st.session_state['conversation'] =None
if 'chat' not in st.session_state:
    st.session_state.chat = chat = ChatOpenAI(temperature=0)
if 'messages' not in st.session_state:
    st.session_state['messages'] =[]
if 'API_Key' not in st.session_state:
    st.session_state['API_Key'] =''
    

def get_prompt_answer(summaries,selected_task,query):
    
    #{task}. {context}."
    # query = task_options[selected_task]
    star_one = summaries['summary-low']
    star_five = summaries['summary-high']
    context = f"\nHere is a summary of 1-star reviews: {star_one}.\n\n Here is a summary of 5-star reviews{star_five}"
    
    
    # task_options = {"summarize":'summarize what customers did and did not like about the product.',
    #                'recommend':'provide a list of 3-5 actionable business recommendations on how to improve the product.'}
    template_assistant = "You are a helpful assistant data scientist who uses NLP analysis of customer reviews to inform business-decision-making. Answer all questions using the following summaries:"
    template_assistant+=context
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



# st.image('images/OpenAI_Logo.svg')

def response_gen(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(.05)		

st.header("AI Recommendations")
with st.container(border=True):
    ## Select summary or recommendation
    col1,col2 = st.columns(2)
    # show_summary = col1.button("Show Summary of Customer Sentiment")
    # show_recommendations = col1.button("Get product improvement recommendations",)
    # show_marketing_recs = col2.button("Get marketing recommendations.")
        
    selected_task = col1.radio("Select task:", options=task_options.keys())
    col2.markdown("> *Click below to query ChatGPT*")
    show_recs = col2.button("Get response.")

    # buttons = {}
    # for task_name in task_options:
    #     buttons[task_name] = st.button(task_name)#, on_click=)



    
# selected_task = col1.selectbox("ChatGPT Task",options=list(task_options.keys()), index=0)
# col2.write("\n\n\n    ")
# ask_chatgpt = col2.button("Ask ChatGPT")

# col1,col2 = st.columns(2)
# check_summarize =  col1.checkbox("Summarize Findings")
# check_recommend =  col2.checkbox("Provide recommendations")
    # for selected_task,v in buttons.items():
    if show_recs:
        st.divider()
        st.markdown(f"#### ChatGPT {selected_task}")
        # st.divider()

        # if 'summar' in selected_task.lower():
        #     # selected_task = 'Summarize'
        #     query = "What are the results of your analysis?"
        # else:
        #     query = "How can we improve our product?"
        #     # col2.write("\n\n")
        query = task_options[selected_task]
        resp = get_prompt_answer(summaries, selected_task,query)
        # st.write(st.session_state.messages)
        st.session_state.messages.append( AIMessage(content=resp))
        # st.write(resp)
        st.write_stream(response_gen(resp))
    else:
        st.empty()




st.divider()

############# Q&A with ChatGPT
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ChatMessageHistory, ConversationSummaryBufferMemory, ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent



## Connect to vector database
fpath_db = FPATHS['data']['app']['vector-db_dir']

# db = Chroma(persist_directory=fpath_db, 
#            embedding_function=OpenAIEmbeddings())
    

# def get_agent(fpath_db, k=8, temperature=0.1,topic =  "answering questions about the product",
#              return_messages=True):
    
#     ## Make retreieval tool
#     tool = create_retriever_tool(
#          db.as_retriever(k=k),
#         "search_reviews",
#         "Searches and returns excerpts from Amazon user reviews.",
#     )
#     tools = [tool]
#     # Pull starter prompt from langchainhub
#     prompt = hub.pull("hwchase17/openai-tools-agent")
#     # Update starter prompt 
#     template = f"You are a helpful assistant for {topic} based on the Amazon product review documents. Include quotes from the documents, when appropriate."
#     # template+=f"Here is some additional metadata about the product for your reference: ```{product.to_string()}```"
#     # template = "You are a helpful assistant for answering questions about the product from the product reviews documents."
#     prompt.messages[0] = SystemMessagePromptTemplate.from_template(template)
#     prompt = ChatPromptTemplate.from_messages(prompt.messages)
#     # prompt.messages[0] = prompt.messages[0].format_messages(topic=topic)

#     llm = ChatOpenAI(temperature=0)
#     agent = create_openai_tools_agent(llm, tools, prompt)
#     agent_executor = AgentExecutor(agent=agent, tools=tools, 
#                                memory=ConversationBufferMemory(return_messages=return_messages))
#     return agent_executor

fpath_llm_csv = FPATHS['data']['app']['reviews-with-target-for-llm_csv']
fpath_db = FPATHS['data']['app']['vector-db_dir']
db = fn.load_vector_database( fpath_db,fpath_llm_csv, delete=True)#, use_previous=False)

def get_agent(fpath_db, k=8, temperature=0.1,
             return_messages=True, verbose=False):
    
    
    # import custom_functions as fn
    from custom_functions.app_functions import load_product_info
    product_string = load_product_info(FPATHS['data']['app']['product-metadata-llm_json'])
    ## Make retreieval tool
    tool = create_retriever_tool(
         db.as_retriever(k=k),
        "search_reviews",
        "Searches and returns excerpts from Amazon user reviews.",
    )
    tools = [tool]

    # Pull starter prompt from langchainhub
    prompt = hub.pull("hwchase17/openai-tools-agent")

    # produt_string = 
    # # Replace system prompt
    template = f"You are a helpful data analyst for answering questions about what customers said about a specific  Amazon product using only content from use reviews."
    product_template = f" Assume all user questions are asking about the content in the user reviews. Note the product metadata is:\n```{product_string}```\n\n"
    template+=product_template
    
    # template+="\n\nUse information from the following review documents to answer questions:"
    # qa_prompt_template= "\n- Here are the review documents:\n----------------\n{agent_scratchpad}\n\n"
    qa_prompt_template ="""Use the following pieces of context (user reviews) to answer the user's question by summarizing the reviews. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n{agent_scratchpad}\n\n"""
    template+=qa_prompt_template
    # template+="Try to infer one based on the review documents, otherwise just say that you don't know, don't try to make up an answer"

    # Replace system prompt
    prompt.messages[0] = SystemMessagePromptTemplate.from_template(template)
    prompt = ChatPromptTemplate.from_messages(prompt.messages)

    if verbose:
        print(prompt.messages)
        
    llm = ChatOpenAI(temperature=temperature)
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, 
                                   memory=ConversationBufferMemory(return_messages=return_messages))
    return agent_executor



def reset_agent():
    st.session_state['qa_agent'] =  get_agent(fpath_db,k=8, return_messages=True)
    st.session_state['qa_messages'] = []        

if 'qa_agent' not in st.session_state:
    reset_agent()








def get_qa_response(query):

    response = st.session_state['qa_agent'].invoke({'input':query})
    # response = st.session_state['conversation'].predict(input=query)
    # st.session_state['messages'].append()
    # print(response['history'])#st.session_state['conversation'].memory.buffer)

    return response
    # return show_history()


def response_gen(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(.05)		
        

st.header("Q&A with ChatGPT")
# Create chatbot with selected temp
col1, col2 = st.columns([.15,.8])

# submit_button = st.button(label="Send")
response_container = st.container(border=True,height=400)


col1.write('')
user_avatar = col1.selectbox("Select an avatar:", options=['📞','🧔🏻‍♂️','👩‍🦰',"🥷","⌨️"], index=1)

temp=0


col2.markdown("Enter your question here and ChatGPT will check the full reviews database.")
user_input =  col2.chat_input(placeholder ="What did customers think about the cook time?")#, on_submit=display_chat)#



def display_chat():
    with response_container:
        # st.write("Response Container:")
        response = get_qa_response(user_input)
        ai_msg_md_prefix = f"**ChatGPT**:\n\t"

        for message in response['history']:#[:-1]:#st.session_state.conversation.memory.buffer_as_messages[:-1]:
            type_message = str(type(message)).lower()
            
            if 'system' in type_message:
                continue
            
            elif "human" in type_message:
                human_message =  st.chat_message("user", avatar= user_avatar)#"🤷‍♂️")
                msg_md_format = f"**User:**\n\t{message.content}"
                human_message.write(msg_md_format)#f"User: {message.content}")
    
            elif 'ai' in type_message:
                ai_message = st.chat_message('ai',avatar= "🤖")
                msg_md_format = f"{ai_msg_md_prefix}:" #\n\t {message.content}"
                
                if message == response['history'][-1]:
                    if streaming==True:
                        ai_message.write_stream(response_gen(message.content))
                    else:
                        ai_message.write(message.content)
                else:
                    # ai_message =  st.chat_message('ai',avatar= "🤖")
                    msg_md_format = f"{ai_msg_md_prefix} {message.content}"
                    # ai_message.write(msg_md_format)
                    ai_message.write(msg_md_format)

        # last_message= response['history'][-1]#st.session_state.conversation.memory.buffer_as_messages[-1]
        # with st.chat_message('ai',avatar= "🤖"):
        #     msg_md_format = f"{ai_msg_md_prefix}:" #\n\t {message.content}"
        #     # st.write(msg_md_format)
        #     # st.markdown(msg_md_format)
            
        #     if streaming==True:
        #         st.write_stream(response_gen(last_message.content))
        #     else:
        #         st.write(last_message.content)



# container = st.chat_message('Human')#st.container()
# response_container.empty()

if user_input:
    display_chat()
    
if st.button("Reset Chat?"):
    reset_agent()
    

# container = st.chat_message('Human')#st.container()
# user_input =  response_container.chat_input(placeholder ="Hello,there!")#, on_submit=display_chat)#



# if user_input:
#     # with st.chat_message("user"):
#     # 	st.markdown(user_input)
#     st.session_state.messages.append(HumanMessage(content=user_input))
#     response = get_response(user_input)
#     st.session_state.messages.append(AIMessage(content=user_input))
#     display_chat()
#     # with st.chat_message("assistant"):
#     # 	st.write(response)

# model_response=#,st.session_state['API_KEY'])

        
# summarise_button = st.sidebar.button("Summarise the conversation", key="summarise")
# if summarise_button:
# 	summarise_placeholder = st.sidebar.write("Nice chatting with you my friend ❤️:\n\n"+st.session_state['conversation'].memory.buffer)

        

# clear = st.sidebar.button("Clear history?")

# if clear:
#     reset()