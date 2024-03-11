import streamlit as st
import custom_functions as fn
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.callbacks import StdOutCallbackHandler
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import CSVLoader

from langchain.agents import AgentExecutor, create_openai_tools_agent
# Memory: agent token buffer used in original example blog post
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.memory import ConversationBufferMemory

#https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/chat_with_documents.py
from langchain.memory.chat_message_histories.streamlit import StreamlitChatMessageHistory
# Changing the Layout
st.set_page_config( #layout="wide", 
                   page_icon="🤖AI Recommendations")

import openai, os
openai.api_key = os.getenv("OPENAI_API_KEY")

FPATHS = fn.load_filepaths_json()
fpath_llm_csv = FPATHS['data']['app']['reviews-with-target-for-llm_csv']
fpath_db = FPATHS['data']['app']['vector-db_dir']

@st.cache_data    
def load_df(fpath):
    import joblib
    return joblib.load(fpath)

@st.cache_data
def load_metadata(fpath):
    import pandas as pd
    return pd.read_json(fpath)



@st.cache_data
def load_summaries(fpath):
    import json
    with open(fpath) as f:
        summaries = json.load(f)
    return summaries

summaries = load_summaries(FPATHS['results']['review-summary-01_json'])
# df = load_df(FPATHS['data']['processed-nlp']['processed-reviews-with-target_joblib'])
meta_df = load_metadata(FPATHS['data']['app']['product-metadata_json'])
product= meta_df.iloc[0]



ai_avatar  = "🤖"
user_avatar = "💬"


st.header("Summaries & Recommendations")
st.markdown('We leveraged pre-trained summarization models from HuggingFace transformers to summarize all low and all high reviews. These summaries will be the contextual information that ChatGPT will use to provide the final conclusions.')

st.divider()
# def load_product_info(fpath):
#     import json
#     with open(fpath,'r') as f:
#         product_json = json.load(f)
        
#     product_string = "Product Info:\n"
#     for k,v in product_json.items():
#         if k.lower()=='description':
#             continue
#         product_string+=f"\n{k} = {v}\n"
        
#     return product_string


show_product= st.checkbox("Show Product Information", value=True)

if show_product==True:
    st.subheader("Product Information")


    # st.markdown(f'Product Title: ***{product["Title (Raw)"]}***')
    # st.divider()
    col1,col2 = st.columns(2)


    col1.markdown(fn.display_metadata(meta_df))
    col2.image(product['Product Image'],width=300)
else:
    col1,col2 =st.columns(2)
    col1.empty()
    col2.empty()


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

## Updated function
fpath_llm_csv = FPATHS['data']['app']['reviews-with-target-for-llm_csv']
fpath_db = FPATHS['data']['app']['vector-db_dir']

# with st.spinner("Constructing vector database for ChatGPT..."):
# Running one time to delete the database and make fresh


    
# retriever  = load_vector_database( fpath_db,fpath_llm_csv, k=8, use_previous=False, as_retriever=True)    
if os.path.exists(fpath_db):
    retriever = fn.load_vector_database(fpath_db, fpath_llm_csv, use_previous=True, as_retriever=True)
else:
    retriever = fn.load_vector_database(fpath_db, fpath_llm_csv, use_previous=False, as_retriever=True)


# if 'retriever' not in st.session_state:
#     st.session_state['retriever'] = retriever

# Create chat container early
# st
st.header("AI Recommendations")
summary_container = st.container()

st.divider()

chat_container = st.container()
chat_container.header("Q&A")
output_container = chat_container.container(border=True)


    
## For steramlit try this as raw code, not a function
def print_history(agent_executor,ai_avatar  = "🤖", user_avatar = "💬"):
    # Simulate streaming for final message

    session_state_messages = agent_executor.memory.buffer_as_messages
    for msg in session_state_messages:#[:-1]:
        if isinstance(msg, AIMessage):
            # notebook
            print(f"Assistant: {msg.content}")
            # streamlit
            st.chat_message("assistant", avatar=ai_avatar).write(msg.content)
        
        elif isinstance(msg, HumanMessage):
            # notebook
            print(f"User: {msg.content}")
            # streamlit
            st.chat_message("user", avatar=user_avatar).write(msg.content)
        print()


# def fake_streaming(response):
#     import time
#     for word in response.split(" "):
#         yield word + " "
#         time.sleep(.05)		
            
            
def reset_agent(#fpath_db = FPATHS['data']['app']['vector-db_dir'],
                retriever=retriever, #st.session_state['retriever'] , 
                starter_message = "Hello, there! Enter your question here and I will check the full reviews database to provide you the best answer.",
               get_agent_kws={}):
    # fpath_db
    agent_factory = fn.AgentFactory()
    agent_exec = agent_factory.get_agent(retriever=retriever, **get_agent_kws)
    print("\n\n\n",type(agent_exec),"\n\n")
    agent_exec.memory.chat_memory.add_ai_message(starter_message)
    # with chat_container:
    #     st.chat_message("assistant", avatar=ai_avatar).write_stream(fn.fake_streaming(starter_message))
        # print_history(agent_exec)
    return agent_exec
    

# def fn.fake_streaming(response):
#     import time
#     for word in response.split(" "):
#         yield word + " "
#         time.sleep(.05)		
        
            
    
# ## For steramlit try this as raw code, not a function
# def print_history(agent_executor):
#     # Simulate streaming for final message

#     session_state_messages = agent_executor.memory.buffer_as_messages
#     for msg in session_state_messages:#[:-1]:
#         if isinstance(msg, AIMessage):
#             # notebook
#             print(f"Assistant: {msg.content}")
#             # streamlit
#             st.chat_message("assistant", avatar=ai_avatar).write(msg.content)
        
#         elif isinstance(msg, HumanMessage):
#             # notebook
#             print(f"User: {msg.content}")
#             # streamlit
#             st.chat_message("user", avatar=user_avatar).write(msg.content)
#         print()


# def get_task_options(options_only=False):
#     task_prompt_dict= {
#         # "Summary of Customer Sentiment":'Provide a summary list of what 1-star reviews did not like and a summary of what did 5-star reviews liked.',
#                    'Product Recommendations':'Provide a list of 3-5 actionable business recommendations on how to improve the product.',
#                    'Marketing Recommendations':'Provide a list of 3-5 recommendations for the marketing team to on how to better set customer expectations before purchasing the product or to better target the customers who will enjoy it.'}
#     if options_only:
#         return list(task_prompt_dict.keys())
#     else:
#         return task_prompt_dict



# def get_template_string_interpret(context_low, context_high, context_type='BERT-summary'):
#     # task_prompt_dict = get_task_options(options_only=False)
#     # system_prompt = task_prompt_dict[selected_task]
    
#     # template_assistant = "You are a helpful assistant data scientist who uses NLP analysis of customer reviews to inform business-decision-making:"
#     # product_template = f" Assume all user questions are asking about the content in the user reviews. Note the product metadata is:\n```{product_string}```\n\n"
#     template_starter = get_template_string_reviews()
#     context = f"\nGroup Contexts:\n Here is a {context_type} of 1-star reviews: ```{context_low}```.\n\n Here is a {context_type} of 5-star reviews:```{context_high}."
#     context += f"Use the {context_type} first before using the retrieved documents."
#     template_assistant=template_starter+ context
#     return template_assistant



if 'agent' not in st.session_state:
    # agent = get_agent(retriever)
    # with output_container:
    st.session_state['agent'] =reset_agent(retriever=retriever)#st.session_state['retriever'] )


if 'agent-summarize' not in st.session_state:
    factory = fn.AgentFactory()
    st.session_state['agent-summarize'] = factory.get_agent(retriever=retriever,#st.session_state['retriever'] ,
        template_string_func=lambda: factory.get_template_string_interpret(context_low=summaries['summary-low'],
                                                                   context_high=summaries['summary-high'])
    )


# ### ADD SUMMARRY AGENT
# st.header("Summaries and Recommendations")

## CHANGE CODE TO ADD HUMAN/AI MESSAGE to empty history
# Specify task options
# task_options = get_task_options(options_only=True) #task_prompt_dict.keys()
task_options  = fn.get_task_options(options_only=False)
with summary_container:

    with st.container(border=True):
        ## Select summary or recommendation
        col1,col2 = st.columns(2)
        # show_summary = col1.button("Show Summary of Customer Sentiment")
        # show_recommendations = col1.button("Get product improvement recommendations",)
        # show_marketing_recs = col2.button("Get marketing recommendations.")
            
        selected_task = col1.radio("Select task:", options=task_options.keys())
        col2.markdown("> *Click below to query ChatGPT*")
        show_recs = col2.button("Get response.")
    if show_recs:
        prompt_text =  task_options[selected_task]
        st.chat_message("user", avatar=user_avatar).write(prompt_text)
        
        response = st.session_state['agent-summarize'].invoke({'input':prompt_text})

        
        # print_history(st.session_state['agent-summarize'])

        # response = st.session_state['agent'].invoke({"input":prompt_text})
        st.chat_message('assistant', avatar=ai_avatar).write(fn.fake_streaming(response['output']))



with chat_container:
    # output_container = st.container(border=True)
    user_text = st.chat_input(placeholder="Enter your question here.")


    with output_container:
            
        fn.print_history(st.session_state['agent'])
        if user_text:
            st.chat_message("user", avatar=user_avatar).write(user_text)
        
            response = st.session_state['agent'].invoke({"input":user_text})
            st.chat_message('assistant', avatar=ai_avatar).write(fn.fake_streaming(response['output']))


reset_chat = st.sidebar.button("Reset Chat?")
if reset_chat:
    with output_container:
        st.session_state['agent'] =reset_agent(retriever=retriever)#st.session_state['retriever'] )
        # print_history(st.session_state['agent'])
        
st.sidebar.subheader("Author Information")
    
with open("app-assets/author-info.md") as f:
    author_info = f.read()
    
with st.sidebar.container():
    st.markdown(author_info, unsafe_allow_html=True)