import streamlit as st
import custom_functions as fn
import openai
import os
import joblib
import pandas as pd
import json
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
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories.streamlit import StreamlitChatMessageHistory

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set page configuration
st.set_page_config(page_icon="🤖AI Recommendations")

# Load file paths
FPATHS = fn.load_filepaths_json()
fpath_llm_csv = FPATHS['data']['app']['reviews-with-target-for-llm_csv']
fpath_db = FPATHS['data']['app']['vector-db_dir']

# Cache functions for loading data
@st.cache_data    
def load_df(fpath):
    return joblib.load(fpath)

@st.cache_data
def load_metadata(fpath):
    return pd.read_json(fpath)

@st.cache_data
def load_summaries(fpath):
    with open(fpath) as f:
        summaries = json.load(f)
    return summaries

# Load summaries
summaries = load_summaries(FPATHS['results']['review-summary-01_json'])

# Load metadata
meta_df = load_metadata(FPATHS['data']['app']['product-metadata_json'])
product = meta_df.iloc[0]

# Set avatars
ai_avatar = "🤖"
user_avatar = "💬"

# Display headers and information
st.header("Summaries & Recommendations")
st.markdown('We leveraged pre-trained summarization models from HuggingFace transformers to summarize all low and all high reviews. These summaries will be the contextual information that ChatGPT will use to provide the final conclusions.')
# st.write("(made with HuggingFace transformers)")

# col1.image('images/hf-logo.png', width=100)
with st.expander("🤗Show HuggingFace model details", expanded=False):
    # col1, col2 = st.columns([.3, .7])
    # col2.markdown("##### HuggingFace Model Details")
    st.write(summaries['model-info'])
st.divider()

# Display summarized low and high reviews
st.subheader("Summarized Low & High Reviews")
# Display product information
# show_product = st.checkbox("Show Product Information", value=True)
# if show_product:
with st.expander("Product Information",expanded=True):
    # st.subheader("Product Information")
        col1, col2 = st.container(border=True).columns(2)
        col1.markdown(fn.display_metadata(meta_df))
        col2.image(product['Product Image'], width=300)
# else:
#     col1, col2 = st.columns(2)
#     col1.empty()
#     col2.empty()


# Display low and high reviews summaries
st.subheader("Low Reviews")
st.markdown(">" + summaries['summary-low'])
st.subheader("High Reviews")
st.markdown(">" + summaries['summary-high'])
st.divider()

# Load vector database
if os.path.exists(fpath_db):
    retriever = fn.load_vector_database(fpath_db, fpath_llm_csv, use_previous=True, as_retriever=True)
else:
    retriever = fn.load_vector_database(fpath_db, fpath_llm_csv, use_previous=False, as_retriever=True)

# Set up AI Recommendations section
st.header("AI Recommendations")
summary_container = st.container()
st.divider()
chat_container = st.container()
chat_container.header("Q&A")
output_container = chat_container.container(border=True)

# Reset agent function
def reset_agent(retriever, starter_message="Hello, there! Enter your question here and I will check the full reviews database to provide you the best answer.", get_agent_kws={}):
    agent_factory = fn.AgentFactory()
    agent_exec = agent_factory.get_agent(retriever=retriever, **get_agent_kws)
    agent_exec.memory.chat_memory.add_ai_message(starter_message)
    return agent_exec

# Reset agent and agent-summarize
if 'agent' not in st.session_state:
    st.session_state['agent'] = reset_agent(retriever=retriever)

if 'agent-summarize' not in st.session_state:
    factory = fn.AgentFactory()
    st.session_state['agent-summarize'] = factory.get_agent(retriever=retriever, template_string_func=lambda: factory.get_template_string_interpret(context_low=summaries['summary-low'], context_high=summaries['summary-high']))

# Display task options
task_options = fn.get_task_options(options_only=False)
with summary_container:
    with st.container(border=True):
        col1, col2 = st.columns(2)
        selected_task = col1.radio("Select task:", options=task_options.keys())
        col2.markdown("> *Click below to query ChatGPT*")
        show_recs = col2.button("Get response.")
    if show_recs:
        prompt_text = task_options[selected_task]
        st.chat_message("user", avatar=user_avatar).write(prompt_text)
        response = st.session_state['agent-summarize'].invoke({'input': prompt_text})
        st.chat_message('assistant', avatar=ai_avatar).write(fn.fake_streaming(response['output']))

# Chat input and output
with chat_container:
    user_text = st.chat_input(placeholder="Enter your question here.")
    with output_container:
        fn.print_history(st.session_state['agent'])
        if user_text:
            st.chat_message("user", avatar=user_avatar).write(user_text)
            response = st.session_state['agent'].invoke({"input": user_text})
            st.chat_message('assistant', avatar=ai_avatar).write(fn.fake_streaming(response['output']))

# Reset chat button
reset_chat = st.sidebar.button("Reset Chat?")
if reset_chat:
    with output_container:
        st.session_state['agent'] = reset_agent(retriever=retriever)
with open("app-assets/author-info.md") as f:
    author_info = f.read()

with st.sidebar.container(border=True):
    st.subheader("Author Information")
    st.markdown(author_info, unsafe_allow_html=True)