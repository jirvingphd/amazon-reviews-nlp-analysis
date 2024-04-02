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
st.header("Review Summaries & AI Recommendations")
with st.expander("Click to Show Product Information",expanded=False):
    # st.subheader("Product Information")
        col1, col2 = st.container(border=True).columns(2)
        col1.markdown(fn.display_metadata(meta_df))
        col2.image(product['Product Image'], width=300)

# Display summarized low and high reviews
st.subheader("Review Summaries(🤗HuggingFace)")
        
st.markdown('>We leveraged pre-trained summarization models from HuggingFace transformers to summarize all low and all high reviews.')# These summaries will be the contextual information that ChatGPT will use to provide the final conclusions.')
# st.write("(made with HuggingFace transformers)")
# col1.image('images/hf-logo.png', width=100)
with st.expander("🤗Show HuggingFace model details", expanded=False):
    # col1, col2 = st.columns([.3, .7])
    # col2.markdown("##### HuggingFace Model Details")
    st.write(summaries['model-info'])






# Display low and high reviews summaries
review_container = st.container()
col1, col2 = review_container.columns(2)
col1.subheader("Low Reviews") 
col1.markdown("*" + summaries['summary-low'].strip() + "*")
col2.subheader("High Reviews")
col2.markdown("*" + summaries['summary-high'].strip() + "*")


# st.divider()

st.divider()

# Load vector database
if os.path.exists(fpath_db):
    retriever = fn.load_vector_database(fpath_db, fpath_llm_csv, use_previous=True, as_retriever=True)
else:
    retriever = fn.load_vector_database(fpath_db, fpath_llm_csv, use_previous=False, as_retriever=True)

# Set up AI Recommendations section
st.header("ChatGPT Recommendations / Q&A")
# st.divider()
chat_container = st.container(border=True)
output_container = chat_container.container(border=False)
menu_container = chat_container.container(border=True)
with menu_container:
    # with st.container(border=True):
    st.markdown("***Select a task or enter your question in the text box below to get answers.***")

    col1, col2 = st.columns(2)
    
    button_product_recs = col1.button('Get Product Recommendations')
    button_marketing_recs = col2.button('Get Marketing Recommendations')
    
    if button_product_recs:
        prompt_text = '**Product Recommendations:** Provide a list of 3-5 actionable business recommendations on how to improve the product to address review feedback.'
    
    if button_marketing_recs:
        prompt_text= '**Marketing Recommendations:** Provide a list of 3-5 recommendations for the marketing team to better set customer expectations before purchasing the product or to better target the customers who will enjoy it.'
    
    
    
user_text = menu_container.chat_input(placeholder="Type your question here.")
factory = fn.AgentFactory()
# chat_container.header("Q&A")


# Reset agent function
def reset_agent(retriever, starter_message="Hello, there! Select one of the options below or enter your question and I will check the full reviews database to provide you the best answer.", get_agent_kws={}):
    
    agent_exec = factory.get_agent(retriever=retriever, **get_agent_kws)

    if starter_message is not None:
        agent_exec.memory.chat_memory.add_ai_message(starter_message)
        
    return agent_exec


# if 'agent-summarize' not in st.session_state:
get_agent_kws = dict(template_string_func=lambda: factory.get_template_string_interpret(context_low=summaries['summary-low'], context_high=summaries['summary-high']))
if 'agent' not in st.session_state:    
    # st.session_state['agent-summarize'] = 
    st.session_state['agent'] = reset_agent(retriever=retriever, get_agent_kws=get_agent_kws)#factory.get_agent(retriever=retriever, template_string_func=lambda: factory.get_template_string_interpret(context_low=summaries['summary-low'], context_high=summaries['summary-high']))
    
if 'chat-history' not in st.session_state:
    st.session_state['chat-history'] = []

# Display task options
task_options = fn.get_task_options(options_only=False)

with open("app-assets/author-info.md") as f:
    author_info = f.read()

with st.sidebar.container(border=True):
    st.subheader("Author Information")
    st.markdown(author_info, unsafe_allow_html=True)

# st.sidebar.divider()        
    


def get_response(user_text, agent_key='agent-summarize', combined_memory = None):
    
    st.chat_message("user", avatar=user_avatar).write(user_text)
    response = st.session_state[agent_key].invoke({"input": user_text})
    st.chat_message('assistant', avatar=ai_avatar).write(fn.fake_streaming(response['output']))
    
    if combined_memory is not None:
        combined_memory.append( HumanMessage(user_text))
        combined_memory.append( AIMessage(response['output']))
    return response

    
def display_history(chat_history, user_avatar="💬", ai_avatar="🤖"):
    session_state_messages = chat_history
    for message in session_state_messages:#[:-1]:
        if isinstance(message, SystemMessage):
            # st.chat_message(f" : {message.text}")
            continue
        elif isinstance(message, HumanMessage):
            st.chat_message('user', avatar=user_avatar).write(message.content)
            
        elif isinstance(message, AIMessage):
            st.chat_message("assistant",avatar=ai_avatar).write(message.content)
        else:
            # st.write(f"Unknown message type: {message}")
            continue
        

        

def download_history(chat_history, filename="chat-history.md"):
        
    avatar_dict = {'human': user_avatar,
                   'ai':ai_avatar,
                   'SystemMessage':"💻"}
    
    md_history = []
    # history = st.session_state['llm'].memory.buffer_as_messages
    history=chat_history
    for msg in history:
        type_message = msg.type#type(msg) x
            # with st.chat_message(name=i["role"],avatar=avatar_dict[ i['role']]):
        md_history.append(f"{avatar_dict[type_message]}: {msg.content}")
    return "\n\n".join(md_history)
    
# st.divider()
st.markdown("> ***Reveal the sidebar (`>`) to reset chat history or download chat history as a markdown file.***")

reset_container = st.sidebar.container(border=True)
reset_container.markdown("#### *Click below to reset chat history:*")
reset_chat = reset_container.button("🧹 Reset Chat?")
download_container =  st.sidebar.container(border=True)
download_container.markdown("#### *Download chat as markdown file:*")
# admin_container.markdown("### Admin Options")
# admin_col1, admin_col2 = admin_container.columns(2)
md_filename = download_container.text_input("Enter filename for chat history", value="chat-history.md")
download_chat = download_container.download_button("⤵️ Download chat history.", file_name=md_filename,
                   data=download_history(st.session_state['agent'].memory.buffer_as_messages))
# download_chat = admin_container.button("⤵️ Download Chat History")

if reset_chat:
    with output_container:
        st.session_state['chat-history'] = []
        # st.session_state['agent'] = reset_agent(retriever=retriever)
        # st.session_state['agent-summarize'] = #
        st.session_state['agent'] = reset_agent(retriever=retriever, get_agent_kws=get_agent_kws)
  
                                                            

    
with output_container:
    display_history(st.session_state['agent'].memory.buffer_as_messages,#st.session_state['chat-history'],
                    user_avatar=user_avatar, ai_avatar=ai_avatar)

    if button_product_recs or button_marketing_recs:
        # output_container.chat_message("user", avatar=user_avatar).write(prompt_text)

        response = get_response(prompt_text, agent_key='agent',#'agent-summarize',
                                # combined_memory=st.session_state['chat-history']
                                )   
        
    if user_text:
        # output_container.chat_message("user", avatar=user_avatar).write(user_text)
        response = get_response(user_text, agent_key='agent', combined_memory=None)#st.session_state['chat-history'])

