# @st.cache_data
def load_filepaths_json(fname="config/filepaths.json"):
    ##Load in the data
    import json
    with open(fname) as f:
        FPATHS = json.load(f)
    print("Top-Level Keys in FPATHS dict:")
    # [print(f'- {k}') for k in FPATHS.keys()]
    print(FPATHS.keys())
    return FPATHS


FPATHS = load_filepaths_json()
## Adding caching to reduce api usage
import os
from langchain.cache import InMemoryCache
# from langchain.document_loaders import CSVLoader
from langchain_community.document_loaders import CSVLoader
from langchain.globals import set_llm_cache
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate, PromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
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
# set_llm_cache(InMemoryCache())

from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool


def get_template_string_reviews(FPATHS):
     # Create template with product info
    template = f"You are a helpful data analyst for answering questions about what customers said about a specific  Amazon product using only content from use reviews."
    from custom_functions.app_functions import load_product_info
    product_string = load_product_info(FPATHS['data']['app']['product-metadata-llm_json'])

    product_template = f" Assume all user questions are asking about the content in the user reviews. Note the product metadata is:\n```{product_string}```\n\n"
    template+=product_template
    
    qa_prompt_template ="""Use the following pieces of context (user reviews) to answer the user's question by summarizing the reviews. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n{agent_scratchpad}\n\n"""
    template+=qa_prompt_template
    return template
    
    


def get_agent(retriever=None,fpath_db=FPATHS['data']['app']['vector-db_dir'],
              ## Updated function
              fpath_csv = FPATHS['data']['app']['reviews-with-target-for-llm_csv'],
              k=8, temperature=0.1, verbose=False,
             template_string_func=get_template_string_reviews):
    
    ## Make retreieval tool
    if retriever is None:
        retriever  = load_vector_database( fpath_db,fpath_csv, k=k, use_previous=True, as_retriever=True)#, use_previous=False)
    tool = create_retriever_tool(
        retriever,
        "search_reviews",
        "Searches and returns excerpts from Amazon user reviews.",
    )
    tools = [tool]
    
    
   ## Get template via function for template string
    template = template_string_func()


    # Create the chatprompttemplate
    prompt_template = OpenAIFunctionsAgent.create_prompt(
        system_message=SystemMessage(template),
        extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
    )
    
    if verbose:
        print(prompt_template.messages)
        
    llm = ChatOpenAI(temperature=temperature,streaming=True, api_key=os.getenv("OPENAI_API_KEY"))
    agent = create_openai_tools_agent(llm, tools, prompt_template)
    
    ## Creating streamlit-friendly memory for streaming
    agent_executor = AgentExecutor(agent=agent, tools=tools,  verbose=True, #return_intermediate_steps=True,
                                   memory=ConversationBufferMemory(memory_key="history",return_messages=True))
    return agent_executor


            
            
def reset_agent(#fpath_db = FPATHS['data']['app']['vector-db_dir'],
                retriever=None,
                ai_avatar="🤖", user_avatar = "💬",
                starter_message = "Hello, there! Enter your question here and I will check the full reviews database to provide you the best answer.",
               get_agent_kws={}):
    # fpath_db
    if retriever is None:
        retriever = st.session_state['retriever']
    agent_exec = get_agent(retriever, **get_agent_kws)
    agent_exec.memory.chat_memory.add_ai_message(starter_message)
    # with chat_container:
    st.chat_message("assistant", avatar=ai_avatar).write_stream(fake_streaming(starter_message))
        # print_history(agent_exec)
    return agent_exec
    

def fake_streaming(response):
    import time
    for word in response.split(" "):
        yield word + " "
        time.sleep(.05)		
        
            
    
## For steramlit try this as raw code, not a function
def print_history(agent_executor, ai_avatar ="🤖", user_avatar="💬"):
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
        

def display_metadata(meta_df,iloc=0, include_details=False):
    product = meta_df.iloc[iloc]
    # md = "#### Product Being Reviewed"
    md = ""
    md += f'\n- Product Title:\n***\"{product["Title (Raw)"]}\"***'
    # md += f"<p><img src='{product['Product Image']}' width=300px></p>"
    md += f'\n- Brand: {product["Brand"]}'
    md += f"\n- Price: {product['Price']}"
    md += f"\n- Ranked {product['Rank']} (2018)"

    md += f"\n- Categories:\n    - "
    md += "; ".join(product['Categories'])
    # md += 
    # md += f"\n- Categories:{', '.join(product['Categories'])}"
    
    
    return md


def load_product_info(fpath):
    import json
    with open(fpath,'r') as f:
        product_json = json.load(f)
        
    product_string = "Product Info:\n"
    for k,v in product_json.items():
        if k.lower()=='description':
            continue
        product_string+=f"\n{k} = {v}\n"
        
    return product_string


@st.cache_resource
def load_vector_database(fpath_db, fpath_csv=None, metadata_columns = ['reviewerID'],
                         chunk_size=500, use_previous = False,
                        #  delete=False, 
                         as_retriever=False, k=8, **retriever_kwargs):
    import os
     # Use EMbedding --> embed chunks --> vectors
    embedding_func = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
    
    # if delete==True:
    #     # print("Deleting previous Chroma db...")
    #     # Set use_pervious to False
    #     use_previous= False
    #     # db = Chroma(persist_directory=fpath_db, 
    #     #    embedding_function=embedding_func)
    #     # db.delete_collection()

    if use_previous==True:
        print("Using previous vector db...")
        db = FAISS.load_local(fpath_db, embedding_func)

    else:
        print("Creating embeddings/Chromadb database")
        if fpath_csv == None:
            raise Exception("Must pass fpath_csv if use_previous==False or delete==True")
                
        # Load Document --> Split into chunks
        loader = CSVLoader(fpath_csv,metadata_columns=metadata_columns)
        documents = loader.load()
        
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size)
        docs = text_splitter.split_documents(documents)
        
        # db = Chroma.from_documents(docs, embedding_func, persist_directory= fpath_db)
        db = FAISS.from_documents(docs, embedding_func)
        # Use persist to save to disk
        # db.persist()
        db.save_local(fpath_db)

    if as_retriever:
        return db.as_retriever(k=k, **retriever_kwargs)
    else:
        return db
    
    