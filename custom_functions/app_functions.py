import os
from langchain.cache import InMemoryCache
from langchain_community.document_loaders import CSVLoader
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.callbacks import StdOutCallbackHandler
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import MessagesPlaceholder
from langchain_community.document_loaders import CSVLoader
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.memory import ConversationBufferMemory
from langchain import hub
from langchain.tools.retriever import create_retriever_tool
import streamlit as st
import json
import json
import joblib
import pandas as pd
import json
import time
import os

ai_avatar = "🤖"
user_avatar = "💬"

## Adding caching to reduce API usage
from langchain.prompts import (
    ChatPromptTemplate, PromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)


def load_filepaths_json(fname="config/filepaths.json", verbose=False):
    """
    Load file paths from a JSON file.

    Parameters:
    - fname (str): The file path of the JSON file.
    - verbose (bool): Whether to print the top-level keys in the loaded dictionary.

    Returns:
    - FPATHS (dict): The loaded file paths dictionary.
    """
    with open(fname) as f:
        FPATHS = json.load(f)
    if verbose:
        print("Top-Level Keys in FPATHS dict:")
        print(FPATHS.keys())
    return FPATHS


def load_product_info(fpath):
    """
    Load product information from a JSON file.

    Parameters:
    - fpath (str): The file path of the JSON file.

    Returns:
    - product_string (str): The formatted product information string.
    """
    with open(fpath, 'r') as f:
        product_json = json.load(f)

    product_string = "Product Info:\n"
    for k, v in product_json.items():
        if k.lower() == 'description':
            continue
        product_string += f"\n{k} = {v}\n"

    return product_string


def load_df(fpath):
    """
    Load a DataFrame from a file.

    Parameters:
    - fpath (str): The file path of the DataFrame file.

    Returns:
    - df (DataFrame): The loaded DataFrame.
    """
    return joblib.load(fpath)


def load_metadata(fpath):
    """
    Load metadata from a JSON file.

    Parameters:
    - fpath (str): The file path of the JSON file.

    Returns:
    - metadata (DataFrame): The loaded metadata DataFrame.
    """
    return pd.read_json(fpath)


def load_summaries(fpath):
    """
    Load summaries from a JSON file.

    Parameters:
    - fpath (str): The file path of the JSON file.

    Returns:
    - summaries (dict): The loaded summaries dictionary.
    """
    with open(fpath) as f:
        summaries = json.load(f)
    return summaries


def fake_streaming(response):
    """
    Simulate streaming by yielding words from a response with a delay.

    Parameters:
    - response (str): The response string.

    Yields:
    - word (str): The next word in the response.
    """
    for word in response.split(" "):
        yield word + " "
        time.sleep(.05)


def get_task_options(options_only=False):
    """
    Get the available task options.

    Parameters:
    - options_only (bool): Whether to return only the task options or the full task prompt dictionary.

    Returns:
    - task_options (list or dict): The task options or the full task prompt dictionary.
    """
    task_prompt_dict = {
        'Product Recommendations': 'Provide a list of 3-5 actionable business recommendations on how to improve the product.',
        'Marketing Recommendations': 'Provide a list of 3-5 recommendations for the marketing team to better set customer expectations before purchasing the product or to better target the customers who will enjoy it.'
    }
    if options_only:
        return list(task_prompt_dict.keys())
    else:
        return task_prompt_dict


def get_template_string_reviews():
    """
    Get the template string for answering user questions about customer reviews.

    Returns:
    - template (str): The template string.
    """
    template = "You are a helpful data analyst for answering questions about what customers said about a specific Amazon product using only content from user reviews."
    product_string = load_product_info(FPATHS['data']['app']['product-metadata-llm_json'])
    product_template = f" Assume all user questions are asking about the content in the user reviews. Note the product metadata is:\n```{product_string}```\n\n"
    template += product_template
    qa_prompt_template = """Use the following pieces of context (user reviews) to answer the user's question by summarizing the reviews. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n{agent_scratchpad}\n\n"""
    template += qa_prompt_template
    return template


def get_template_string_interpret(context_low, context_high, context_type='BERT-summary'):
    """
    Get the template string for interpreting user questions based on context.

    Parameters:
    - context_low (str): The context of 1-star reviews.
    - context_high (str): The context of 5-star reviews.
    - context_type (str): The type of context (default is 'BERT-summary').

    Returns:
    - template_assistant (str): The template string for the assistant.
    """
    template_starter = get_template_string_reviews()
    context = f"\nGroup Contexts:\n Here is a {context_type} of 1-star reviews: ```{context_low}```.\n\n Here is a {context_type} of 5-star reviews:```{context_high}."
    context += f"Use the {context_type} first before using the retrieved documents."
    template_assistant = template_starter + context
    return template_assistant


def print_history(agent_executor=None, combined_memory=None, streamlit=True):
    """
    Print the conversation history.

    Parameters:
    - agent_executor (AgentExecutor): The agent executor object.
    """
    if agent_executor is not None:
        session_state_messages = agent_executor.memory.buffer_as_messages
    elif combined_memory is not None:
        session_state_messages = combined_memory
    else:
        raise Exception("Either agent_executor or combined_memory must be provided.")
        
    for msg in session_state_messages:
        if isinstance(msg, AIMessage):
            print(f"Assistant: {msg.content}")
        elif isinstance(msg, HumanMessage):
            print(f"User: {msg.content}")
        print()


def display_metadata(meta_df, iloc=0):
    """
    Display metadata information.

    Parameters:
    - meta_df (DataFrame): The metadata DataFrame.
    - iloc (int): The index location of the metadata to display.

    Returns:
    - md (str): The formatted metadata string.
    """
    product = meta_df.iloc[iloc]
    md = ""
    md += f'\n- Product Title:\n***\"{product["Title (Raw)"]}\"***'
    md += f'\n- Brand: {product["Brand"]}'
    md += f"\n- Price: {product['Price']}"
    md += f"\n- Ranked {product['Rank']} (2018)"
    md += f"\n- Categories:\n    - "
    md += "; ".join(product['Categories'])
    return md


def load_vector_database(fpath_db, fpath_csv=None, metadata_columns=['reviewerID'],
                         chunk_size=500, use_previous=False,
                         as_retriever=False, k=8, **retriever_kwargs):
    """
    Loads or creates a vector database for document embeddings.

    Parameters:
    - fpath_db (str): The file path to the vector database.
    - fpath_csv (str, optional): The file path to the CSV file containing the documents. Required if use_previous is False or delete is True.
    - metadata_columns (list, optional): The list of column names in the CSV file to be used as metadata for the documents. Default is ['reviewerID'].
    - chunk_size (int, optional): The size of each chunk to split the documents into. Default is 500.
    - use_previous (bool, optional): Whether to use the previous vector database if it exists. Default is False.
    - as_retriever (bool, optional): Whether to return the vector database as a retriever. Default is False.
    - k (int, optional): The number of nearest neighbors to retrieve. Required if as_retriever is True.
    - **retriever_kwargs (optional): Additional keyword arguments to be passed to the retriever.

    Returns:
    - db (object): The vector database.

    Raises:
    - Exception: If fpath_csv is not provided when use_previous is False or delete is True.
    """
    embedding_func = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))

    if use_previous == True:
        print("Using previous vector db...")
        db = FAISS.load_local(fpath_db, embedding_func, allow_dangerous_deserialization=True)
    else:
        print("Creating embeddings/Chromadb database")
        if fpath_csv == None:
            raise Exception("Must pass fpath_csv if use_previous==False or delete==True")

        loader = CSVLoader(fpath_csv, metadata_columns=metadata_columns)
        documents = loader.load()

        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size)
        docs = text_splitter.split_documents(documents)

        db = FAISS.from_documents(docs, embedding_func)
        db.save_local(fpath_db)

    if as_retriever:
        return db.as_retriever(k=k, **retriever_kwargs)
    else:
        return db
