ai_avatar ="🤖"
user_avatar="💬"
## Adding caching to reduce api usage
import os
import joblib, json
from langchain.cache import InMemoryCache
# from langchain.document_loaders import CSVLoader
from langchain_community.document_loaders import CSVLoader
# from langchain.globals import set_llm_cache
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate, PromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
# import streamlit as st
# import custom_functions as fn
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.callbacks import StdOutCallbackHandler
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import MessagesPlaceholder
# from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import CSVLoader

from langchain.agents import AgentExecutor, create_openai_tools_agent
# Memory: agent token buffer used in original example blog post
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.memory import ConversationBufferMemory
# set_llm_cache(InMemoryCache())

from langchain import hub
# from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool


    
from functools import wraps
import joblib, json, os
import pandas as pd

class AgentFactory():
    """
    A factory class for creating agents and loading data for Amazon reviews NLP analysis.
    """

    def __str__(self):
        msg = ">> AgentFactory class for creating LangChain AgentExecutor & loading Amazon reviews NLP analysis."
        # print(msg)
        return msg
    def __repr__(self):
        return self.__str__()
    
    def __init__(self, filepath_json="config/filepaths.json"):
        """
        Initializes the AgentFactory object.

        Parameters:
        - filepath_json (str): The filepath to the JSON file containing filepaths for data.

        Returns:
        - None
        """
        self.FPATHS = self.load_filepaths_json(filepath_json)

    def load_filepaths_json(cls, fname="config/filepaths.json", verbose=False):
        """
        Loads the filepaths from a JSON file.

        Parameters:
        - fname (str): The filepath to the JSON file.
        - verbose (bool): Whether to print the top-level keys in the loaded JSON.

        Returns:
        - dict: A dictionary containing the filepaths.
        """
        with open(fname) as f:
            FPATHS = json.load(f)
        if verbose:
            print("Top-Level Keys in FPATHS dict:")
            print(FPATHS.keys())
        return FPATHS

    def load_product_info(cls, fpath):
        """
        Loads the product information from a JSON file.

        Parameters:
        - fpath (str): The filepath to the JSON file.

        Returns:
        - str: A string containing the product information.
        """
        with open(fpath, 'r') as f:
            product_json = json.load(f)

        product_string = "Product Info:\n"
        for k, v in product_json.items():
            if k.lower() == 'description':
                continue
            product_string += f"\n{k} = {v}\n"

        return product_string

    def load_df(cls, fpath):
        """
        Loads a dataframe from a file.

        Parameters:
        - fpath (str): The filepath to the file.

        Returns:
        - DataFrame: The loaded dataframe.
        """
        return joblib.load(fpath)

    def load_metadata(cls, fpath):
        """
        Loads metadata from a JSON file into a dataframe.

        Parameters:
        - fpath (str): The filepath to the JSON file.

        Returns:
        - DataFrame: The loaded metadata dataframe.
        """
        return pd.read_json(fpath)

    def load_summaries(cls, fpath):
        """
        Loads summaries from a JSON file.

        Parameters:
        - fpath (str): The filepath to the JSON file.

        Returns:
        - dict: A dictionary containing the loaded summaries.
        """
        with open(fpath) as f:
            summaries = json.load(f)
        return summaries

    def get_task_options(cls, options_only=False):
        """
        Gets the task options for the agent.

        Parameters:
        - options_only (bool): Whether to return only the task options or the full dictionary.

        Returns:
        - list or dict: The task options.
        """
        task_prompt_dict = {
            'Product Recommendations': 'Provide a list of 3-5 actionable business recommendations on how to improve the product.',
            'Marketing Recommendations': 'Provide a list of 3-5 recommendations for the marketing team to on how to better set customer expectations before purchasing the product or to better target the customers who will enjoy it.'
        }
        if options_only:
            return list(task_prompt_dict.keys())
        else:
            return task_prompt_dict

    def get_template_string_reviews(cls):
        """
        Gets the template string for answering questions about customer reviews.

        Returns:
        - str: The template string.
        """
        template = f"You are a helpful data analyst for answering questions about what customers said about a specific  Amazon product using only content from use reviews."
        product_string = cls.load_product_info(cls.FPATHS['data']['app']['product-metadata-llm_json'])
        product_template = f" Assume all user questions are asking about the content in the user reviews. Note the product metadata is:\n```{product_string}```\n\n"
        template += product_template
        qa_prompt_template = """Use the following pieces of context (user reviews) to answer the user's question by summarizing the reviews. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n{agent_scratchpad}\n\n"""
        template += qa_prompt_template
        return template

    def get_template_string_interpret(cls, context_low, context_high, context_type='BERT-summary'):
        """
        Gets the template string for interpreting user questions.

        Parameters:
        - context_low (str): The context of 1-star reviews.
        - context_high (str): The context of 5-star reviews.
        - context_type (str): The type of context (default: 'BERT-summary').

        Returns:
        - str: The template string.
        """
        template_starter = cls.get_template_string_reviews()
        context = f"\nGroup Contexts:\n Here is a {context_type} of 1-star reviews: ```{context_low}```.\n\n Here is a {context_type} of 5-star reviews:```{context_high}."
        context += f"Use the {context_type} first before using the retrieved documents."
        template_assistant = template_starter + context
        return template_assistant

    def get_agent(cls, retriever=None, fpath_db=None, fpath_llm_csv=None, k=8, temperature=0.1, verbose=False, template_string_func=None):
        """
        Gets the agent for NLP analysis.

        Parameters:
        - retriever: The retriever object.
        - fpath_db (str): The filepath to the vector database.
        - fpath_llm_csv (str): The filepath to the CSV file containing reviews with target for LLM.
        - k (int): The number of retrievals.
        - temperature (float): The temperature for the ChatOpenAI model.
        - verbose (bool): Whether to print verbose output.
        - template_string_func: The function to generate the template string.

        Returns:
        - AgentExecutor: The agent executor object.
        """
        if template_string_func is None:
            template_string_func = cls.get_template_string_reviews

        if fpath_db is None:
            fpath_db = cls.FPATHS['data']['app']['vector-db_dir']

        if fpath_llm_csv is None:
            fpath_llm_csv = cls.FPATHS['data']['app']['reviews-with-target-for-llm_csv']

        if retriever is None:
            retriever = cls.load_vector_database(fpath_db, fpath_llm_csv, k=k, use_previous=True, as_retriever=True)

        tool = create_retriever_tool(
            retriever,
            "search_reviews",
            "Search Amazon custom reviews for relevant information."
        )        
        cls.tools = [tool]

        template = template_string_func()

        prompt_template = OpenAIFunctionsAgent.create_prompt(
            system_message=SystemMessage(template),
            extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
        )

        llm = ChatOpenAI(temperature=temperature, api_key=os.getenv("OPENAI_API_KEY"))
        cls.llm = llm
        
        agent = create_openai_tools_agent(cls.llm, cls.tools, prompt_template)
        cls.agent = agent
        
        agent_executor = AgentExecutor(agent=cls.agent, tools=cls.tools, verbose=True, memory=ConversationBufferMemory(memory_key="history", return_messages=True))
        cls.executor = agent_executor
        return agent_executor

    def load_vector_database(cls, fpath_db, fpath_csv=None, metadata_columns=['reviewerID'], chunk_size=500, use_previous=False, as_retriever=False, k=8, **retriever_kwargs):
        """
        Loads the vector database for retrieval.

        Parameters:
        - fpath_db (str): The filepath to the vector database.
        - fpath_csv (str): The filepath to the CSV file containing the data.
        - metadata_columns (list): The columns to use as metadata.
        - chunk_size (int): The chunk size for splitting the documents.
        - use_previous (bool): Whether to use the previous vector database.
        - as_retriever (bool): Whether to return the retriever object.
        - k (int): The number of retrievals.
        - retriever_kwargs: Additional keyword arguments for the retriever.

        Returns:
        - FAISS or Retriever: The loaded vector database or retriever object.
        """
        embedding_func = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))

        if use_previous == True:
            print("Using previous vector db...")
            db = FAISS.load_local(fpath_db, embedding_func)
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



# # @classmethod
# def print_history(cls, agent_executor):
#     session_state_messages = agent_executor.memory.buffer_as_messages
#     for msg in session_state_messages:
#         if isinstance(msg, AIMessage):
#             print(f"Assistant: {msg.content}")
#         elif isinstance(msg, HumanMessage):
#             print(f"User: {msg.content}")
#         print()

# # @classmethod
# def display_metadata(cls, meta_df, iloc=0):
#     product = meta_df.iloc[iloc]
#     md = ""
#     md += f'\n- Product Title:\n***\"{product["Title (Raw)"]}\"***'
#     md += f'\n- Brand: {product["Brand"]}'
#     md += f"\n- Price: {product['Price']}"
#     md += f"\n- Ranked {product['Rank']} (2018)"
#     md += f"\n- Categories:\n    - "
#     md += "; ".join(product['Categories'])
#     return md
