import os
import json
# from langchain.cache import InMemoryCache
from langchain_community.document_loaders import CSVLoader
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate, PromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import MessagesPlaceholder
# from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.memory import ConversationBufferMemory
# from langchain import hub
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent
from langchain.chains.base import Chain


class AgentWrapper(AgentExecutor):
    def __init__(self, retriever=None, fpath_db=None, fpath_llm_csv=None, k=8, temperature=0.1, verbose=False, template_string_func=None,
                 FPATHS_json=None, callbacks=None, **kwargs):
        
        # self.ai_avatar = "🤖"
        # self.user_avatar = "💬"
        if FPATHS_json is None:
            FPATHS_json = "config/filepaths.json"
        self.FPATHS = self.load_filepaths_json(FPATHS_json)
                
        if fpath_db is None:
            fpath_db = self.FPATHS['data']['app']['vector-db_dir']
        if fpath_llm_csv is None:  
            fpath_llm_csv = self.FPATHS['data']['app']['reviews-with-target-for-llm_csv']
                
        
        if template_string_func is None:
            template_string_func = self.get_template_string_reviews

        # if fpath_db is None or fpath_llm_csv is None:
        #     FPATHS = self.load_filepaths_json()
        #     fpath_db = fpath_db or FPATHS['data']['app']['vector-db_dir']
        #     fpath_llm_csv = fpath_llm_csv or FPATHS['data']['app']['reviews-with-target-for-llm_csv']

        if retriever is None:
            retriever = self.load_vector_database(fpath_db, fpath_llm_csv, k=k, use_previous=True, as_retriever=True)

        tool = create_retriever_tool(
            retriever,
            "search_reviews",
            "Search Amazon custom reviews for relevant information."
        )
        tools = [tool]

        template = template_string_func()
        prompt_template = OpenAIFunctionsAgent.create_prompt(
            system_message=SystemMessage(template),
            extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
        )

        if verbose:
            print(prompt_template.messages)

        llm = ChatOpenAI(temperature=temperature, api_key=os.getenv("OPENAI_API_KEY"))
        agent = create_openai_tools_agent(llm, tools, prompt_template)
        # self.agent = agent
        # super().__init__(agent=agent, tools=tools, verbose=True,
        #                     memory=ConversationBufferMemory(memory_key="history", return_messages=True)
        #                     )        
        super().__init__(agent=agent, tools=tools, callbacks=callbacks, **kwargs)


        # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,
                                    #    memory=ConversationBufferMemory(memory_key="history", return_messages=True))
        # self.agent_executor = agent_executor
        # super().__init__(self, agent=agent, tools=tools, verbose=True,memory=ConversationBufferMemory(memory_key="history", return_messages=True))
        
    # @staticmethod
    def load_filepaths_json(self, fname="config/filepaths.json", verbose=False):
        with open(fname) as f:
            FPATHS = json.load(f)
        if verbose:
            print("Top-Level Keys in FPATHS dict:")
            print(FPATHS.keys())
        return FPATHS

    def load_product_info(self, fpath=None):
        if fpath is None:
            FPATHS = self.FPATHS
            fpath = FPATHS['data']['app']['product-metadata-llm_json']
            
        with open(fpath, 'r') as f:
            product_json = json.load(f)

        product_string = "Product Info:\n"
        for k, v in product_json.items():
            if k.lower() == 'description':
                continue
            product_string += f"\n{k} = {v}\n"

        return product_string

    def get_template_string_reviews(self, fpath=None):
        if fpath  is None:
            FPATHS = self.FPATHS
            fpath = FPATHS['data']['app']['product-metadata-llm_json']
        product_string = self.load_product_info(fpath)

        template = f"You are a helpful data analyst for answering questions about what customers said about a specific Amazon product using only content from use reviews."
        product_template = f" Assume all user questions are asking about the content in the user reviews. Note the product metadata is:\n```{product_string}```\n\n"
        template += product_template

        qa_prompt_template = """Use the following pieces of context (user reviews) to answer the user's question by summarizing the reviews. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n{agent_scratchpad}\n\n"""
        template += qa_prompt_template
        return template

    def get_template_string_interpret(self, context_low, context_high, context_type='BART-summary'):
        template_starter = self.get_template_string_reviews()
        context = f"\nGroup Contexts:\n Here is a {context_type} of 1-star reviews: ```{context_low}```.\n\n Here is a {context_type} of 5-star reviews:```{context_high}."
        context += f" Use the {context_type} first before using the retrieved documents."
        template_assistant = template_starter + context
        return template_assistant


    def load_vector_database(self, fpath_db, fpath_csv=None, metadata_columns=['reviewerID'], chunk_size=500, use_previous=False, as_retriever=False, k=8, **retriever_kwargs):
        embedding_func = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))

        if use_previous:
            print("Using previous vector db...")
            db = FAISS.load_local(fpath_db, embedding_func)
        else:
            print("Creating embeddings/Chromadb database")
            if fpath_csv is None:
                raise Exception("Must pass fpath_csv if use_previous==False")

            loader = CSVLoader(fpath_csv, metadata_columns=metadata_columns)
            documents = loader.load()

            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size)
            docs = text_splitter.split_documents(documents)

            db = FAISS.from_documents(docs, embedding_func)
            db.save_local(fpath_db)

        self._vector_db= db
        self.retriever = db.as_retriever(k=k, **retriever_kwargs)
        
        if as_retriever:
            return db.as_retriever(k=k, **retriever_kwargs)
        else:
            return db
