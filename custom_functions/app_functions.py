# @st.cache_data

## Adding caching to reduce api usage
from langchain.cache import InMemoryCache
from langchain.document_loaders import CSVLoader
from langchain.globals import set_llm_cache
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate, PromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.text_splitter import CharacterTextSplitter#, SpacyTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings

# set_llm_cache(InMemoryCache())

from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool

def display_metadata(meta_df,iloc=0, include_details=False):
    # product = meta_df.iloc[iloc]
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

def load_vector_database(fpath_db, fpath_csv=None, metadata_columns = ['reviewerID'],
                         chunk_size=500, use_previous = True,
                         delete=False, as_retriever=False, k=8, **retriever_kwargs):
    
     # Use EMbedding --> embed chunks --> vectors
    embedding_func = OpenAIEmbeddings()
    
    if delete==True:
        # Set use_pervious to False
        use_previous= False
        db = Chroma(persist_directory=fpath_db, 
           embedding_function=embedding_func)
        db.delete_collection()

    if use_previous==True:
        db =  Chroma(persist_directory=fpath_db, 
           embedding_function=embedding_func)
    else:
        if fpath_csv == None:
            raise Exception("Must pass fpath_csv if use_previous==False or delete==True")
                
        # Load Document --> Split into chunks
        loader = CSVLoader(fpath_csv,metadata_columns=metadata_columns)
        documents = loader.load()
        
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size)
        docs = text_splitter.split_documents(documents)
        
        db = Chroma.from_documents(docs, embedding_func, persist_directory= fpath_db)
        # Use persist to save to disk
        db.persist()

    if as_retriever:
        return db.as_retriever(k=k, **retriever_kwargs)
    else:
        return db

    
    