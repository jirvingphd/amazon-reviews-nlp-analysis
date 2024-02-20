import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from PIL import Image
import custom_functions as fn
import plotly.express as px
import plotly.io as pio
pio.templates.default='streamlit'

import streamlit as st
import custom_functions as fn
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.callbacks import StdOutCallbackHandler
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import CSVLoader

# from langchain.agents.agent_toolkits import create_pandas_dataframe_agent


from langchain.agents import AgentExecutor, create_openai_tools_agent
# Memory: agent token buffer used in original example blog post
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.memory import ConversationBufferMemory

# Changing the Layout
st.set_page_config( #layout="wide", 
                   page_icon="🖼️Amazon Reviews NLP EDA")



# Get Fpaths
@st.cache_data
def get_app_fpaths(fpath='config/filepaths.json'):
	import json
	with open(fpath ) as f:
		return json.load(f)



##Load in the data
import json
with open("config/filepaths.json") as f:
    FPATHS = json.load(f)
    
# FPATHS = fn.load_filepaths_json()
fpath_llm_csv = FPATHS['data']['app']['reviews-with-target-for-llm_csv']
fpath_db = FPATHS['data']['app']['vector-db_dir']

        
        

if 'retriever' not in st.session_state:
    # retriever  = load_vector_database( fpath_db,fpath_llm_csv, k=8, use_previous=False, as_retriever=True)    
    if os.path.exists(fpath_db):
        retriever = fn.load_vector_database(fpath_db, fpath_llm_csv, use_previous=True, as_retriever=True)
    else:
        retriever = fn.load_vector_database(fpath_db, fpath_llm_csv, use_previous=False, as_retriever=True)
    
    st.session_state['retriever'] = retriever



# Create chat container early
    
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
    
## Title /header
# st.header("Exploratory Data Analysis of Amazon Reviews ")
# st.divider()
st.header("NLP Exploratory Visualizations")

# Setting sidebar controls ahead of time
### TEXT SELECTION OPTIONS FOR SIDEBAR
## Select which text column/preprocessing
# st.sidebar.header("Display Options")
# st.sidebar.markdown(">*Select visualizations for main pane.*")
# show_review_graphs = st.sidebar.checkbox("Show rating distibutions.", value=True)
# show_yearly =st.sidebar.checkbox('Show yearly trends in reviews', value=True)
# show_wordclouds = st.sidebar.checkbox('Show Word Couds', value=True)
# show_scattertext = st.sidebar.checkbox("Show ScatterText Visual", value=False)
# st.sidebar.divider()
    

    
@st.cache_data    
def load_df(fpath):
    import joblib
    return joblib.load(fpath)

@st.cache_data
def load_metadata(fpath):
    import pandas as pd
    return pd.read_json(fpath)

@st.cache_data
def get_rating_percent_by_year(df,**kwargs):
    return fn.get_rating_percent_by_year(df,**kwargs)

@st.cache_data
def get_average_rating_by_year(df, **kwargs):
    return fn.get_average_rating_by_year(df,**kwargs)

df = load_df(FPATHS['data']['processed-nlp']['processed-reviews-with-target_joblib'])
meta_df = load_metadata(FPATHS['data']['app']['product-metadata_json'])
product= meta_df.iloc[0]




# st.image(FPATHS['images']['banner_png'],width=700,use_column_width='always')
# st.divider()
## Product metasata
# st.markdown("##### ***👈 Select the Display Options to enable/disable app components.***")
# st.divider()
# st.subheader("Exploratory Analysis ")
st.divider()
show_product= st.checkbox("Show Product Information", value=False)

if show_product==True:
    st.subheader("Product Information")


    # st.markdown(f'Product Title: ***{product["Title (Raw)"]}***')
    # st.divider()
    col1,col2 = st.columns(2)

    # @st.cache_data
    def display_metadata(meta_df,iloc=0):
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
        # md += f"\n- Categories:{', '.join(product['Categories'])}"
        
        
        return md

    col1.markdown(display_metadata(meta_df))
    col2.image(product['Product Image'],width=300)
else:
    col1,col2 =st.columns(2)
    col1.empty()
    col2.empty()




st.sidebar.subheader("Text Preprocessing Options")
settings_menu = st.sidebar.container(border=True)

# with st.container(border=True):
settings_menu.markdown(">*Select form of text for NLP EDA visuals.*")

# col1,col2=st.columns(2)


text_col_map  ={"Original Text":'review-text-full',
                "Tokenized Text (no stopwords)":'tokens',
            'Lemmatzied Text':'lemmas'
            }
text_preprocessing_selection  =  settings_menu.radio("Select Tokenization",options=list(text_col_map.keys()),# ['Original','Lemmas','Cleaned Tokens'],
                                            index=0)
text_col_selection = text_col_map[text_preprocessing_selection]
## Select # of words/ngrams
ngram_map = {'Unigrams/Tokens (1 Word)':1,
            'Bigrams (2 words)':2,
            'Trigrams (3 words)':3,
            'Quadgrams (4 words)':4}


ngram_selection = settings_menu.radio("Select ngrams", options=list(ngram_map.keys()), #['Single Words','Bigrams','Trigrams','Quadgrams'],
                        index=1)
ngram_n = ngram_map[ngram_selection]
# Select custom stopwords
add_stopwords_str = settings_menu.text_area("Enter list of words to exclude:",value='five,one,two,star,stars,angel,hair,miracle,noodles,shirataki,pasta')
stopwords_list = fn.get_stopwords_from_string(add_stopwords_str)


st.sidebar.divider()



## word clouds

st.divider()
st.subheader("Word Clouds")



# Get groups dict
@st.cache_data
def fn_get_groups_freqs_wordclouds(df,ngrams=ngram_n, as_freqs=True, 
                                        group_col='target-rating', text_col = text_col_selection,
                                        stopwords=stopwords_list):
    kwargs = locals()
    group_texts = fn.get_groups_freqs_wordclouds(**kwargs) #testing stopwords
    return group_texts


    
def download_fig(fig,):
    sname = 'word clodus'
    # Source: https://stackoverflow.com/questions/71713951/how-to-download-matplotlib-graphs-generated-in-a-streamlit-app
    import io
    img = io.BytesIO()
    fig.savefig(img, format='png', dpi=300,bbox_inches='tight' )
    return img
    # if submit_button:
    
# chat_options.markdown("**Clear conversation history.**")

# reset_button  = chat_options.button("Clear",on_click=reset)



with st.container(border=True):
    c1, c2,c3 =st.columns(3)
    c1.markdown("#### Text Preprocesing")
    c1.markdown(">👈 *Change Text Preprocessing Options on the sidebar.*")
    
    c1.markdown("- Recommended Settings:\n    - Tokenization = 'Original Text' \n    - ngrams=Bigrams/Trigrams")
    group_texts = fn_get_groups_freqs_wordclouds(df,ngrams=ngram_n, as_freqs=True,group_col='target-rating', text_col = text_col_selection,
                                        stopwords=stopwords_list )
# preview_group_freqs(group_texts)

    # col1, col2 = st.columns(2)
    c2.markdown("##### WordCloud Options")
    min_font_size = c2.number_input("Minumum Font Size",min_value=4, max_value=50,value=6, step=1)
    max_words = c2.number_input('Maximum # of Words', min_value=10, max_value=1000, value=200, step=5)
    fig  = fn.make_wordclouds_from_freqs(group_texts,stopwords=stopwords_list,min_font_size=min_font_size, max_words=max_words,
                                     figsize=(16,10))
    c3.markdown('##### Download word cloud image.')
    filename = c3.text_input("Filename (png)", value='wordcloud-comparison.png')
    download_fig_button =c3.download_button("Download image.", data =download_fig(fig),file_name=filename,mime="image/png", )

with st.container():
    st.pyplot(fig)

st.divider()


## Add creating ngrams
st.subheader('N-Gram Bar Graphs')

ngram_menu = st.container(border=True)
with ngram_menu:
    col1,col2,col3 = ngram_menu.columns(3)
    col1.markdown("> 👈 *Change Text Preprocessing Options on the sidebar*")
    col1.markdown("- Recommended Settings:\n    - Tokenization = 'Lemmatized Text'\n    - ngrams=Trigrams")

    # ngrams = st.radio('n-grams', [2,3,4],horizontal=True,index=1)
    # top_n = st.select_slider('Compare Top # Ngrams',[10,15,20,25],value=15)
    top_n = col2.slider("Compare Top # Ngrams", min_value=5, max_value=100, step=5,value=20)
    use_plotly = col2.checkbox("Interactive graph", value=False)
    col3.markdown('##### Download word cloud image.')
    filename = col3.text_input("Filename (png)", key='flilename_mgrams',value='ngram-comparison.png')
    download_fig_button_ngram =col3.download_button("Download image.",key='download_ngram', data =download_fig(fig),file_name=filename,mime="image/png",)
    ## Compare n-grams
    
ngrams_df = fn.show_ngrams(df,top_n=top_n, ngrams=ngram_n,text_col_selection=text_col_selection,stopwords_list=stopwords_list)
# st.dataframe(ngrams_df)

if use_plotly:
    fig = fn.plotly_group_ngrams_df(ngrams_df,show=False, title=f"Top {top_n} Most Common ngrams",width=800)
    st.plotly_chart(fig,)
else:
    fig = fn.plot_group_ngrams(ngrams_df,   group1_colname='Low', group2_colname="High", top_n=top_n)#,figsize=(8,12))
    st.pyplot(fig,use_container_width=True)
    
## add chat gpt
chat_container = st.container()
button_ai = st.button("Interpret with ChatGPT")


        
    
def get_template_string(context_low=ngrams_df.loc[:,'Low'].to_string(), context_high=ngrams_df.loc[:,'High'].to_string(), context_type='top ngrams'):
    # task_prompt_dict = get_task_options(options_only=False)
    # system_prompt = task_prompt_dict[selected_task]
    template_starter = f"You are a helpful data analyst for answering questions about what customers said about a specific  Amazon product using only content from use reviews."
    product_string = load_product_info(FPATHS['data']['app']['product-metadata-llm_json'])

    product_template = f" Assume all user questions are asking about the content in the user reviews. Note the product metadata is:\n```{product_string}```\n\n"
    template+=product_template
    
    # template_assistant = "You are a helpful assistant data scientist who uses NLP analysis of customer reviews to inform business-decision-making:"
    # product_template = f" Assume all user questions are asking about the content in the user reviews. Note the product metadata is:\n```{product_string}```\n\n"
    # template_starter = get_template_string_reviews()
    context = f"\nGroup Contexts:\n Here is a {context_type} of 1-star reviews: ```{context_low}```.\n\n Here is a {context_type} of 5-star reviews:```{context_high}."
    context += f"Use the {context_type} first before using the retrieved documents."
    template_assistant=template_starter+ context
    return template_assistant


def get_agent(retriever=None,fpath_db=FPATHS['data']['app']['vector-db_dir'], k=8, temperature=0.1, verbose=False,
             template_string_func=get_template_string):
    
    ## Make retreieval tool
    if retriever is None:
        retriever  = fn.load_vector_database( fpath_db,fpath_llm_csv, k=k, use_previous=True, as_retriever=True)#, use_previous=False)
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


            
            
# def reset_agent(#fpath_db = FPATHS['data']['app']['vector-db_dir'],
#                 retriever=st.session_state['retriever'] , 
#                 starter_message = "Hello, there! Enter your question here and I will check the full reviews database to provide you the best answer.",
#                get_agent_kws={}):
#     # fpath_db
#     agent_exec = get_agent(retriever, **get_agent_kws)
#     agent_exec.memory.chat_memory.add_ai_message(starter_message)
#     with chat_container:
#         st.chat_message("assistant", avatar=ai_avatar).write_stream(fake_streaming(starter_message))
#         # print_history(agent_exec)
#     return agent_exec
    

def fake_streaming(response):
    import time
    for word in response.split(" "):
        yield word + " "
        time.sleep(.05)		

# if 'agent' not in st.session_state:
#     # agent = get_agent(retriever)
#     st.session_state['agent'] =get_agent(template_string_func=lambda: get_template_string(context_low=['']))# reset_agent(retriever=st.session_state['retriever'] )

# if button_ai:
#     ngrams_df = fn.show_ngrams(df,top_n=50, ngrams=ngram_n,text_col_selection=text_col_selection,stopwords_list=stopwords_list)
#     st.session_state['agent'] = get_agent(retriever=s)
#     response = st.session_state['agent'].invoke({'input'})


st.sidebar.subheader("Author Information")
with open("app-assets/author-info.html") as f:
    author_info = f.read()
with st.sidebar.container():
    components.html(author_info)#"""