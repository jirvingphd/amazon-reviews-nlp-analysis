"""James Irving app 
"""
import streamlit as st 
if st.__version__ <"1.31.0":
    streaming=False
else:
    streaming=True

import time,os
# from streamlit_chat

## LLM Classes 
from langchain_openai import OpenAI
# from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage, SystemMessage, AIMessage


## Memory Modules
from langchain.chains.conversation.memory import (ConversationBufferMemory, 
                                                  ConversationSummaryBufferMemory,
                                                  ConversationBufferWindowMemory,
                                                  ConversationSummaryMemory)
# Template for changing conversation chain's "flavor"
from langchain.prompts.prompt import PromptTemplate

# Create required session_state containers
if 'messages' not in st.session_state:
    st.session_state.messages=[]
    
if 'API_KEY' not in st.session_state:
    st.session_state['API_KEY'] = os.environ['OPENAI_API_KEY'] # Could have user paste in via sidebar

if 'conversation' not in st.session_state:
    st.session_state['conversation'] = None


def reset():
    if 'messages' in st.session_state:
        st.session_state.messages=[]

    if 'conversation' in st.session_state:
        st.session_state['conversation'] = None


st.set_page_config(page_title="ChatGPT Clone", page_icon=':robot:')
# st.header("Hey, I'm your Chat GPT")
st.header("How can I assist you today?")


## Define chatbot personalities/flavors
flavor_options = {
    "helpful": "a helpful assistant.",
    "sassy": " a sassy assistant who uses sarcasm and insults.",
    "child": " a 4 year old with limited vocabulary and childish grammar. You annotate your physical actions with asterisks as you answer.",
    "mother": "the user's mother and are overly affectionate and say embarrassing things about my childhood. You annotate your physical actions with asterisks as you answer.",
    "bartender": " a charming and emotionally intelligent bartender who gives great advice. You annotate your physical actions with asterisks as you answer.",
    "angry/mean": " an impatient and hostile assistant who provides answers, but with contempt and insults.",
    "flirtatious": " a flirtatious assistant who uses a lot of double-entendres and hits on the user. You annotate your physical actions with asterisks as you answer.",
    "evil hypnotist":"an charming evil hypnotist who tries to entrance the user in all answers. You annotate your physical actions with asterisks as you answer."
}

# temp = st.sidebar.slider("Select model temperature:",min_value=0.0, max_value=2.0, value=0.1)

def set_conversation_flavor(llm,flavor_name):
    # Select the correct prompt from the dictionary of options
    flavor= flavor_options[flavor_name]
 
    # Use an f-string to constuct the new start of prompt
    flavor_text = f"The following is a conversation between a human and an assistant. The assistant is {flavor}."
    # Add the rest of the prompt
    template = flavor_text + """
    Current conversation:
    {history}
    Human: {input}
    AI Assistant:"""
    PROMPT = PromptTemplate(input_variables=["flavor","history", "input"], template=template)
    conversation = ConversationChain(
        prompt=PROMPT,
        llm=llm,
        verbose=True,
        memory=ConversationBufferMemory(ai_prefix="AI Assistant"), #SummaryMemory?
    )
    return conversation

    

# Create chatbot with selected temp
user_avatar = st.sidebar.selectbox("Select an avatar:", options=['📞','🧔🏻‍♂️','👩‍🦰',"🥷","🌀"], index=1)

temp=st.sidebar.slider("model temperature:",min_value=0.0, max_value=2.0, value=0.0, step=.1)
reset_on_change = st.sidebar.checkbox("Reset when flavor changed?",value=False)
if reset_on_change == True:
    flavor = st.sidebar.selectbox("Which type of chatbot?", key='reset_flavor',options=list(flavor_options.keys()), index=0, on_change=reset())
else:
    flavor = st.sidebar.selectbox("Which type of chatbot?", key='no_reset',options=list(flavor_options.keys()), index=0,)



def get_response(query):
    
    if st.session_state['conversation'] is None:
        llm = OpenAI(max_tokens=500,
            openai_api_key=st.session_state['API_KEY'],
               temperature=float(temp),
            model_name='gpt-3.5-turbo-instruct'  # 'text-davinci-003' model is depreciated now, so we are using the openai's recommended model
        )
  
    
    if st.session_state['conversation'] is None:
        st.session_state['conversation'] = set_conversation_flavor(llm,flavor_name=flavor)

    response=st.session_state['conversation'].predict(input=query)
    # st.session_state['messages'].append()
    print(st.session_state['conversation'].memory.buffer)

    return response
    # return show_history()

def response_gen(response):
    for word in response.split():
        yield word + " "
        time.sleep(.05)		



def display_chat():
    with response_container:
        # st.write("Response Container:")
    
        for message in st.session_state.conversation.memory.buffer_as_messages[:-1]:
            type_message = str(type(message)).lower()
            if 'system' in type_message:
                continue
            elif "human" in type_message:
                human_message =  st.chat_message("user", avatar= user_avatar)#"🤷‍♂️")
    
                msg_md_format = f"**User:**\n\t{message.content}"
                human_message.write(msg_md_format)#f"User: {message.content}")
    
            elif 'ai' in type_message:
                ai_message =  st.chat_message('ai',avatar= "🤖")
                msg_md_format = f"**({flavor.title()}) AI**:\n\t {message.content}"
                # ai_message.write(msg_md_format)
                ai_message.markdown(msg_md_format)
    
        last_message= st.session_state.conversation.memory.buffer_as_messages[-1]
        with st.chat_message('ai',avatar= "🤖"):
            msg_md_format = f"**({flavor.title()}) AI**:" #\n\t {message.content}"
            # st.write(msg_md_format)
            st.markdown(msg_md_format)
            
            if streaming==True:
                st.write_stream(response_gen(last_message.content))
            else:
                st.write(last_message.content)



# Here we will have a container for user input text box
# container = st.chat_message('Human')#st.container()
user_input =  st.chat_input(placeholder ="Hello,there!")#, on_submit=display_chat)#

# submit_button = st.button(label="Send")
response_container = st.container()



if user_input:
    # with st.chat_message("user"):
    # 	st.markdown(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))
    response = get_response(user_input)
    st.session_state.messages.append(AIMessage(content=user_input))
    display_chat()
    # with st.chat_message("assistant"):
    # 	st.write(response)

# model_response=#,st.session_state['API_KEY'])

        
# summarise_button = st.sidebar.button("Summarise the conversation", key="summarise")
# if summarise_button:
# 	summarise_placeholder = st.sidebar.write("Nice chatting with you my friend ❤️:\n\n"+st.session_state['conversation'].memory.buffer)

        

clear = st.sidebar.button("Clear history?")

if clear:
    reset()