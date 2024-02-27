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
# Changing the Layout
st.set_page_config( #layout="wide", 
                   page_icon="🤔 Model Predictions")


##Load in the data
import json
with open("config/filepaths.json") as f:
    FPATHS = json.load(f)

fpath_best_ml = FPATHS['results']['best-ml-clf_joblib']
    
st.sidebar.subheader("Author Information")
    
with open("app-assets/author-info.md") as f:
    author_info = f.read()
    
with st.sidebar.container():
    st.markdown(author_info, unsafe_allow_html=True)
    

@st.cache_resource
def load_best_model_results(fpath_results_joblib):
    import joblib
    return joblib.load(fpath_results_joblib)

## Loading our training and test data
@st.cache_data
def load_Xy_data(joblib_fpath):
    return joblib.load(joblib_fpath)

# X_train, y_train = load_Xy_data(FPATHS['data']['ml-nlp']['train_joblib'])
# X_test, y_test = load_Xy_data(FPATHS['data']['ml-nlp']['test_joblib'])

    
    
## VISIBLE APP COMPONENTS START HERE
st.title("NLP Models & Predictions")
st.subheader("Predicting Amazon Review Rating")

# st.image("Images/dalle-yelp-banner-1.png",width=800,)
st.divider()

## VISIBLE APP COMPONENTS CONTINUE HERE
st.header("Get Model Predictions")

X_to_pred = st.text_area("### Enter text to predict here:", value="I've tried many low carb noodles over the years and I have to say that I was shocked with how bad these miracle noodles were!")

## Lime Explanation Fucntions
from lime.lime_text import LimeTextExplainer
@st.cache_resource
def get_explainer(class_names = None):
	lime_explainer = LimeTextExplainer(class_names=class_names)
	return lime_explainer

def explain_instance(explainer, X_to_pred,predict_func):
	explanation = explainer.explain_instance(X_to_pred, predict_func, labels=(1,))
	return explanation.as_html(predict_proba=False)

# st.markdown("> Predict & Explain:")
get_any_preds = st.button("Get Predictions:")

get_pred_ml = True#st.checkbox("Machine Learning Model",value=True)
# get_pred_nn = st.checkbox("Neural Network", value=True)


def predict_decode(X_to_pred, best_ml_clf,lookup_dict):
    
    if isinstance(X_to_pred, str):
        X = [X_to_pred]
    else:
        X = X_to_pred

    # Get Predixtion
    pred_class = best_ml_clf.predict(X)[0]
    
    # In case the predicted class is missing from the lookup dict
    try:
        # Decode label
        class_name = lookup_dict[pred_class]
    except:
        class_name = pred_class
    return class_name


@st.cache_data
def load_target_lookup(encoder_fpath = FPATHS['metadata']['label_encoder_joblib']):
    # Load encoder and make lookup dict
    encoder = joblib.load(encoder_fpath)

    lookup_dict = {i:class_ for i,class_ in enumerate(encoder.classes_)}
    return encoder, lookup_dict


# Loading the ML model
@st.cache_resource
def load_ml_model(fpath):
    loaded_model = joblib.load(fpath)
    return loaded_model

encoder,target_lookup = load_target_lookup()

explainer = get_explainer(class_names=encoder.classes_)

best_ml_model = FPATHS['models']['ml']['logreg_joblib']
best_ml_clf = joblib.load(best_ml_model)



# check_explain_preds  = st.checkbox("Explain predictions with Lime",value=False)
if (get_pred_ml) & (get_any_preds):
    st.markdown(f"> #### The ML Model predicted:")
    # with st.spinner("Getting Predictions..."):
    # st.write(f"[i] Input Text: '{X_to_pred}' ")
    pred = predict_decode(X_to_pred, lookup_dict=target_lookup,best_ml_clf=best_ml_clf)

    st.markdown(f"#### \t Rating=_{pred}_")
    st.markdown("> Explanation for how the words pushed the model towards its prediction:")
    explanation_ml = explain_instance(explainer, X_to_pred, best_ml_clf.predict_proba )#lime_explainer.explain_instance(X_to_pred, best_ml_clf.predict_proba,labels=label_index_ml)
    with st.container():
        components.html(explanation_ml,height=800)
else: 
    st.empty()


st.divider()

st.header("Model Evaluation")


# col1,col2,col3 = st.columns(3)
# show_train = col1.checkbox("Show training data.", value=True)
# show_test = col2.checkbox("Show test data.", value=True)
# show_model_params =col3.checkbox("Show model params.", value=False)
st.sidebar.header("Model Evaluation Options")
# col1,col2,col3 = st.columns(3)
show_train = st.sidebar.checkbox("Show training data.", value=True)
show_test = st.sidebar.checkbox("Show test data.", value=True)
show_model_params =st.sidebar.checkbox("Show model params.", value=False)

# show_train = st.checkbox("Show training data.", value=True)
# show_test = st.checkbox("Show test data.",value=True)
# show_model_params =st.checkbox("Show model params.", value=False)

# st.subheader("Machine Learning Model")
if st.checkbox("Show model evaluation results."):
    st.markdown('> 👈 ***Select the results that are displayed via the sidebar.***')

    with st.spinner("Loading model results..."):
        results = load_best_model_results(FPATHS['results']['best-ml-clf_joblib'])
        
    if show_train == True:
        # col1,col2=st.columns(2)
        # y_pred_train = clf_bayes_pipe.predict(X_train)
        # report_str, conf_mat = classification_metrics_streamlit(y_train, y_pred_train, label='Training Data')
        st.text(results['train']['classification_report'])
        st.pyplot(results['train']['confusion_matrix'])
        st.text("\n\n")


    if show_test == True: 
        # y_pred_test = clf_bayes_pipe.predict(X_test)
        # report_str, conf_mat = classification_metrics_streamlit(y_test, y_pred_test, cmap='Reds',label='Test Data')
        st.text(results['test']['classification_report'])
        st.pyplot(results['test']['confusion_matrix'])
        st.text("\n\n")

    if show_model_params:
        st.markdown("####  Model Parameters:")
        st.write(results['model'].get_params())

else:
    st.empty()
## Load files and models
st.divider()


# if get_pred_nn & get_any_preds:
# 	st.markdown(f"> #### The Neural Network predicted:")
# 	with st.spinner("Getting Predictions..."):
# 		pred_network, label_index_nn = predict_decode_deep(X_to_pred, network=network , lookup_dict=lookup, )
# 		# st.markdown("**From the Deep Model:**")
# 		# st.write(f"Prediction:\t{pred_network}!")
# 		st.markdown(f"### \t _{pred_network}_")
# 		explanation_nn = explain_instance(lime_explainer, X_to_pred, network.predict,labels=label_index_nn )#lime_explainer.explain_instance(X_to_pred, best_ml_clf.predict_proba,labels=label_index_ml)
# 		# explanation_nn = lime_explainer.explain_instance(X_to_pred, network.predict, labels=label_index_nn)
# 		components.html(explanation_nn)#.as_html())
# else:
# 	st.empty()	

# st.divider()

# st.header("Model Performance")

# button_show_results_ml = st.checkbox("Show ML Model Results", value=False)
# button_show_results_nn = st.checkbox("Show Neural Network Results", value=False)

# if button_show_results_ml:
# 	## MACHINE LEARNING MODEL
# 	st.markdown('#### Machine Learning Model:')
# 	show_results_ml(results_ml)
# 	st.divider()
# else:
# 	st.empty()

# # if button_show_results_nn & button_show_results_ml:
    
 
# if button_show_results_nn:
# 	## Load Neural Network
# 	st.markdown('#### Neural Network Sequence Model:')
# 	show_results_nn(nn_results)
# else:
#     st.empty()

# button_show_nn_results = st.checkbox("Neural Newtwork Results",value=True)
# if button_show_nn_results:
	

# st.divider()
