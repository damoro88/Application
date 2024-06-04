#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
import pickle
import sklearn 
from PIL import Image


# In[2]:


# Load the saved components:
#with open("dt_model.pkl", "rb") as f:
    #components = pickle.load(f)


# In[12]:


# Load the saved components:
with open("dt_model.pkl", "rb") as f:
    components = pickle.load(f)


# In[13]:


# Extract the individual components
num_imputer = components["num_imputer"]
cat_imputer = components["cat_imputer"]
encoder = components["encoder"]
scaler = components["scaler"]
dt_model = components["models"]


# In[14]:


# Create the app

st.set_page_config(
    layout="wide"
)


# In[15]:


# Add an image or logo to the app
image = Image.open('copofav.jpg')


# In[16]:


# Open the image file
st.image(image)


# In[17]:


#add app title
st.title("SALES PREDICTION APP")


# In[18]:


# Add some text
st.write("Please ENTER the relevant data and CLICK Predict.")


# In[19]:
# Create the input fields
input_data = {}
col1,col2,col3 = st.columns(3)
with col1:
    input_data['store_nbr'] = st.slider("Store Number",0,54)
    input_data['products'] = st.selectbox("Products Family", ['OTHERS', 'CLEANING', 'FOODS', 'STATIONERY', 'GROCERY', 'HARDWARE',
       'HOME', 'CLOTHING'])
    input_data['onpromotion'] =st.number_input("Discount Amt On Promotion",step=1)
    input_data['state'] = st.selectbox("State", ['Pichincha', 'Cotopaxi', 'Chimborazo', 'Imbabura',
       'Santo Domingo de los Tsachilas', 'Bolivar', 'Pastaza',
       'Tungurahua', 'Guayas', 'Santa Elena', 'Los Rios', 'Azuay', 'Loja',
       'El Oro', 'Esmeraldas', 'Manabi'])
with col2:    
    input_data['store_type'] = st.selectbox("Store Type",['D', 'C', 'B', 'E', 'A'])
    input_data['cluster'] = st.number_input("Cluster",step=1)
    input_data['dcoilwtico'] = st.number_input("DCOILWTICO",step=1)
    input_data['year'] = st.number_input("Year to Predict",step=1)
with col3:    
    input_data['month'] = st.slider("Month",1,12)
    input_data['day'] = st.slider("Day",1,31)
    input_data['dayofweek'] = st.number_input("Day of Week,0=Sunday and 6=Satruday",step=1)
    input_data['end_month'] = st.selectbox("Is it End of the Month?",['True','False'])





