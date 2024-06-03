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
#with open("export/dt_model.pkl", "rb") as f:
    #components = pickle.load(f)


# In[12]:


# Load the saved components:
with open("export/dt_model.pkl", "rb") as f:
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
image = Image.open(r'export/copofav.jpg')


# In[16]:


# Open the image file
st.image(image)


# In[17]:


#add app title
st.title("SALES PREDICTION APP")


# In[18]:


# Add some text
st.write("Please ENTER the relevant data and CLICK Predict.")


# In[ ]:




