#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data Handling

import pandas as pd
import numpy as np

#Visualization librairies

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import matplotlib.cm as cm


# Statistical Analysis
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score

# Feature Processing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Modeling
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV


# In[2]:


events_df=pd.read_csv(r'C:\Users\merchandisers\Desktop\FAV DATA\holidays_events.csv')
Items_df=pd.read_csv(r'C:\Users\merchandisers\Desktop\FAV DATA\items.csv')
oil_df=pd.read_csv(r'C:\Users\merchandisers\Desktop\FAV DATA\oil.csv')
Sample_submission_df=pd.read_csv(r'C:\Users\merchandisers\Desktop\FAV DATA\sample_submission.csv')
store_df=pd.read_csv(r'C:\Users\merchandisers\Desktop\FAV DATA\stores.csv')
test_df=pd.read_csv(r'C:\Users\merchandisers\Desktop\FAV DATA\test.csv')
df_transactions=pd.read_csv(r'C:\Users\merchandisers\Desktop\FAV DATA\transactions.csv')


# In[3]:


#Import train dataset

filename = 'train.txt'
train_df = np.loadtxt(filename, delimiter=',', usecols=(1,2,3,4,5),dtype='str')
print(train_df)


# In[4]:


train_df=pd.DataFrame(train_df)
train_df


# In[5]:


#ADDING COLUMNS NAMES

train_df.columns =['date', 'store_nbr', 'family', 'sales','onpromotion']
train_df.head()


# In[6]:


train_df.info()


# In[7]:


#convert columns types

train_df['sales']=train_df['sales'].astype('float')
train_df['store_nbr']=train_df['store_nbr'].astype('int')


# In[8]:


# Convert 'date' column to datetime format
train_df['date'] = pd.to_datetime(train_df['date'])


# # Training Data

# In[9]:


train_df.head()


# In[10]:


train_df.info()


# In[11]:


train_df.shape


# In[12]:


train_df.head().isnull().sum()


# # Events Data

# In[13]:


events_df.head()


# In[14]:


events_df.info()


# In[15]:


events_df.shape


# In[16]:


events_df.isnull().sum()


# # Oil Data

# In[17]:


oil_df.head()


# In[18]:


oil_df.info()


# In[19]:


oil_df['date'].agg(['min', 'max'])


# In[20]:


#Getting the missing dates
oil_missing_dates=pd.date_range(start='2013-01-01', end='2017-08-31').difference(oil_df.date.unique())


# In[21]:


oil_missing_dates


# In[22]:


missing_data = pd.DataFrame(oil_missing_dates, columns=['date'])
missing_data


# In[23]:


#combining the original train data with the missing date dataframe
oil_data=pd.concat([oil_df,missing_data],ignore_index= True)
oil_data.head()


# In[24]:


oil_data['dcoilwtico'].interpolate(method ='linear', limit_direction ='both',inplace=True)


# In[25]:


oil_data.isnull().sum()


# In[26]:


oil_data.shape


# # Store Data

# In[27]:


store_df.head()


# In[28]:


store_df.info()


# In[29]:


store_df.isnull().sum()


# In[30]:


store_df.shape


# # Test data

# In[31]:


test_df.head()


# In[32]:


test_df.info()


# In[33]:


test_df.isnull().sum()


# In[34]:


test_df.shape


# # Exploratory Data Analysis: EDA

# # 1. Is the train dataset complete (has all the required dates)?

# In[35]:


train_df['date'].agg(['min', 'max'])


# In[36]:


#Getting the missing dates
train_missing_dates=pd.date_range(start='2013-01-01', end='2017-08-15').difference(train_df.date.unique())
train_missing_dates


# # filling the missing dates

# In[37]:


from itertools import product

added_dates=list(product(train_missing_dates,train_df.store_nbr.unique(),train_df.family.unique()))


# In[38]:


train_missing_data = pd.DataFrame(added_dates, columns=['date','store_nbr','family'])


# In[39]:


train_missing_data.head()


# In[40]:


#combining the original train data with the missing date dataframe
train_data=pd.concat([train_df,train_missing_data],ignore_index= True)
train_data.head()


# In[41]:


train_data.duplicated().sum()


# In[42]:


train_data.isna().sum()


# In[43]:


train_data.shape


# In[44]:


events_df.info()


# In[45]:


oil_df.info()


# In[46]:


# Convert 'date' column to datetime format
events_df['date'] = pd.to_datetime(events_df['date'])
oil_df['date'] = pd.to_datetime(oil_df['date'])


# # merging the different csv files into one dataframe

# In[47]:


#combining train data with storedata
train_store_df=pd.merge(train_data,store_df,on='store_nbr',how='left')
train_store_df.head()


# In[48]:


#Combining train_store_df with the events_df on the dates
combined_df=pd.merge(train_store_df, events_df, on='date',how='left')
combined_df.head()


# In[49]:


oil_data.info()


# In[50]:


oil_data['date'] = pd.to_datetime(oil_data['date'])


# In[ ]:





# In[51]:


df=pd.merge(combined_df,oil_data,on='date',how='left')


# In[52]:


df.tail()


# In[53]:


df.isnull().sum()


# In[54]:


df.shape


# In[55]:


#Changing Index to Date Column
df = df.set_index(["date"])


# # 2. Which dates have the lowest and highest sales for each year?

# In[56]:


#Extracting the dates with lowest sales
lowest_sales = df.groupby(df.index.year)['sales'].idxmin()
lowest_sales


# In[57]:


#Extracting dates with the highest sales in each year
highest_sales = df.groupby(df.index.year)['sales'].idxmax()
highest_sales


# The beginning of each year recorded the lowest sales of that year this can be attributed to the feastive season and many people are still home. As for the highest sales that was varied across the different years.

# # 3. Did the earthquake impact sales?

# In[58]:


# Define the date of the earthquake happend
earthquake_date = pd.to_datetime("2016-04-16")


# In[59]:


# Get the sales two weeks before the earthquake
before_sales = df.loc[earthquake_date - pd.Timedelta(weeks=2):earthquake_date, 'sales']
print("Sales two weeks before {}: {}".format(earthquake_date, before_sales))


# In[60]:


# Get the sales two weeks after the earthquake
after_sales = df.loc[earthquake_date :earthquake_date + pd.Timedelta(weeks=2),'sales']
print("Sales two weeks after {}: {}".format(earthquake_date, after_sales))


# In[61]:


#Extracting the dates 2 weeks before the earthquake
before_df = df.loc[earthquake_date - pd.Timedelta(weeks=2):earthquake_date]

#Extracting the dates 2 weeks after the earthquake
after_df = df.loc[earthquake_date:earthquake_date + pd.Timedelta(weeks=2)]

# Plot the sales before and after the earthquake
plt.figure(figsize=(20, 10))
Before=plt.plot(before_df.index, before_df['sales'], label='Before')
After=plt.plot(after_df.index, after_df['sales'], label='After')

plt.axvline(earthquake_date, color='r', linestyle='--')

plt.title("Sales for 2 Weeks Before and After {}".format(earthquake_date))
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()


# We did a comparison of sales before the earthquake 2 weeks before and after, we can conclude after the earthquake the sales increased significantly

# # 4. Are certain groups of stores selling more products? (Cluster, city, state, type)

# In[62]:


# Group the data by cluster
cluster_groups = df.groupby(['cluster'])['sales'].sum()
plt.figure(figsize=(15, 10))
cluster_groups.plot(kind = 'barh')

# Add a title and axis labels
plt.title("Total Sales per Cluster")
plt.xlabel("Total Sales in Millions")
plt.ylabel("Cluster ")
plt.show()


# From the chart we see that cluster 14,6 and 8 are the top 3 with most sales of mote than 1.0 million.

# In[63]:


# Group the data by cluster
city_groups = df.groupby(['city'])['sales'].sum()
plt.figure(figsize=(15, 10))
city_groups.plot(kind = 'barh')

# Add a title and axis labels
plt.title("Total Sales per City")
plt.xlabel("Total Sales in Millions")
plt.ylabel("City")

plt.show()


# From the chart we see that Quito city has the most sales compared to the rest and this is because many stores are located in Quito.

# In[64]:


# Group the data by cluster
state_groups = df.groupby(['state'])['sales'].sum()
plt.figure(figsize=(15, 10))
state_groups.plot(kind = 'barh')

# Add a title and axis labels
plt.title("Total Sales per State")
plt.xlabel("Total Sales in Millions")
plt.ylabel("State")

plt.show()


# In[65]:


# Group the data by cluster
type_groups = df.groupby(['type_x'])['sales'].sum()
plt.figure(figsize=(15, 10))
type_groups.plot(kind = 'bar')

# Add a title and axis labels
plt.title("Total Sales by Type X")
plt.xlabel("Type")
plt.ylabel("Total Sales in Millions")

plt.show()


# # 5. Are sales affected by promotions, oil prices and holidays?

# In[66]:


df.head()


# In[67]:


df.info()


# In[ ]:





# In[68]:


#computing the Pearson correlation coefficient using corr()
#corr_matrix= df.corr()


# In[69]:


#the Pearson correlation coefficient is to measure the strength of the relationship btwn variables with numeric values
#corr_matrix['sales'].sort_values(ascending=False)


# In[70]:


#another method to check the correlation is the pandas scatter_matrix function which plots
#every numerical attribute against every other numerical attribute.
#attributes=['onpromotion','sales','dcoilwtico']
#scatter_matrix(df[attributes],figsize=(18,20))
#plt.show()


# # 6. What analysis can we get from the date and its extractable features?

# In[71]:


#Using the month extracted fromthe date to get monthly trend of sales
monthly_sales = df.groupby(df.index.month)['sales'].sum()
monthly_sales.plot(linewidth=1.2, figsize=(10,5))

# Add a title and axis labels
plt.title("Trend of Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Total Sales (in Millions) ")

plt.show()


# In[72]:


# Using the days of the week extracted from date get the trend of sales of the days of the week
day_sales = df.groupby(df.index.day_name())['sales'].sum()
day_sales.plot(linewidth=1.2, figsize=(20,10))

# Add a title and axis labels
plt.title("Trend of Monthly Sales")
plt.xlabel("Days of the Week")
plt.ylabel("Total Sales (in Millions) ")

plt.show()


# # 7. What are the annual sales that each store make

# In[73]:


# Group the data by store and year
annual_sales = df.groupby(['store_nbr',df.index.year])['sales'].agg('sum')


# In[74]:


# Reset the index to make the store and year columns columns
annual_sales = annual_sales.reset_index()


# In[75]:


# Pivot the table so that each row represents a store and columns are the year
annual_sales = annual_sales.pivot(index='store_nbr', columns='date', values='sales')


# In[76]:


# Plot the yearly sales for each store
annual_sales.plot(kind='bar', stacked=True, figsize=(20,10))

# Add a title and axis labels
plt.title("Annual_sales Sales by Store")
plt.xlabel("Store Numbers")
plt.ylabel("Total Sales (in Millions) ")

plt.show()


# # 8. How many sales were made in the 1st quater of each year

# In[77]:


#Extracting the quaterly sales
Quarterly_sales = df.groupby([df.index.to_period('Q'),df.index.year])['sales'].sum()


# In[78]:


# filter for 1st quarter
first_quarter_data = Quarterly_sales.loc[['2013Q1','2014Q1','2015Q1','2016Q1','2017Q1']]


# In[79]:


# create bar chart to represent the sales made in the 1st quater of each year
first_quarter_data.plot(kind='bar', stacked=False, figsize=(15,10))
plt.xticks(rotation=45)
plt.title('Sales Trend for 2013 and 2017')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.show()


# There has been a steady growth across the years. The sales keep growing Quarter on Quarter every year.

# # 9. What is the trend of the sales over the years?

# In[80]:


#Trend of sales over the years
df['sales'].plot(linewidth=0.5, figsize=(20,10))
plt.title('Sales Trend for 2013 and 2017')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.show()


# # Feature Processing & Engineering

# # New Features Creation

# In[81]:


#Creating the Day, Month and Year column from the Date Column
df['year'] = df.index.year
df['month'] =df.index.month
df['day']=df.index.day
df['dayofweek']=df.index.weekday
df['end_month']=df.index.is_month_end


# In[82]:


# categorizing the products
food_families = ['BEVERAGES', 'BREAD/BAKERY', 'FROZEN FOODS', 'MEATS', 'PREPARED FOODS', 'DELI','PRODUCE', 'DAIRY','POULTRY','EGGS','SEAFOOD']
df['family'] = np.where(df['family'].isin(food_families), 'FOODS', df['family'])
home_families = ['HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES']
df['family'] = np.where(df['family'].isin(home_families), 'HOME', df['family'])
clothing_families = ['LINGERIE', 'LADYSWARE']
df['family'] = np.where(df['family'].isin(clothing_families), 'CLOTHING', df['family'])
grocery_families = ['GROCERY I', 'GROCERY II']
df['family'] = np.where(df['family'].isin(grocery_families), 'GROCERY', df['family'])
stationery_families = ['BOOKS', 'MAGAZINES','SCHOOL AND OFFICE SUPPLIES']
df['family'] = np.where(df['family'].isin(stationery_families), 'STATIONERY', df['family'])
cleaning_families = ['HOME CARE', 'BABY CARE','PERSONAL CARE']
df['family'] = np.where(df['family'].isin(cleaning_families), 'CLEANING', df['family'])
hardware_families = ['PLAYERS AND ELECTRONICS','HARDWARE']
df['family'] = np.where(df['family'].isin(hardware_families), 'HARDWARE', df['family'])
others_families = ['AUTOMOTIVE', 'BEAUTY','CELEBRATION', 'LADIESWEAR', 'LAWN AND GARDEN', 'LIQUOR,WINE,BEER',  'PET SUPPLIES']
df['family'] = np.where(df['family'].isin(others_families), 'OTHERS', df['family'])


# In[83]:


df['family'].unique()


# In[84]:


df.rename(columns = {"type_x":"store_type","family":"products"}, inplace = True)


# In[85]:


df.head()


# In[86]:


df.drop(['locale','locale_name','description','transferred','city','type_y'],axis=1,inplace=True)


# In[87]:


df.reset_index(inplace=True)


# In[88]:


df.head(5)


# In[89]:


df.isnull().sum()


# In[90]:


df['sales'].fillna(0,inplace=True)


# In[91]:


df['products'].unique()


# In[92]:


df['state'].unique()


# In[93]:


df['store_type'].unique()


# # Data spliting

# In[94]:


# Calculate the number of rows in the data
n_rows = df.shape[0]

# Calculate the split point
split_point = int(n_rows * 0.85)

# Select the first 85% of the rows as the training data
X_train = df.iloc[:split_point]
y_train = X_train['sales']
X_train = X_train.drop('sales', axis=1)

# Select the remaining 15% of the rows as the validation data
X_eval = df.iloc[split_point:]
y_eval = X_eval['sales']
X_eval = X_eval.drop('sales', axis=1)


# In[95]:


X_train.shape,X_eval.shape,y_train.shape,y_eval.shape


# # Impute Missing Values

# In[96]:


categorical_columns = ['products', 'end_month', 'store_type', 'state']


# In[97]:


numerical_columns =['store_nbr','onpromotion','cluster','dcoilwtico','year','month','day','dayofweek']


# In[98]:


'''creating copy of the categorical features and numerical features
before imputing null value to avoid modifying the orginal dataset'''

X_train_cat = X_train[categorical_columns].copy()
X_train_num = X_train[numerical_columns].copy()

X_eval_cat = X_eval[categorical_columns].copy()
X_eval_num = X_eval[numerical_columns].copy()


# In[99]:


# Creating imputer variables
from sklearn.impute import SimpleImputer

numerical_imputer = SimpleImputer(strategy = "mean")
categorical_imputer = SimpleImputer(strategy = "most_frequent")


# In[100]:


# Fitting the Imputer
X_train_cat_imputed = categorical_imputer.fit_transform(X_train_cat)
X_train_num_imputed = numerical_imputer.fit_transform(X_train_num)

X_eval_cat_imputed = categorical_imputer.fit_transform(X_eval_cat)
X_eval_num_imputed = numerical_imputer.fit_transform(X_eval_num)


# # Features Encoding

# In[101]:


from sklearn.preprocessing import OneHotEncoder,StandardScaler

encoder=OneHotEncoder(handle_unknown='ignore')


# In[102]:


# encoding the xtrain categories and converting to a dataframe
X_train_cat_encoded = encoder.fit(X_train_cat_imputed)
X_train_cat_encoded = pd.DataFrame(encoder.transform(X_train_cat_imputed).toarray(),
                                   columns=encoder.get_feature_names_out(categorical_columns))


# In[103]:


# encoding the xeval categories and converting to a dataframe
X_eval_cat_encoded = encoder.fit(X_eval_cat_imputed)
X_eval_cat_encoded = pd.DataFrame(encoder.transform(X_eval_cat_imputed).toarray(),
                                   columns=encoder.get_feature_names_out(categorical_columns))


# # Data scalling

# In[104]:


scaler= StandardScaler()


# In[105]:


X_train_num_scaled = scaler.fit_transform(X_train_num_imputed)
X_train_num_sc = pd.DataFrame(X_train_num_scaled, columns = numerical_columns)


# In[106]:


X_eval_num_scaled = scaler.fit_transform(X_eval_num_imputed)
X_eval_num_sc = pd.DataFrame(X_eval_num_scaled, columns = numerical_columns)


# # Combining the xtrain cat and xtrain num

# In[107]:


X_train_df = pd.concat([X_train_num_sc,X_train_cat_encoded], axis =1)
X_eval_df = pd.concat([X_eval_num_sc,X_eval_cat_encoded], axis =1)


# In[108]:


X_train_df.head()


# In[109]:


X_eval_df.head()


# # Machine learning models

# # Decision Tree Regression Model

# In[110]:


from sklearn.tree import DecisionTreeRegressor

#fitting decision tree model
dt_model=DecisionTreeRegressor()
dt_model.fit(X_train_df,y_train)


# In[111]:


#to measure the this regression model's rmse
dt_pred=dt_model.predict(X_eval_df)
dt_mse=mean_squared_error(y_eval,dt_pred)
dt_rmse= np.sqrt(dt_mse)
dt_MAE=mean_absolute_error(y_eval,dt_pred)
dt_rmsle = np.sqrt(mean_squared_log_error(y_eval,dt_pred))


# In[112]:


results=pd.DataFrame([['Decision Tree', dt_mse,dt_rmse,dt_MAE,dt_rmsle]],
                     columns = ['Model', 'MSE','RMSE','MAE','RMSLE'])
results


# # Cleaning Test Data

# In[113]:


test_df['date'] = pd.to_datetime(test_df['date'])


# In[114]:


test_events_df=pd.merge(test_df,events_df,on='date',how='left')
test_events_df.head()


# In[115]:


oil_test_events=pd.merge(test_events_df,oil_data,on ='date',how= 'left')
oil_test_events.head()


# In[116]:


merged_test=pd.merge(oil_test_events,store_df,on ='store_nbr',how= 'left')
merged_test.head()


# In[117]:


merged_test=merged_test.drop(['id','locale','locale_name','description','transferred','city','type_x'],axis=1)


# In[118]:


merged_test.rename(columns={'type_y':'store_type','family':'products'},inplace =True)


# In[119]:


merged_test=merged_test.set_index(['date'])


# In[120]:


merged_test['year'] = merged_test.index.year
merged_test['month'] =merged_test.index.month
merged_test['day']=merged_test.index.day
merged_test['dayofweek']=merged_test.index.weekday
merged_test['end_month']=merged_test.index.is_month_end


# In[121]:


# categorizing the products
food_families = ['BEVERAGES', 'BREAD/BAKERY', 'FROZEN FOODS', 'MEATS', 'PREPARED FOODS', 'DELI','PRODUCE', 'DAIRY','POULTRY','EGGS','SEAFOOD']
merged_test['products'] = np.where(merged_test['products'].isin(food_families), 'FOODS', merged_test['products'])
home_families = ['HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES']
merged_test['products'] = np.where(merged_test['products'].isin(home_families), 'HOME', merged_test['products'])
clothing_families = ['LINGERIE', 'LADYSWARE']
merged_test['products'] = np.where(merged_test['products'].isin(clothing_families), 'CLOTHING', merged_test['products'])
grocery_families = ['GROCERY I', 'GROCERY II']
merged_test['products'] = np.where(merged_test['products'].isin(grocery_families), 'GROCERY', merged_test['products'])
stationery_families = ['BOOKS', 'MAGAZINES','SCHOOL AND OFFICE SUPPLIES']
merged_test['products'] = np.where(merged_test['products'].isin(stationery_families), 'STATIONERY', merged_test['products'])
cleaning_families = ['HOME CARE', 'BABY CARE','PERSONAL CARE']
merged_test['products'] = np.where(merged_test['products'].isin(cleaning_families), 'CLEANING', merged_test['products'])
hardware_families = ['PLAYERS AND ELECTRONICS','HARDWARE']
merged_test['products'] = np.where(merged_test['products'].isin(hardware_families), 'HARDWARE', merged_test['products'])
others_families = ['AUTOMOTIVE', 'BEAUTY','CELEBRATION', 'LADIESWEAR', 'LAWN AND GARDEN', 'LIQUOR,WINE,BEER',  'PET SUPPLIES']
merged_test['products'] = np.where(merged_test['products'].isin(others_families), 'OTHERS', merged_test['products'])


# In[122]:


merged_test.reset_index(inplace=True)


# In[123]:


merged_test.shape


# In[124]:


merged_test.head()


# In[125]:


merged_test['products']


# In[126]:


merged_test = merged_test.drop("date", axis=1)


# # Imputing missing values

# In[127]:


X_test_cat = merged_test[categorical_columns].copy()
X_test_num = merged_test[numerical_columns].copy()


# In[128]:


categorical_columns


# In[129]:


# fitting imputer
X_test_cat_imputed = categorical_imputer.fit_transform(X_test_cat)
X_test_num_imputed = numerical_imputer.fit_transform(X_test_num)


# # Feature Encoding

# In[130]:


# encoding the xtrain categories and converting to a dataframe
X_test_cat_encoded = encoder.fit(X_test_cat_imputed)
X_test_cat_encoded = pd.DataFrame(encoder.transform(X_test_cat_imputed).toarray(),
                                   columns=encoder.get_feature_names_out(categorical_columns))


# # Data scalling

# In[131]:


X_test_num_scaled = scaler.fit_transform(X_test_num_imputed)
X_test_num_sc = pd.DataFrame(X_test_num_scaled, columns = numerical_columns)


# # Combining xtest cat and xtest num

# In[132]:


X_test_df = pd.concat([X_test_num_sc,X_test_cat_encoded], axis =1)


# In[133]:


X_test_df.head()


# # Making predictions with unseen data

# In[134]:


dtree_predictions=dt_model.predict(X_test_df)


# In[135]:


X_test_df['sales'] = dtree_predictions


# In[136]:


X_test_df['sales'].to_csv('submission.csv', index=False)


# # Exporting Key Components

# In[137]:


import os

if not os.path.exists("export"):
    os.makedirs("export")


# In[138]:


# set the destination path to the "export" directory
destination = os.path.join(".", "export")


# In[139]:


components = {
    "num_imputer":numerical_imputer,
    "cat_imputer": categorical_imputer,
    "encoder": encoder,
    "scaler": scaler,
    "models": dt_model
}


# In[140]:


import pickle

# Export the model
with open(os.path.join(destination,"dt_model.pkl"), "wb") as f:
    pickle.dump(components, f)


# In[141]:


get_ipython().system('pip list --format=freeze >requirements.txt')


# In[142]:


merged_test['products'].unique()


# In[ ]:




