#!/usr/bin/env python
# coding: utf-8

# ## Exploration data Analysis
# ####  Wilson adejo
# #### 21-02-2021

# In[1]:


#Load in the essential Libraries
import pandas as pd
import numpy as np


# In[4]:


#Read in the dataset with pandas and check the top 5 rows
dataset= pd.read_csv('50_Startups.csv')
dataset.head()


# In[5]:


# Create a statistical summary
dataset.describe()


# In[6]:


#create a correlation table
dataset.corr()


# In[7]:


#Load in the visual libraries
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


x=dataset['Administration']
y=dataset['Profit']
plt.scatter(x,y)
plt.title("Scatter Plot")
plt.xlabel('Administration')
plt.ylabel('Profit');


# In[10]:


plt.bar(dataset['State'],dataset['Marketing Spend'])
plt.title("Bar Plot")
plt.xlabel('State')
plt.ylabel('Marketing Spend');


# In[11]:


import seaborn as sns


# In[12]:


sns.boxplot(x='Profit',data=dataset)


# In[13]:


sns.violinplot(x='State',y='Profit',data=dataset)


# In[14]:


sns.swarmplot(x='State',y='Profit',data=dataset,color='black')
sns.violinplot(x='State',y='Profit',data=dataset)


# In[15]:


sns.distplot(dataset['Profit']);


# In[16]:


sns.jointplot(x='Profit',y='Marketing Spend',data=dataset)


# In[17]:


sns.pairplot(data=dataset)


# In[18]:


dataset.isnull().sum()


# In[19]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[20]:


dataset = pd.get_dummies(dataset)
X = dataset['Profit']
y=dataset.iloc[:,:-1]


# In[21]:


y


# In[ ]:




