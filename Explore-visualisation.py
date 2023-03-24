#!/usr/bin/env python
# coding: utf-8

# <h2>Import Libraries</h2>

# ## Explore Visualization
# ####  Wilson adejo
# #### 21-10--2022

# In[1]:


get_ipython().system('pip install folium')


# In[2]:



import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from numpy import ravel # For matrices
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
from sklearn.decomposition import PCA
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, ConfusionMatrixDisplay, accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold # Feature selector
# Various pre-processing steps
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, PowerTransformer, MaxAbsScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV # For optimization

import folium
import folium.plugins as plugins
from folium.plugins import HeatMapWithTime
from folium.plugins import HeatMap

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


kdf = pd.read_csv('Road_Safety_Accidents.csv') ##read the data
kdf


# <h2>Explore the data base with numeric database</h2>

# In[4]:


kdf.describe()


# <h2>View the data for each column data type, and column names</h2>

# In[5]:


kdf.info()


# In[6]:


if kdf.isnull().values.any() == True:
    print(kdf.isnull().sum()) ##shows a table with one column showing number(sum) of missing value on that row
    print(kdf.isnull().sum().sum()) ##shows total summation of all missing values


# In[7]:


##loooking at the data above we can see that all columns containing nan values, have them in large number, hence need to drop all columns with nan values 


# In[8]:


rdf = kdf.dropna(axis=1, how="any") ##drop columns with more than 10 nan values


# In[9]:


plt.figure(figsize=(12,4))
plt.scatter(rdf['NUMBER_OF_VEHICLES'],rdf['NUMBER_OF_CASUALTIES'])


# In[10]:


##shows that accident with 1-6 vehicles have majority number of casualties


# In[11]:


plt.figure(figsize=(12,4))
plt.bar(rdf['ACCIDENT_SEVERITY'],rdf['SPEED_LIMIT'])


# In[12]:


##low number of fatal accident, and equal numbers of slight and serious accidents


# In[13]:


#creating function to add month column
def month(string):
    return int(string[5:7])
kdf['Month']=kdf['DATE_'].apply(lambda x: month(x))


# In[14]:


#check accident severity with month
df_chk = pd.DataFrame(data=kdf,columns=['DAY_OF_WEEK','Month','ACCIDENT_SEVERITY'])
df_chk


# In[15]:


# getting cases of 'Serious Accidents' only.
df_chk = df_chk[df_chk.ACCIDENT_SEVERITY == 'Serious']
df_chk.head(15)


# In[16]:


corr = rdf.corr()
#corr.to_excel('correlationMatrix.xlsx')
corr


# In[17]:



corr = rdf.corr()
#corr.to_excel('correlationMatrix.xlsx')
corr
corr.style.background_gradient(cmap='coolwarm')


# In[74]:


# The labels

y = rdf['ACCIDENT_SEVERITY']

# Encode the labels into unique integers
encoder = LabelEncoder()
y = encoder.fit_transform(ravel(y))
y.shape


# In[75]:


#X = rdf.iloc[:, :].values
X = rdf.drop(['ACCIDENT_SEVERITY'], axis=1)
if len(X.select_dtypes(include=['object']).columns) > 0:
    print(X.select_dtypes(include=['object']).columns) ##print out the columns containing string values
    
    d_lst = list(X.select_dtypes(include=['object']).columns)
    dummies = pd.get_dummies(X, columns = d_lst)
    trr = []
    for i in range(len(d_lst)):
        lbl = LabelEncoder()
        tt = lbl.fit_transform(ravel(X[d_lst[i]]))
        trr.append(tt)
        
    dum = pd.DataFrame(data=trr).T
    dum.columns = d_lst
    X = X.drop(columns = d_lst)
    X = pd.concat([X, dum], axis='columns')
    # drop the values
    #X = merged.drop(d_lst, axis='columns')
#X
X


# In[76]:


pca = PCA(n_components = 5) 
principalComponents = pca.fit_transform(X)          ##checking how pca and standard scaling works
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4','principal component 5'])
principalDf


# In[77]:


print('\n Total Variance Explained using 5 coomponents:', round(sum(list(pca.explained_variance_ratio_))*100, 2))


# In[78]:




components = pca.fit_transform(X)

fig = px.scatter(components, x=0, y=1, color=rdf['ACCIDENT_SEVERITY'])
fig.show()
##distribution of accident severity on a scatter plot using dimensionally reduced data(pca)


# In[79]:


fig = px.scatter(rdf, y="SPEED_LIMIT", x="DAY_OF_WEEK", color="ACCIDENT_SEVERITY")
fig.update_traces(marker_size=10)
fig.update_layout(barmode="group", bargap=0.75)
fig.show()


# In[80]:


##the figure above shows that - 
##we have less fatal accident on friday
##we have more fatal accident at speed limit 30 and more slight accident at speed limit 70


# In[81]:


new_df = rdf[['NUMBER_OF_CASUALTIES','NUMBER_OF_VEHICLES']]
new_df = new_df.cumsum()

plt.figure()
new_df.plot()
plt.legend(loc='best')


# In[82]:


def generateBaseMap(default_location=[55.860916, -4.251433], default_zoom_start=5):
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return base_map


# In[83]:


base_map = generateBaseMap()
base_map


# In[84]:


m = HeatMap(data=rdf[['LATITUDE', 'LONGITUDE', 'NUMBER_OF_CASUALTIES']].groupby(['LATITUDE','LONGITUDE']).sum().reset_index().values.tolist(), radius=7, max_zoom=10).add_to(base_map)
##display number of casualities per each loaction on the map as 'heats' 
m.save('heatmap.html') ##save


# In[85]:


#pca=PCA()
 
X_red = pca.fit_transform(X) 
#X_red = pca.fit_transform(StandardScaler().fit_transform(X))


# In[86]:


X_train, X_test, y_train, y_test = train_test_split(X_red,y,test_size=0.30,random_state=42)


# In[87]:


##without pca, uncomment this code if you want to train using pca
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=42)


# In[88]:


X_train.shape


# In[92]:


model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test) 

mlAcc = accuracy_score(y_test, predictions)
print('The Accuracy of the model is : %s' % mlAcc)
print('\n')
print(classification_report(y_test, predictions))

importances = pd.DataFrame(data={'Attribute': X_train.columns, 'Importance':model.coef_[0]})
importances = importances.sort_values(by='Importance', ascending=False)


# ##The table below shows the sequence of how important each features are in determining model prediction 

# In[93]:


importances


# In[53]:





# In[ ]:




