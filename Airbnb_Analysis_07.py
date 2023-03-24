#!/usr/bin/env python
# coding: utf-8

# # Airbnb Data modelling
# ###  Wilson adejo
# ###  01-02-2021

# In[2]:


#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_profiling as pp
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import r2_score
import seaborn as sns
#import geopandas
import shapely
#from shapely.geometry import Point
import missingno as msn

get_ipython().run_line_magic('matplotlib', 'inline')


# # Read in the dataset

# In[3]:


df=pd.read_csv("Airbnb.csv")
df.head(2)
df.sample(3)  # sample  show the data randomly


# In[4]:


## Quick summary of the Airbnb data
df.describe()
df['price'].describe()


# In[5]:


df.groupby('room_type').id.count()


# # Grouping By Columns

# In[6]:


df.groupby('property_type').id.count()


# In[7]:


grouped_df = df.groupby(["property_type", "room_type"])['price'].mean().reset_index()
print (grouped_df)


# In[8]:


grouped_df.plot(kind='bar', title = 'Property-type and Price Per Night')
plt.ylabel("Price")


# In[9]:


## Grouping by room type and price
grouped_df2 = df.groupby(["room_type"])['price'].mean().reset_index()

print (grouped_df2)


# In[10]:



#grouped_df2.plot(kind="bar",title = 'Room-type against price', plt.ylabel("Price"))


# ##  Group By One Column and Get Mean, Min, and Max values by Group

# In[11]:



df.groupby("room_type")['price'].mean().plot(kind='bar',title = 'Room-type against average price per night')
plt.ylabel("Price")


# In[12]:


## Room type and  price per night
grouped_room = df.groupby('room_type').agg({'price': ['mean', 'min', 'max']})
grouped_room

# rename columns
grouped_room.columns = ['Price_mean', 'Price_min', 'Price_max']

# reset index to get grouped columns back
grouped_room = grouped_room.reset_index()

print(grouped_room)


# In[13]:


grouped_room.plot(kind='bar',title = 'List of Room-type with prices')
plt.ylabel("Price")


# ##  Group By two Columns - Room type  and Minimum Night

# In[14]:


grouped_min_night = df.groupby('room_type')['minimum_nights'].mean().plot(kind='bar',title = 'List of Room-type with minimum night requiremet')
plt.ylabel("Minimum Night of Stay")
grouped_min_night


# In[15]:


## Private room has the a lower than home and apartment
df.groupby("room_type")['minimum_nights'].mean().plot(kind='bar',title = 'List of Room-type with minimum night requiremet')
plt.ylabel("Minimum Night of Stay")


# In[16]:


## List of Property-type with minimum night requirement
df.groupby("property_type")['minimum_nights'].mean().sort_values(ascending=True).plot(kind='bar',title = 'List of Property-type with minimum night requirement')
plt.ylabel("Minimum Night of Stay")
#plt.figure(figsize=(70,40))


# In[17]:



grouped_multiple = df.groupby(['property_type','room_type']).agg({'price': ['mean', 'min', 'max']})
grouped_multiple.columns = ['Price_mean', 'Price_min', 'Price_max']
grouped_multiple = grouped_multiple.reset_index()
grouped_multiple


# In[18]:


df.groupby("property_type")['price'].mean().plot(kind='bar',figsize = (12,7),title = 'Property _type by  average price per night')
plt.ylabel("Price per Night (£)")


# In[19]:


##Arrange in ascending order
df.groupby("property_type")['price'].mean().sort_values(ascending=True).plot(kind='bar',figsize = (12,7),title = 'Property _type by  average price per night')
plt.ylabel("Price per Night (£)")


# In[ ]:





# In[20]:


df.groupby("bedrooms")['price'].mean().plot(kind='bar')
plt.xlabel("Number of bedroom")
plt.ylabel("Price")


# In[21]:


#Property type and distance  from city centre
df.groupby("property_type")['distance_from_centre'].count().plot(kind='bar',figsize = (12,7),title = 'Property Type based on distance from city centre') ## make the graphy bigger


# In[22]:


df.groupby('distance_from_centre').quantile(0.50)


# In[23]:


df.groupby('room_type').id.count().plot(kind="bar")


# In[24]:


# forming ProfileReport and save as output.html file 
profile = pp.ProfileReport(df) 
#profile.to_file("output.html")
profile


# # Preprocessing of data

# In[25]:


# Drop all unnecessary columns 
df.drop(["name",'id','neighbourhood','description'], axis=1, inplace = True)
df.head(3)


# In[26]:


#Convert Categorical Data to Numerical Data
df.host_is_superhost = df.host_is_superhost.astype(int)
df["bedrooms"] = df["bedrooms"].astype(str).astype(float)
df["beds"] = df["beds"].astype(str).astype(float)


# In[27]:


df.head()


# In[28]:


### Convert labels to number using dictionary 
#possible_labels=df.property_type.unique()

#label_dict={}
#for index,possible_label in enumerate(possible_labels): label_dict[possible_label]=index
#label_dict
#df["Property_type"]=df.property_type.replace(label_dict)
#df.head(2)

import category_encoders as ce
ce_ordinal = ce.OrdinalEncoder(cols=['property_type']) # create an object of the OrdinalEncoding

df=ce_ordinal.fit_transform(df)  # fit and transform and you will get the encoded data
df.head()


# In[29]:


import category_encoders as ce
ce_ordinal = ce.OrdinalEncoder(cols=['room_type']) # create an object of the OrdinalEncoding

df=ce_ordinal.fit_transform(df)  # fit and transform and you will get the encoded data
df.head()


# In[30]:


# Drop all unnecessary columns 
#df.drop(['property_type'], axis=1, inplace = True)
#df.head(2)


# In[31]:


# Rearrange columns
df1 = df[['host_is_superhost','latitude','longitude','accommodates','property_type','room_type','bedrooms','beds','distance_from_centre','price']]
df1.head()


# In[ ]:





# In[32]:


print(df.columns)
df.isnull().sum()  #there is no null data


# In[33]:


# filling missing value using fillna()   
df.fillna(1) 
#filling a null values using fillna()  
df["beds"].fillna("1", inplace = True)
df["bedrooms"].fillna("1", inplace = True)
#df["bathrooms_text"].fillna("1", inplace = True)


# In[34]:


df.isnull().sum()  #check there is no null data


# # Exploartory data analysis 2

# In[35]:


# finding the  correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns
correlation = df.corr()
plt.figure(figsize = (20,20))
sns.heatmap(correlation,annot=True)


# In[36]:


# sorting the correlation in descending order
correlation['price'].sort_values(ascending=False)


# In[37]:


#Not surprisingly correlation between price and accomodates, distance from centre, longitude and latitude,n other features.
#It means closer to city center, bigger price. Thesame applies to accommodates


# # Using multiple linear regression
# 

# In[38]:


from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np


# In[39]:


# check that a linear relationship exists

plt.scatter(df1["accommodates"],df1["price"], color="red")
plt.title( "Analysis 101")
plt.xlabel("Accommodates", fontsize=14)
plt.ylabel("Price", fontsize=14)
plt.grid(True)
plt.show()


# In[40]:


# check that a linear relationship exists
plt.scatter(df1["distance_from_centre"],df1["price"],color="blue")

plt.title( "Analysis 102")
plt.xlabel("Distance_from centre", fontsize=14)
plt.ylabel("Price", fontsize=14)
plt.grid(True)
plt.show()


# # Method 1-Multiple linear regression and Feature importance

# In[41]:


# Drop all unnecessary columns 
df1.drop(["latitude",'longitude'], axis=1, inplace = True)
df1.head(2)


# In[42]:


import matplotlib.pyplot as plt
import seaborn as sns
correlation2 = df1.corr()
plt.figure(figsize = (20,20))
sns.heatmap(correlation,annot=True)


# In[43]:


correlation2['price'].sort_values(ascending=False)


# In[44]:


# filling missing value using fillna()   
df1.fillna(1) 
#filling a null values using fillna()  
df1["beds"].fillna("1", inplace = True)
df1["bedrooms"].fillna("1", inplace = True)


# In[45]:


X=df1[["host_is_superhost","accommodates","property_type","room_type","bedrooms","beds","distance_from_centre"]]
Y=df1["price"]


# In[46]:


# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)


# In[47]:


from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# In[48]:


# checking prediction with sklearn using one of the Airbnb data
#"host_is_superhost","accommodates","property_type","room_type","bedrooms","beds","distance_from_centre"


host_is_superhost =1
accommodates = 2
property_type= 1
room_type=1
beds=1
bedrooms=1
distance_from_centre = 1.19


print ('Predicted Price: \n', regr.predict([[host_is_superhost,accommodates,property_type,room_type,bedrooms,beds,distance_from_centre]]))


# # Method 2  Feature Importance using coefficient of model

# In[49]:


# define the model
# X=["host_is_superhost","accommodates","property_type","room_type","bedrooms","beds","distance_from_centre"]
# Y= [price]

X = df1.iloc[:, :-1]
Y = df1.iloc[:, -1]

#model = LinearRegression()
model=linear_model.LinearRegression()
# fit the model
model.fit(X, Y)

# get importance
importance = model.coef_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
    
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.title("Bar Chart of MLR Coefficients as Feature Importance Scores")
plt.xlabel("Feature")
plt.ylabel("Contribution to Price per Night")
plt.show()


# ###  Model Evaluation of Model- Method 1

# In[50]:


df1.head(2)


# In[51]:


###### fill in missing data
#df1 = df1.fillna(lambda x: x.median())
df1 = df1.fillna(method='ffill')
df1.isnull().any()
df1['beds'] = pd.to_numeric(df1['beds'], errors='coerce').fillna(0)
df1['bedrooms'] = pd.to_numeric(df1['bedrooms'], errors='coerce').fillna(0)

df1.dtypes


# In[52]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X = df1.iloc[:,:-1].values #price is the last columns. this line means handle all the columns except the last one
y = df1.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 42)
linReg=LinearRegression()
linReg.fit(x_train,y_train)


# In[53]:



y_pred = linReg.predict(x_test)

## using  different metrics
test_set_rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
test_set_r2 = r2_score(y_test, y_pred)

print("RMSE is :",test_set_rmse) # Note that for rmse, the lower that value is, the better the fit
print("The r2 is:",test_set_r2)  # The closer towards 1, the better the fit


# In[54]:


#df1.to_csv("Airbnb2.csv")   #saving the current df1- propably for further use


# ###  Model Evaluation of Model Method 2

# In[55]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x=df1[["host_is_superhost","accommodates","property_type","room_type","bedrooms","beds","distance_from_centre"]]
y=df1["price"]


# In[56]:


#Model
regressor=LinearRegression()
regressor.fit(x,y)


# In[57]:


#Accuracy
predictions= regressor.predict(x)

mae =0
for i in range (0,len(predictions)):
    prediction =predictions[i]
    actual =y.iloc[i]
    error=abs(actual -prediction)
    mae=mae+error
    
    mae=mae/len(predictions)


# In[58]:


mae


# 
