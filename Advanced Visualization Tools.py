#!/usr/bin/env python
# coding: utf-8

# ### Importing required libraries

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image


# ### Loading data

# In[2]:


df = pd.read_excel(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx',
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)

print('Data read into a pandas dataframe!')


# In[3]:


# in pandas axis=0 represents rows (default) and axis=1 represents columns.
df.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)
df.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)
df['Total'] = df.sum(axis=1)
df.set_index('Country', inplace=True)


# In[4]:


df.head()


# # Waffle Charts <a id="6"></a>
# 
# A `waffle chart` is an interesting visualization that is normally created to display progress toward goals. It is commonly an effective option when you are trying to add interesting visualization features to a visual that consists mainly of cells, such as an Excel dashboard.
# 

# Let's revisit the previous case study about Pakistan, China, and India.
# 

# In[5]:


# let's create a new dataframe for these three countries 
df_pci = df.loc[['Pakistan', 'China', 'India'], :]

# let's take a look at our dataframe
df_pci


# Unfortunately, unlike R, `waffle` charts are not built into any of the Python visualization libraries. Therefore, we will learn how to create them from scratch.
# 

# **Step 1.** The first step into creating a waffle chart is determing the proportion of each category with respect to the total.
# 

# In[6]:


# compute the proportion of each category with respect to the total
total_values = df_pci['Total'].sum()
category_proportions = df_pci['Total'] / total_values

# print out proportions
pd.DataFrame({"Category Proportion": category_proportions})


# **Step 2.** The second step is defining the overall size of the `waffle` chart.
# 

# In[7]:


width = 40 # width of chart
height = 10 # height of chart

total_num_tiles = width * height # total number of tiles

print(f'Total number of tiles is {total_num_tiles}.')


# **Step 3.** The third step is using the proportion of each category to determe it respective number of tiles
# 

# In[8]:


# compute the number of tiles for each category
tiles_per_category = (category_proportions * total_num_tiles).round().astype(int)

# print out number of tiles per category
pd.DataFrame({"Number of tiles": tiles_per_category})


# Based on the calculated proportions, Pakistan will occupy 61 tiles of the `waffle` chart, China will occupy 166 tiles, and India will occupy 174 tiles.
# 

# **Step 4.** The fourth step is creating a matrix that resembles the `waffle` chart and populating it.
# 

# In[9]:


# initialize the waffle chart as an empty matrix
waffle_chart = np.zeros((height, width), dtype = np.uint)

# define indices to loop through waffle chart
category_index = 0
tile_index = 0

# populate the waffle chart
for col in range(width):
    for row in range(height):
        tile_index += 1

        # if the number of tiles populated for the current category is equal to its corresponding allocated tiles...
        if tile_index > sum(tiles_per_category[0:category_index]):
            # ...proceed to the next category
            category_index += 1       
            
        # set the class value to an integer, which increases with class
        waffle_chart[row, col] = category_index
        
print ('Waffle chart populated!')


# Let's take a peek at how the matrix looks like.
# 

# In[10]:


waffle_chart


# As expected, the matrix consists of three categories and the total number of each category's instances matches the total number of tiles allocated to each category.
# 

# **Step 5.** Map the `waffle` chart matrix into a visual.
# 

# In[11]:


# instantiate a new figure object
fig = plt.figure()

# use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()
plt.show()


# **Step 6.** Prettify the chart.
# 

# In[12]:


# instantiate a new figure object
fig = plt.figure()

# use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()

# get the axis
ax = plt.gca()

# set minor ticks
ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
ax.set_yticks(np.arange(-.5, (height), 1), minor=True)
    
# add gridlines based on minor ticks
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

plt.xticks([])
plt.yticks([])
plt.show()


# **Step 7.** Create a legend and add it to chart.
# 

# In[13]:


# instantiate a new figure object
fig = plt.figure()

# use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()

# get the axis
ax = plt.gca()

# set minor ticks
ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
ax.set_yticks(np.arange(-.5, (height), 1), minor=True)
    
# add gridlines based on minor ticks
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

plt.xticks([])
plt.yticks([])

# compute cumulative sum of individual categories to match color schemes between chart and legend
values_cumsum = np.cumsum(df_pci['Total'])
total_values = values_cumsum[len(values_cumsum) - 1]

# create legend
legend_handles = []
for i, category in enumerate(df_pci.index.values):
    label_str = category + ' (' + str(df_pci['Total'][i]) + ')'
    color_val = colormap(float(values_cumsum[i])/total_values)
    legend_handles.append(matplotlib.patches.Patch(color=color_val, label=label_str))

# add legend to chart
plt.legend(handles=legend_handles,
           loc='lower center', 
           ncol=len(df_pci.index.values),
           bbox_to_anchor=(0., -0.2, 0.95, .1)
          )
plt.show()


# And there you go! What a good looking *delicious* `waffle` chart, don't you think?
# 

# Now it would very inefficient to repeat these seven steps every time we wish to create a `waffle` chart. So let's combine all seven steps into one function called *create_waffle_chart*. This function would take the following parameters as input:
# 
# > 1.  **categories**: Unique categories or classes in dataframe.
# > 2.  **values**: Values corresponding to categories or classes.
# > 3.  **height**: Defined height of waffle chart.
# > 4.  **width**: Defined width of waffle chart.
# > 5.  **colormap**: Colormap class
# > 6.  **value_sign**: In order to make our function more generalizable, we will add this parameter to address signs that could be associated with a value such as %, $, and so on. **value_sign** has a default value of empty string.
# 

# In[14]:


def create_waffle_chart(categories, values, height, width, colormap, value_sign=''):

    # compute the proportion of each category with respect to the total
    total_values = sum(values)
    category_proportions = [(float(value) / total_values) for value in values]

    # compute the total number of tiles
    total_num_tiles = width * height # total number of tiles
    print ('Total number of tiles is', total_num_tiles)
    
    # compute the number of tiles for each catagory
    tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]

    # print out number of tiles per category
    for i, tiles in enumerate(tiles_per_category):
        print (df_pci.index.values[i] + ': ' + str(tiles))
    
    # initialize the waffle chart as an empty matrix
    waffle_chart = np.zeros((height, width))

    # define indices to loop through waffle chart
    category_index = 0
    tile_index = 0

    # populate the waffle chart
    for col in range(width):
        for row in range(height):
            tile_index += 1

            # if the number of tiles populated for the current category 
            # is equal to its corresponding allocated tiles...
            if tile_index > sum(tiles_per_category[0:category_index]):
                # ...proceed to the next category
                category_index += 1       
            
            # set the class value to an integer, which increases with class
            waffle_chart[row, col] = category_index
    
    # instantiate a new figure object
    fig = plt.figure()

    # use matshow to display the waffle chart
    colormap = plt.cm.coolwarm
    plt.matshow(waffle_chart, cmap=colormap)
    plt.colorbar()

    # get the axis
    ax = plt.gca()

    # set minor ticks
    ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
    ax.set_yticks(np.arange(-.5, (height), 1), minor=True)
    
    # add dridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    plt.xticks([])
    plt.yticks([])

    # compute cumulative sum of individual categories to match color schemes between chart and legend
    values_cumsum = np.cumsum(values)
    total_values = values_cumsum[len(values_cumsum) - 1]

    # create legend
    legend_handles = []
    for i, category in enumerate(categories):
        if value_sign == '%':
            label_str = category + ' (' + str(values[i]) + value_sign + ')'
        else:
            label_str = category + ' (' + value_sign + str(values[i]) + ')'
            
        color_val = colormap(float(values_cumsum[i])/total_values)
        legend_handles.append(matplotlib.patches.Patch(color=color_val, label=label_str))

    # add legend to chart
    plt.legend(
        handles=legend_handles,
        loc='lower center', 
        ncol=len(categories),
        bbox_to_anchor=(0., -0.2, 0.95, .1)
    )
    plt.show()


# Now to create a `waffle` chart, all we have to do is call the function `create_waffle_chart`. Let's define the input parameters:
# 

# In[15]:


width = 40 # width of chart
height = 10 # height of chart

categories = df_pci.index.values # categories
values = df_pci['Total'] # correponding values of categories

colormap = plt.cm.coolwarm # color map class


# And now let's call our function to create a `waffle` chart.
# 

# In[16]:


create_waffle_chart(categories, values, height, width, colormap)


# # Word Clouds <a id="8"></a>
# 
# `Word` clouds (also known as text clouds or tag clouds) work in a simple way: the more a specific word appears in a source of textual data (such as a speech, blog post, or database), the bigger and bolder it appears in the word cloud.
# 

# Luckily, a Python package already exists in Python for generating `word` clouds. The package, called `word_cloud`. You can learn more about the package by following this [link](https://github.com/amueller/word_cloud/).
# 
# Let's use this package to learn how to generate a word cloud for a given text document.
# 

# First, let's install the package.
# 

# In[18]:


# install wordcloud
# Run this command on your anaconda prompt
#Command:!pip3 install wordcloud

# import package and its set of stopwords
from wordcloud import WordCloud, STOPWORDS

print ('Wordcloud is installed and imported!')


# `Word` clouds are commonly used to perform high-level analysis and visualization of text data. Accordinly, let's digress from the immigration dataset and work with an example that involves analyzing text data. Let's try to analyze a short novel written by **Lewis Carroll** titled *Alice's Adventures in Wonderland*. Let's go ahead and download a *.txt* file of the novel.
# 

# In[19]:


import urllib

# open the file and read it into a variable alice_novel
alice_novel = urllib.request.urlopen("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/alice_novel.txt").read().decode("utf-8")


# In[20]:


alice_novel


# Next, let's use the stopwords that we imported from `word_cloud`. We use the function *set* to remove any redundant stopwords.
# 

# In[21]:


stopwords = set(STOPWORDS)


# Create a word cloud object and generate a word cloud. For simplicity, let's generate a word cloud using only the first 2000 words in the novel.
# 

# In[22]:


# instantiate a word cloud object
alice_wc = WordCloud(
    background_color='white',
    max_words=2000,
    stopwords=stopwords
)

# generate the word cloud
alice_wc.generate(alice_novel)


# Awesome! Now that the `word` cloud is created, let's visualize it.
# 

# In[23]:


# display the word cloud
plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# Interesting! So in the first 2000 words in the novel, the most common words are **Alice**, **said**, **little**, **Queen**, and so on. Let's resize the cloud so that we can see the less frequent words a little better.
# 

# In[24]:


fig = plt.figure(figsize=(14, 18))

# display the cloud
plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# Much better! However, **said** isn't really an informative word. So let's add it to our stopwords and re-generate the cloud.
# 

# In[25]:


stopwords.add('said') # add the words said to stopwords

# re-generate the word cloud
alice_wc.generate(alice_novel)

# display the cloud
fig = plt.figure(figsize=(14, 18))

plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# Excellent! This looks really interesting! Another cool thing you can implement with the `word_cloud` package is superimposing the words onto a mask of any shape. Let's use a mask of Alice and her rabbit. We already created the mask for you, so let's go ahead and download it and call it *alice_mask.png*.
# 

# In[26]:


# save mask to alice_mask
alice_mask = np.array(Image.open(urllib.request.urlopen('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/labs/Module%204/images/alice_mask.png')))


# Let's take a look at how the mask looks like.
# 

# In[28]:


fig = plt.figure(figsize=(14, 18))

plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis('off')
plt.show()


# Shaping the `word` cloud according to the mask is straightforward using `word_cloud` package. For simplicity, we will continue using the first 2000 words in the novel.
# 

# In[29]:


# instantiate a word cloud object
alice_wc = WordCloud(background_color='white', max_words=2000, mask=alice_mask, stopwords=stopwords)

# generate the word cloud
alice_wc.generate(alice_novel)

# display the word cloud
fig = plt.figure(figsize=(14, 18))

plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# Really impressive!
# 

# Unfortunately, our immigration data does not have any text data, but where there is a will there is a way. Let's generate sample text data from our immigration dataset, say text data of 90 words.
# 

# Let's recall how our data looks like.
# 

# In[30]:


df.head()


# And what was the total immigration from 1980 to 2013?
# 

# In[31]:


total_immigration = df['Total'].sum()
total_immigration


# Using countries with single-word names, let's duplicate each country's name based on how much they contribute to the total immigration.
# 

# In[32]:


max_words = 90
word_string = ''
for country in df.index.values:
    # check if country's name is a single-word name
    if country.count(" ") == 0:
        repeat_num_times = int(df.loc[country, 'Total'] / total_immigration * max_words)
        word_string = word_string + ((country + ' ') * repeat_num_times)

# display the generated text
word_string


# We are not dealing with any stopwords here, so there is no need to pass them when creating the word cloud.
# 

# In[33]:


# create the word cloud
wordcloud = WordCloud(background_color='white').generate(word_string)

print('Word cloud created!')


# In[34]:


# display the cloud
plt.figure(figsize=(14, 18))

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# According to the above word cloud, it looks like the majority of the people who immigrated came from one of 15 countries that are displayed by the word cloud. One cool visual that you could build, is perhaps using the map of Canada and a mask and superimposing the word cloud on top of the map of Canada. That would be an interesting visual to build!
# 

# # Regression Plots <a id="10"></a>
# 
# > Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics. 

# In[35]:


# install seaborn
# !pip3 install seaborn

# import library
import seaborn as sns

print('Seaborn installed and imported!')


# Create a new dataframe that stores that total number of landed immigrants to Canada per year from 1980 to 2013.
# 

# In[36]:


years=list(range(1980,2014))
# we can use the sum() method to get the total population per year
df_tot = pd.DataFrame(df[years].sum(axis=0))

# change the years to type float (useful for regression later on)
df_tot.index = map(float, df_tot.index)

# reset the index to put in back in as a column in the df_tot dataframe
df_tot.reset_index(inplace=True)

# rename columns
df_tot.columns = ['year', 'total']

# view the final dataframe
df_tot.head()


# With *seaborn*, generating a regression plot is as simple as calling the **regplot** function.
# 

# In[37]:


sns.regplot(x='year', y='total', data=df_tot)


# This is not magic; it is *seaborn*! You can also customize the color of the scatter plot and regression line. Let's change the color to green.
# 

# In[38]:


sns.regplot(x='year', y='total', data=df_tot, color='green')
plt.show()


# You can always customize the marker shape, so instead of circular markers, let's use `+`.
# 

# In[39]:


ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+')
plt.show()


# Let's blow up the plot a little so that it is more appealing to the sight.
# 

# In[40]:


plt.figure(figsize=(15, 10))
sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+')
plt.show()


# And let's increase the size of markers so they match the new size of the figure, and add a title and x- and y-labels.
# 

# In[41]:


plt.figure(figsize=(15, 10))
ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})

ax.set(xlabel='Year', ylabel='Total Immigration') # add x- and y-labels
ax.set_title('Total Immigration to Canada from 1980 - 2013') # add title
plt.show()


# And finally increase the font size of the tickmark labels, the title, and the x- and y-labels so they don't feel left out!
# 

# In[42]:


plt.figure(figsize=(15, 10))

sns.set(font_scale=1.5)

ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})
ax.set(xlabel='Year', ylabel='Total Immigration')
ax.set_title('Total Immigration to Canada from 1980 - 2013')
plt.show()


# Amazing! A complete scatter plot with a regression fit with 5 lines of code only. Isn't this really amazing?
# 

# If you are not a big fan of the purple background, you can easily change the style to a white plain background.
# 

# In[43]:


plt.figure(figsize=(15, 10))

sns.set(font_scale=1.5)
sns.set_style('ticks')  # change background to white background

ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})
ax.set(xlabel='Year', ylabel='Total Immigration')
ax.set_title('Total Immigration to Canada from 1980 - 2013')
plt.show()


# Or to a white background with gridlines.
# 

# In[44]:


plt.figure(figsize=(15, 10))

sns.set(font_scale=1.5)
sns.set_style('whitegrid')

ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})
ax.set(xlabel='Year', ylabel='Total Immigration')
ax.set_title('Total Immigration to Canada from 1980 - 2013')
plt.show()

