#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import pickle

filename = "G:/DATA SC/R+Python/Pandas/Data/heart.csv"
df = pd.read_csv(filename)
df.head()


# In[4]:


df.head()


# In[5]:


data = np.loadtxt(filename, delimiter = ",", skiprows=1)
print(data)


# In[7]:


data = np.genfromtxt(filename, delimiter=",", dtype=None, names=True, encoding="utf-8-sig")
print(data)


# In[11]:


print(df['age'].mean())


# In[14]:


#random dataframe
data = np.random.random(size=(5,3))
print(data)
df =pd.DataFrame(data)
df


# In[16]:


df = pd.DataFrame(np.random.random(size=(100000,4)), columns= ["A", "B", "C","D"])
df.head()


# In[18]:


df.to_csv("G:/DATA SC/R+Python/Pandas/Data/save.csv", index = False, float_format="%0.4f")
#df.to_hdf("G:/DATA SC/R+Python/Pandas/Data/save.hdf", key="data", format="table")


# In[20]:


filename = "G:/DATA SC/R+Python/Pandas/Data/astronauts.csv"
df = pd.read_csv(filename)
df.head()


# In[24]:


df.sample(2)


# In[25]:


df.describe()


# In[26]:


df.info()


# In[27]:


#Basic Plots
filename = "G:/DATA SC/R+Python/Pandas/Data/heart.csv"
df = pd.read_csv(filename)
df.head()
chest_pain = df.groupby(by="cp").median().reset_index()
chest_pain.head()


# In[28]:


chest_pain.plot.bar(x="cp", y = "chol");


# In[33]:


#seaborn for basic exploratory data plots
#visualization by pandas
import seaborn as sb
df.age.plot.hist(bins=30);


# In[36]:


sb.violinplot(data=df[["trestbps", "thalach"]], inner="quartile", bw=0.2);


# In[37]:


df = pd.read_csv("G:/DATA SC/R+Python/Pandas/Data/meteorite-landings.csv")
df.head()


# In[38]:


df.info()


# In[39]:


df= df.dropna(subset = ["reclong", "reclat"])
df = df[df.reclong<300]


# In[42]:


import matplotlib.pyplot as plt
plt.hist2d(df.reclong, df.reclat, bins=200, vmax=4);
plt.colorbar();


# In[44]:


from sklearn import manifold #manifold learning


# In[45]:


#basic manipulation
df = pd.read_csv("G:/DATA SC/R+Python/Pandas/Data/AB_NYC_2019.csv")
df.head(3)


# In[47]:


#indexing - refering to each row via unique identifier
df2 =df.set_index("id")
df2.head(3)


# In[ ]:




