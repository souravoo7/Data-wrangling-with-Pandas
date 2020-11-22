#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import numpy as np
import pandas as pd

df = pd.read_csv("G:/DATA SC/R+Python/Pandas/Data/all_stocks_5yr.csv",
                 parse_dates=["date"],#parse the date column 
                 low_memory=False)
df.head()


# In[2]:


df.describe()


# In[3]:


df2 = df.set_index(["Name","date"]).sort_index()#set an sort the index for the data
df2.head()


# In[7]:


df2.close.loc["AAL"]#get the aal data


# In[8]:


df2.close.loc["AAL"].plot();


# In[9]:


aal = df2.loc["AAL"]#only  AAL data
aal.head()


# In[12]:


aal["2017-02"]


# In[13]:


#REINDEXING
aal.head()#the aal data


# In[14]:


#the start and end-date in the data
start, end =  aal.index.min(), aal.index.max()
print(start, end)


# In[15]:


new_index = pd.date_range(start, end, freq="1D")#create index range from start to finish
new_index


# In[16]:


aal.reindex(new_index, method="ffill")#re-index the data using the new index


# In[18]:


#will it work with multi-index case? NO
start, end =df2.index.levels[1].min(),df2.index.levels[1].max()
print(start, end)
date_range = pd.date_range(start, end)
#cannot fill across levels


# In[19]:


df2 = df2.unstack("Name")


# In[20]:


df2.head()


# In[21]:


new_index = pd.date_range(start, end, freq="1D")#create index range from start to finish
new_index


# In[22]:


df2.reindex(new_index, method="ffill")#re-index the data using the new index


# In[23]:


df2.reindex(new_index, method="ffill").describe()


# In[25]:


df_daily = df2.reindex(new_index, method="ffill")
df_daily.head()


# In[40]:


df_daily_diff = 100 * (df_daily.close-df_daily.open).div(df_daily.open) # calculate the % daily retuens
df_daily_diff.head()


# In[5]:


#TIME ZONES
import pandas as pd

df = pd.read_csv("G:/DATA SC/R+Python/Pandas/Data/data_elonmusk.csv",
                 encoding = 'latin1',
                 parse_dates=["Time"],#parse the date column 
                 low_memory=False)
df.head()


# In[6]:


df.Time.hist(bins=200, grid=False);


# In[8]:


df = df.rename(columns={"Time":"DateTime"})#rename the columns
df.head()


# In[9]:


df["Time"] = df.DateTime.dt.time #create a new column that extracts the time from the date
df.head()


# In[11]:


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()#to handle missing NaNs
df.Time.hist(bins=200, grid=False);


# In[12]:


df.DateTime.dt.hour.value_counts().sort_index().plot.bar();


# In[14]:


df.DateTime = df.DateTime.dt.tz_localize("Europe/Istanbul")#converetd already


# In[15]:


df.head()


# In[16]:


df["LATime"] = df.DateTime.dt.tz_convert("America/Los_Angeles")


# In[17]:


df.LATime.dt.hour.value_counts().sort_index().plot.bar();

