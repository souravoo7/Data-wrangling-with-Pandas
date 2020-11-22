#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("G:/DATA SC/R+Python/Pandas/Data/flights2.csv.gz", low_memory=False)
df.head()


# In[5]:


#explore the reasons for flight delay
df_c = df.set_index(["DESTINATION_AIRPORT", "AIRLINE"])
df_c = df_c.sort_index()
df_c.head()


# In[8]:


df_c.xs(("LAX","AA")).head()#look at a select cross-section


# In[32]:


dfc =df.set_index(["DESTINATION_AIRPORT", "AIRLINE"])
dfc.sort_index()
dfc.head()


# In[30]:


df_select = df[ (df.AIRLINE == "AA") &(df.DESTINATION_AIRPORT == "LAX")]
df_select.head()


# In[28]:


df_new = dfc.xs(("LAX", "AA"))
df_new.head()


# In[34]:


dfc.index


# In[37]:


airlines = df.AIRLINE.to_numpy()
destination = df.DESTINATION_AIRPORT.to_numpy()

display(airlines, destination)


# In[39]:


pd.MultiIndex.from_arrays([airlines, destination])


# In[43]:


tuples = [tuple(x) for x in df[["DESTINATION_AIRPORT", "AIRLINE"]][:10].to_numpy()]#take the first 10 rows and create a tuple
tuples


# In[46]:


pd.MultiIndex.from_tuples(tuples)


# In[47]:


#stacking and unstacking of data
import seaborn as sb

df = pd.read_csv("G:/DATA SC/R+Python/Pandas/Data/train.csv", low_memory=False)
df.dropna()
df.head()


# In[48]:


df_index = df.set_index(["Store", "Date"])
df_index = df_index.sort_index()
df_index.head()


# In[51]:


#stacking will work only when the index is the primary key
df_index.Sales.unstack("Store")


# In[58]:


x = df.set_index(["Store", "Date"])[["Sales"]]#double brackets to get the datfarame and not a series
x.head()


# In[59]:


x=x.unstack("Date")
x


# In[61]:


#normalize the sales data by store to get the pure day variations
means = x.mean(axis=1)
x_norm = x.div(means, axis=0)#row by row


# In[63]:


sb.heatmap(data=x_norm);#heatmap in seaborn


# In[64]:


import numpy as np

df_pres = pd.read_csv("G:/DATA SC/R+Python/Pandas/Data/pres16results.csv", low_memory=False)
df_pres.head()


# In[66]:


top = df_pres.loc[df_pres.fips =="US", ["cand", "votes"]].sort_values("votes", ascending=False)
top.head()


# In[68]:


candidates= top.cand.head()# selects the top 5
candidates


# In[70]:


df_top5  = df_pres[df_pres.cand.isin(candidates)]
df_top5


# In[72]:


p = df_top5.pivot(index="fips", #row
                  columns="cand", #column
                  values="votes")#values
p


# In[73]:


df_count = pd.read_csv("G:/DATA SC/R+Python/Pandas/Data/happiness.csv", 
                       low_memory=False)
df_count.head()


# In[75]:


df_count.pivot_table(index = "Country",
                    values = "Score",
                    aggfunc=[np.mean, np.std,np.median])


# In[80]:


df_count.pivot_table(index = "Country",
                     columns = "Year",
                     values = ["Score", "GDP per capita"],
                     aggfunc=[np.mean],
                     margins=True,
                    margins_name = "Overall")


# In[92]:


result = df_count.pivot_table(index = "Country",
                     columns = "Year",
                     values = ["Score", "GDP per capita"],
                     aggfunc=[np.mean],
                     margins=True,
                    margins_name = "Overall")
result.head()


# In[3]:


import pandas as pd
import numpy as np

df =pd.read_csv("G:/DATA SC/R+Python/Pandas/Data/heroes_information.csv", 
                       low_memory=False)
df.head()


# In[4]:


df.drop(columns=df.columns[0], inplace=True)
df.head()


# In[6]:


df = df.replace(-99,np.NaN)
pd.plotting.scatter_matrix(df);


# In[7]:


##Crosstab
pd.crosstab(index=df["Skin color"], 
            columns=df["Eye color"])


# In[11]:


df.pivot_table(index="Skin color",
              columns="Eye color",
              values="Alignment",
              aggfunc="count",
              dropna=False,
              fill_value=0)


# In[15]:


100 * pd.crosstab(index=df["Gender"], 
            columns=df["Alignment"],
            margins=True,
            normalize="all")#count by default, normalize to get the proportions


# In[18]:


res = df.fillna(0).pivot_table(index      = "Gender",
                               columns    = "Alignment",
                               values     = "Height",
                               aggfunc    = "count",
                               margins    = True,
                               fill_value = 0)
res


# In[ ]:




