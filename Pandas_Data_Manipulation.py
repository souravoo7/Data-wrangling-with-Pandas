#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import pickle

#basic manipulation
df = pd.read_csv("G:/DATA SC/R+Python/Pandas/Data/AB_NYC_2019.csv")
df.head(3)


# In[12]:


df = df.set_index("id")#set an index as desired


# In[13]:


df = df.reset_index()#reset the index
df


# In[15]:


#sorting
df = df.sort_values(["neighbourhood_group","host_name"])
df.head(3)


# In[16]:


df.neighbourhood_group.value_counts()


# In[18]:


df[["id", "host_name", "price"]].head(5)


# In[21]:


df.neighbourhood_group.unique()


# In[26]:


##slicing and filtering
#single condition
df_taz = df[df.host_name=="Taz"]
df_taz.head(5)


# In[28]:


#multiple condition
quick_and_cheap = (df.price <100) & (df.minimum_nights<3)
df[quick_and_cheap].head(3)


# In[29]:


review_cons = np.logical_or((df.reviews_per_month >3), (df.number_of_reviews>50))
df[review_cons].head(3)


# In[30]:


#filtering rows and columns togerther
df.loc[review_cons, ["name", "host_name"]]


# In[31]:


'''Basic Relpacing & Thresholding'''


# In[32]:


df.info()


# In[35]:


df = df.dropna()
df.info()


# In[38]:


#THRESHOLDING
import matplotlib.pyplot as plt
plt.hist(df.price.clip(upper=1000));


# In[39]:


#Adding rows/columns
df = pd.read_csv("G:/DATA SC/R+Python/Pandas/Data/astronauts.csv")
df.head(3)


# In[42]:


birthdate = pd.to_datetime(df["Birth Date"], format="%m/%d/%Y")
birthdate


# In[46]:


df["Military Rank"].unique()
df["Military Rank"] = df["Military Rank"].astype("category")


# In[49]:


df2 = df[["Name", "Year", "Group"]]
df2.head(5)


# In[50]:


df2.drop(columns="Group").head(3)


# In[51]:


#Adding columns
df2["Col_Assg"] = "default"
df2.head(3)


# In[52]:


df2.head(3)


# In[53]:


data = np.round(np.random.normal(size=(4,3)),2)
df = pd.DataFrame(data, columns=["A", "B", "C"])
df.head(5)


# In[54]:


df.apply(lambda x:1+np.abs(x))


# In[55]:


#User defined vectorized functions
data2 =np.random.normal(10,2, size=(100000, 2))
df = pd.DataFrame(data2, columns=["X", "Y"])

def hypot(x,y):
    return np.sqrt(x**2+y**2)

h1=[]

for index, (x,y) in df.iterrows():
    h1.append(hypot(x,y))

print(h1[0])


# In[57]:


#Data Grouping
df = pd.read_csv("G:/DATA SC/R+Python/Pandas/Data/train.csv", low_memory=False)
df.head(3)


# In[58]:


dfg = df.groupby("Store")
dfg.mean()#mean aggregation by Store


# In[ ]:


dfg.plot.scatter("Store", "Sales", s=3, title = "Avg. Sales per Store");


# In[60]:


store_day = df.groupby(["Store", "DayOfWeek"], as_index=False).mean()
store_day.head()


# In[68]:


plt.plot(store_day.head(30).DayOfWeek, store_day.head(30).Sales);


# In[69]:


#Continuous Grouping
bins = [0, 2000, 4000, 6000, 8000, 10000, 50000]
cuts = pd.cut(df.Sales, bins, include_lowest=True)
df["SalesGroup"] = cuts
df.head()


# In[70]:


df.groupby(["Store", "SalesGroup"]).DayOfWeek.value_counts()


# In[71]:


df.groupby(["Store", "SalesGroup"]).DayOfWeek.value_counts().unstack(fill_value=0)


# In[72]:


plt.hist(df.Sales);


# In[73]:


df =df[df.Open==1]#remove closed stores data


# In[74]:


plt.hist(df.Sales);


# In[77]:


mask = np.random.choice(10, size=df.shape[0]) == 0

df["NewSales"] = df.Sales.copy()

df.loc[mask, "NewSales"] = np.nan

plt.hist(df.Sales, label = "Original", histtype = "step")
plt.hist(df.NewSales.fillna(0), label = "Corrupted", histtype = "step");


# In[78]:


#fixing corrupt data
test_fix =df.NewSales.transform(lambda x:x.fillna(x.mean()))


# In[79]:


plt.hist(test_fix, bins=100);


# In[81]:


df.groupby(["Store", "DayOfWeek"]).median().head()


# In[84]:


test_fix = df.groupby(["Store", "DayOfWeek"]).median().NewSales.transform(lambda x:x.fillna(x.mean()))
plt.hist(test_fix, bins=100);


# In[83]:





# In[ ]:




