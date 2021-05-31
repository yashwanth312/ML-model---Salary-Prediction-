#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas


# In[2]:


dataset = pandas.read_csv("SalaryData.csv")


# In[3]:


dataset.info()


# In[4]:


y = dataset["Salary"]


# In[5]:


x = dataset["YearsExperience"]


# In[6]:


x = x.values


# In[7]:


x = x.reshape(30,1)


# In[8]:


x.shape


# In[9]:


from sklearn.linear_model import LinearRegression


# In[10]:


model = LinearRegression()


# In[11]:


from sklearn.model_selection import train_test_split


# # Splitting the data into training and testing data 

# In[12]:


x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[13]:


model.fit(x_train , y_train)


# In[14]:


y_pred = model.predict(x_test)


# In[15]:


y_pred


# In[16]:


y_test


# # Comparing the test values and predicted values graphically 

# In[17]:


import matplotlib.pyplot as plt


# In[18]:


plt.scatter(x_test , y_test)


# In[19]:


plt.plot(x_test , y_pred)


# In[20]:


import seaborn as sns


# In[21]:


sns.set()


# In[22]:


plt.scatter(x_test , y_test , color="red")
plt.plot(x_test , y_pred)
plt.xlabel("Years of Experience")
plt.ylabel("Salary Expected")


# In[ ]:





# In[23]:


import joblib


# In[24]:


joblib.dump(model , "trained_model.h5")


# In[ ]:




