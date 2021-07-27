#!/usr/bin/env python
# coding: utf-8

# # 0. Install and Import Dependencies

# In[ ]:


get_ipython().system('pip install neuralprophet')


# In[ ]:


# https://www.kaggle.com/jsphyg/weather-dataset-rattle-package


# In[ ]:


import pandas as pd
from neuralprophet import NeuralProphet
from matplotlib import pyplot as plt
import pickle


# # 1. Read in Data and Process Dates

# In[ ]:


df = pd.read_csv('weatherAUS.csv')
df.head()


# In[ ]:


df.Location.unique()


# In[ ]:


df.columns


# In[ ]:


melb = df[df['Location']=='Melbourne']
melb['Date'] = pd.to_datetime(melb['Date'])
melb.head()


# In[ ]:


plt.plot(melb['Date'], melb['Temp3pm'])
plt.show()


# In[ ]:


melb['Year'] = melb['Date'].apply(lambda x: x.year)
melb = melb[melb['Year']<=2015]
plt.plot(melb['Date'], melb['Temp3pm'])
plt.show()


# In[ ]:


data = melb[['Date', 'Temp3pm']] 
data.dropna(inplace=True)
data.columns = ['ds', 'y'] 
data.head()


# # 2. Train Model

# In[ ]:


m = NeuralProphet()


# In[ ]:


model = m.fit(data, freq='D', epochs=1000)


# # 3. Forecast Away

# In[ ]:


future = m.make_future_dataframe(data, periods=900)
forecast = m.predict(future)
forecast.head()


# In[ ]:


plot1 = m.plot(forecast)


# In[ ]:


plt2 = m.plot_components(forecast)


# # 4. Save Model

# In[ ]:


with open('saved_model.pkl', "wb") as f:
    pickle.dump(m, f)


# In[ ]:


del m


# In[ ]:


with open('saved_model.pkl', "rb") as f:
    m = pickle.load(f)


# In[ ]:


future = m.make_future_dataframe(data, periods=900)
forecast = m.predict(future)
forecast.head()


# In[ ]:


plot1 = m.plot(forecast)


# In[ ]:




