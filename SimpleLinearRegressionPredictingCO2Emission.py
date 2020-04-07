#!/usr/bin/env python
# coding: utf-8

# In[43]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylab as pl
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[44]:


pwd


# In[45]:


#reading the data and 
df = pd.read_csv(r"E:\NAVTCCAI\MyWork\models\Co2_Emission\FuelConsumptionCo2.csv")
df.head()


# In[46]:


#summerize data
df.describe()


# In[47]:


cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(10)


# In[48]:


viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()


# In[49]:


plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel('FULECONSUMPTION_COMB')
plt.ylabel('Co2EMISSIONS')
plt.show()


# In[50]:


plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS, color='red')
plt.xlabel('ENGINESIZE')
plt.ylabel('Co2EMISSIONS')
plt.show()


# In[51]:


plt.scatter(cdf.CYLINDERS,cdf.CO2EMISSIONS, color='blue')
plt.xlabel('CYLINDERS')
plt.xlabel('Co2EMISSIONS')
plt.show()


# In[52]:


#Train / Test Split
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# In[53]:


#Analyzing the Train data
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='red')
plt.xlabel('ENGINESIZE')
plt.ylabel('Co2EMISSIOM')
plt.show()


# In[54]:


#Model Training
from sklearn import linear_model
reg = linear_model.LinearRegression()
x_train = np.asanyarray(train[['ENGINESIZE']])
y_train = np.asanyarray(train[['CO2EMISSIONS']])
reg.fit (x_train, y_train)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# In[55]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel('ENGINESIZE')
plt.ylabel('C02EMISSIONS')
plt.plot(x_train, regr.coef_[0][0]*x_train + regr.intercept_[0], '-r')


# In[57]:


#Model Evaluation (testing)
from sklearn.metrics import r2_score

x_test = np.asanyarray(test[['ENGINESIZE']])
y_test = np.asanyarray(test[['CO2EMISSIONS']])
y_test_ = reg.predict(x_test)

#Error Finding 
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_test_ - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_test_ - y_test) ** 2))
print("R2-score: %.2f" % r2_score(y_test_ , y_test) )


# In[ ]:




