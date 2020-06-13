#!/usr/bin/env python
# coding: utf-8

# In[3]:


import warnings
warnings.filterwarnings("ignore")


# In[11]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
sns.set_style("whitegrid")
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
os.chdir("C:/Users/dell/Downloads/walmart-recruiting-store-sales-forecasting")


# In[21]:


train=pd.read_csv("C:/Users/dell/Downloads/walmart-recruiting-store-sales-forecasting/train.csv")


# In[22]:


test=pd.read_csv('C:/Users/dell/Downloads/walmart-recruiting-store-sales-forecasting/test.csv')


# In[23]:


store=pd.read_csv('C:/Users/dell/Downloads/walmart-recruiting-store-sales-forecasting/stores.csv')


# In[24]:


feature=pd.read_csv('C:/Users/dell/Downloads/walmart-recruiting-store-sales-forecasting/features.csv')


# In[25]:


train.head()


# In[26]:


test.head()


# In[27]:


store.head()


# In[28]:


feature.head()


# In[29]:


merge_df=pd.merge(train,feature, on=['Store','Date'], how='inner')


# In[30]:


merge_df.head()


# In[31]:


merge_df.describe().transpose()


# In[32]:


from datetime import datetime as dt


# In[33]:


merge_df['DateTimeObj']=[dt.strptime(x,'%Y-%m-%d') for x in list(merge_df['Date'])]
merge_df['DateTimeObj'].head()


# In[34]:


plt.plot(merge_df[(merge_df.Store==1)].DateTimeObj, merge_df[(merge_df.Store==1)].Weekly_Sales, 'ro')
plt.show()


# In[35]:


weeklysales=merge_df.groupby(['Store','Date'])['Weekly_Sales'].apply(lambda x:np.sum(x))
weeklysales[0:5]


# In[36]:


weeklysaledept=merge_df.groupby(['Store','Dept'])['Weekly_Sales'].apply(lambda x:np.sum(x))
weeklysaledept[0:5]


# In[37]:


weeklyscale=weeklysales.reset_index()
weeklyscale[0:5]


# In[38]:


walmartstore=pd.merge(weeklyscale, feature, on=['Store', 'Date'], how='inner')
walmartstore.head()


# In[39]:


walmartstoredf = walmartstore.iloc[:, list(range(5)) + list(range(10,13))]


# In[40]:


walmartstoredf.head()


# In[41]:


walmartstoredf['DateTimeObj'] = [dt.strptime(x, '%Y-%m-%d') for x in list(walmartstoredf['Date'])]
weekNo=walmartstoredf.reset_index()


# In[42]:


weekNo = [(x - walmartstoredf['DateTimeObj'][0]) for x in list(walmartstoredf['DateTimeObj'])]


# In[43]:


walmartstoredf['Week'] = [np.timedelta64(x, 'D').astype(int)/7 for x in weekNo]


# In[44]:


walmartstoredf.head()


# In[45]:


plt.plot(walmartstoredf.DateTimeObj, walmartstoredf.Weekly_Sales, 'ro')
plt.show()


# In[46]:


walmartstoredf['IsHolidayInt'] = [int(x) for x in list(walmartstoredf.IsHoliday)]


# In[47]:


walmartstoredf.head()


# In[48]:


walmartstoredf.Store.unique()


# In[50]:


train_WM, test_WM = train_test_split(walmartstoredf, test_size=0.3,random_state=42)


# In[51]:


plt.plot(walmartstoredf[(walmartstoredf.Store==1)].Week, walmartstoredf[(walmartstoredf.Store==1)].Weekly_Sales, 'ro')
plt.show()


# In[52]:


XTrain = train_WM[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Week', 'IsHolidayInt']]
YTrain = train_WM['Weekly_Sales']


# In[53]:


XTest = test_WM[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Week', 'IsHolidayInt']]
YTest = test_WM['Weekly_Sales']


# In[54]:


wmLinear = linear_model.LinearRegression(normalize=True)
wmLinear.fit(XTrain, YTrain)


# In[55]:


wmLinear.coef_


# In[56]:


YHatTest = wmLinear.predict(XTest)


# In[58]:


plt.plot(YTest, YHatTest,'ro')
plt.plot(YTest, YTest,'b-')
plt.show()


# In[59]:


walmartstoredf['Store'].unique()


# In[60]:


Store_Dummies = pd.get_dummies(walmartstoredf.Store, prefix='Store').iloc[:,1:]
walmartstoredf = pd.concat([walmartstoredf, Store_Dummies], axis=1)


# In[61]:


walmartstoredf.head()


# In[62]:


train_WM, test_WM = train_test_split(walmartstoredf, test_size=0.3,random_state=42)
XTrain = train_WM.iloc[:,([3,4,5,6] + [9,10]) + list(range(11,walmartstoredf.shape[1]))]
yTrain = train_WM.Weekly_Sales
                                                    
XTest = test_WM.iloc[:,([3,4,5,6] + [9,10]) + list(range(11,walmartstoredf.shape[1]))]
yTest=test_WM.Weekly_Sales


# In[63]:


XTrain.head()


# In[64]:


wmLinear = linear_model.LinearRegression(normalize=True)
wmLinear.fit(XTrain, YTrain)


# In[65]:


YHatTest = wmLinear.predict(XTest)
plt.plot(YTest, YHatTest,'ro')
plt.plot(YTest, YTest,'b-')
plt.show()


# In[66]:


MAPE = np.mean(abs((YTest - YHatTest)/YTest))
MSSE = np.mean(np.square(YHatTest - YTest))

print(MAPE, MSSE)


# In[67]:


from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error


# In[68]:


alphas = np.linspace(10, 20, 10)


# In[69]:


testError = np.empty(10)

for i, alpha in enumerate(alphas) :
    
    lasso = Lasso(alpha=alpha)
    lasso.fit(XTrain, YTrain)
    testError[i] = mean_squared_error(YTest, lasso.predict(XTest))


# In[70]:


plt.plot(alphas, testError, 'r-')
plt.show()


# In[71]:


wmLinear = linear_model.LinearRegression(normalize=True)
wmLinear


# In[72]:


lasso = Lasso(alpha=17)
lasso.fit(XTrain, YTrain)

