#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


# In[3]:


car_price=pd.read_csv('https://raw.githubusercontent.com/akjadon/Finalprojects_DS/master/Car_pricing_prediction/CarPrice_Assignment.csv')


# In[4]:


car_price.columns


# In[5]:


car_price.info()


# In[6]:


car_price.describe()


# In[7]:


car_price=car_price.join(car_price['CarName'].str.split(' ',1,expand=True).rename(columns={0:'company',1:'carmodel'}))


# In[8]:


car_price


# In[9]:


car_price.loc[car_price.duplicated()]


# In[10]:


plt.figure(figsize=(30,20))

plt.subplot(1,2,1)
plt1 = car_price['company'].value_counts().plot('bar')
plt.title('Companies Histogram')
plt1.set(xlabel = 'Car Company', ylabel='Frequency of Company')



plt.subplot(1,2,2)
company_vs_price = pd.DataFrame(car_price.groupby(['company'])['price'].mean().sort_values(ascending = False))
plt2=company_vs_price.index.value_counts().plot('bar')
plt.title('Company Name vs Average Price')
plt2.set(xlabel='Car Company', ylabel='Average Price')
xs=company_vs_price.index
ys=company_vs_price['price'].round(2)
plt.bar(xs,ys)


# In[11]:



plt.figure(figsize=(25, 6))


plt.subplot(1,2,1)
plt.title('Fuel Type Chart')
labels=car_price['fueltype'].unique()
plt3 = car_price['fueltype'].value_counts().tolist()
plt.pie(plt3,labels=plt3, autopct='%1.1f%%')
plt.legend(labels)


plt.subplot(1,2,2)
fuel_vs_price = pd.DataFrame(car_price.groupby(['fueltype'])['price'].mean().sort_values(ascending = False))
plt4=fuel_vs_price.index.value_counts().plot('bar')
plt.title('Fuel Type vs Average Price')
plt4.set(xlabel='Fuel Type', ylabel='Average Price')
xs=fuel_vs_price.index
ys=fuel_vs_price['price'].round(2)
plt.bar(xs,ys)


# In[12]:




plt.figure(figsize=(15,10))


plt.subplot(1,2,1)
plt.title('Car Body Type Chart')
labels=car_price['carbody'].unique()
plt5 = car_price['carbody'].value_counts().tolist()
plt.pie(plt5, labels=plt5, autopct='%1.1f%%')
plt.legend(labels, loc=1)


plt.subplot(1,2,2)
car_vs_price = pd.DataFrame(car_price.groupby(['carbody'])['price'].mean().sort_values(ascending = False))
plt6=car_vs_price.index.value_counts().plot('bar')
plt.title('Car Body Type vs Average Price')
plt6.set(xlabel='Car Body Type', ylabel='Average Price')
xs=car_vs_price.index
ys=car_vs_price['price'].round(2)
plt.bar(xs,ys)


# In[13]:



plt.figure(figsize=(25,10))

plt.subplot(1,2,1)
plt.title('symboling chart')
labels=car_price['symboling'].unique()
plt7=car_price['symboling'].value_counts().tolist()
plt.pie(plt7,labels=plt7,autopct='%1.1f%%')
plt.legend(labels)

plt.subplot(1,2,2)
sns.boxplot(x=car_price['symboling'],y=car_price['price'])
plt.show()


# In[14]:



plt.figure(figsize=(25,10))

plt.subplot(1,2,1)
plt8=car_price['enginetype'].value_counts().plot('bar')
plt.title('engine types')
plt8.set(xlabel='engine',ylabel='freq')
xs=car_price['enginetype'].unique()
ys=car_price['enginetype'].value_counts()
plt.bar(xs,ys)


plt.subplot(1,2,2)
sns.boxplot(x=car_price['enginetype'],y=car_price['price'])
plt.title('engine type vs price')


# In[15]:



plt.figure(figsize=(25,10))

plt.subplot(1,2,1)
plt8=car_price['doornumber'].value_counts().tolist()
labels=car_price['doornumber'].unique()
plt.title('door number graph')
plt.pie(plt8,labels=plt8,autopct='%1.1f%%')
plt.legend('labels',loc=1)

plt.subplot(1,2,2)
sns.boxplot(x=car_price['doornumber'],y=car_price['price'])
plt.title('door and price relation')
plt.show()


# In[16]:



plt.figure(figsize=(25,10))

plt.subplot(121)
plt9=car_price['enginelocation'].value_counts().tolist()
labels=car_price['enginelocation'].unique()
plt.pie(plt9,labels=plt9,autopct='%1.1f%%')
plt.title('engine locations')
plt.legend(labels,loc=1)

plt.subplot(122)
sns.boxplot(x=car_price['enginelocation'],y=car_price['price'])
plt.title('relationship')
plt.show()


# In[17]:



plt.figure(figsize=(25,10))

plt.subplot(1,2,1)
plt10=car_price['fuelsystem'].value_counts().plot('bar')
plt10.set(xlabel='fuelsystem',ylabel='freq')
plt.title('fuel system')
ys=car_price['fuelsystem'].value_counts()
xs=car_price['fuelsystem'].unique()
plt.bar(xs,ys)


plt.subplot(1,2,2)
plt.title('Fuel System Type vs Price')
sns.boxplot(x=car_price['fuelsystem'], y=car_price['price'])
plt.show()


# In[18]:



plt.figure(figsize=(25,10))

plt.subplot(121)
plt11=car_price['cylindernumber'].value_counts().plot('bar')
plt.title('cylinder')
plt11.set(xlabel='cylinder number',ylabel='freq')
xs=car_price['cylindernumber'].unique()
ys=car_price['cylindernumber'].value_counts()
plt.bar(xs,ys)
 
plt.subplot(1,2,2)
plt.title('Cylinder Number vs Price')
sns.boxplot(x=car_price['cylindernumber'], y=car_price['price'])
plt.show()


# In[19]:



plt.figure(figsize=(25,10))

plt.subplot(121)
plt12=car_price['aspiration'].value_counts().tolist()
labels=car_price['aspiration'].unique()
plt.title('aspiration')
plt.pie(plt12,labels=plt12,autopct='%1.1f%%')
plt.legend(labels)

plt.subplot(1,2,2)
plt.title('Engine Location vs Price')
sns.boxplot(x=car_price['aspiration'], y=car_price['price'])
plt.show()


# In[20]:



plt.figure(figsize=(15,5))

plt.subplot(121)
plt.subplot(1,2,1)
labels=car_price['drivewheel'].unique()
plt13 = car_price['drivewheel'].value_counts().tolist()
plt.title('Drive Wheel Chart')
plt.pie(plt13, labels=plt13, autopct='%1.1f%%')
plt.legend(labels)


plt.subplot(1,2,2)
plt.title('Drive Wheel vs Price')
sns.boxplot(x=car_price['drivewheel'], y=car_price['price'])
plt.show()


# In[21]:


def scatterplot(df,var):
    
    plt.scatter(df[var],df['price'])
    plt.xlabel(var);plt.ylabel('price')
    plt.title('scatter plot for'+var+ 'vs price')


# In[22]:



plt.figure(figsize=(15,20))
plt.subplot(2,2,1)
scatterplot(car_price,'carlength')    
plt.subplot(2,2,2)
scatterplot(car_price,'carwidth')
plt.subplot(2,2,3)
scatterplot(car_price,'carheight')
plt.show()
plt.tight_layout()


# In[23]:


plt.figure(figsize=(15,20))
plt.subplot(2,2,1)
scatterplot(car_price,'curbweight')    
plt.subplot(2,2,2)
scatterplot(car_price,'horsepower')
plt.subplot(2,2,3)
scatterplot(car_price,'boreratio')
plt.subplot(2,2,4)
scatterplot(car_price,'compressionratio')
plt.show()
plt.tight_layout()


# In[24]:


sns.pairplot(car_price,x_vars=['highwaympg','citympg'],y_vars='price',height=4,aspect=1,kind='scatter')


# In[25]:


sns.pairplot(car_price,x_vars=['boreratio','compressionratio'],y_vars='price',height=4,aspect=1,kind='scatter')


# In[26]:


plt.figure(figsize=(15,20))
plt.subplot(2,2,1)
scatterplot(car_price,'enginesize')    
plt.subplot(2,2,2)
scatterplot(car_price,'stroke')
plt.subplot(2,2,3)
scatterplot(car_price,'peakrpm')
plt.subplot(2,2,4)
scatterplot(car_price,'wheelbase')
plt.show()
plt.tight_layout()


# In[27]:


cor=car_price.corr().round(3).loc['price']
cor=pd.DataFrame(cor)
result=[]
for i in cor['price']:
    if(i>-1 and i<-0.4):result.append('strongly negative')
    elif(i>-0.4 and i<-0.2):result.append('moderately negative')
    elif(i>-0.2 and i<0):result.append('week negative')
    elif(i>0 and i<0.2):result.append('week positive')
    elif(i>0.2 and i<0.5):result.append('moderate positive')
    else:result.append('strongly positive')  
cor['correlation']=result     
cor['correlation'].value_counts()


# In[28]:


varr=cor[(cor.correlation == 'strongly positive') | (cor.correlation == 'strongly negative')]
varr


# In[29]:


cars=car_price[['wheelbase','carlength','carwidth','curbweight','enginesize','boreratio','horsepower','citympg','highwaympg','enginetype','fueltype','carbody','aspiration','cylindernumber','drivewheel']]
cars.columns


# In[30]:


sns.pairplot(cars)
plt.show()


# In[31]:


nums={"fueltype":{"gas":1,"diesel":2},"aspiration":{"std":1,"turbo":2},"doornumber":{"four":4,"two":2},"carbody":{"sedan":1,"hatchback":2,"wagon":3,"hardtop":4,"convertible":5},"drivewheel":{"fwd":1,"rwd":2,"4wd":3},"enginelocation":{"front":1,"rear":2},"enginetype":{"ohc":1,"ohcf":2,"ohcv":3,"dohc":4,"l":5,"rotor":5,"dohcv":6},"cylindernumber":{"four":4,"six":6,"five":5,"eight":8,"two":2,"three":3,"twelve":12},"fuelsystem":{"mpfi":1,"2bbl":2,"idi":3,"1bbl":4,"spdi":5,"4bbl":6,"spfi":7,"mfi":8}}


# In[32]:


car_price.replace(nums, inplace=True)


# In[33]:


car_price.drop(["CarName"],axis=1,inplace=True)


# In[34]:


car_price.head(2)


# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


x=car_price[['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize',
       'boreratio', 'horsepower', 'citympg', 'highwaympg', 'enginetype',
       'fueltype', 'carbody', 'aspiration', 'cylindernumber', 'drivewheel']]
y=car_price['price']


# In[37]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[38]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()


# In[39]:


regressor.fit(x_train,y_train)


# In[40]:


y_pred=pd.DataFrame(regressor.predict(x_test),columns=['y_test'])


# In[41]:


pd.DataFrame(y_test)


# In[42]:


predicted_price=pd.DataFrame(regressor.predict(x))
car_price['predicted_price']=predicted_price
car_price.head()


# In[43]:


comp=pd.DataFrame(car_price[['price','predicted_price']])


# In[44]:


regressor.score(x_test,y_test).round(2)


# In[45]:


plt.figure(figsize=(25,15))
comp.plot()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




