#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


df=pd.read_csv(r"D:\ML projects/hotel_bookings.csv")


# In[17]:


df.head(2)


# # Performing data cleaning

# In[18]:


df.isnull().sum()    # no. of missing values in every features


# In[19]:


df.drop(['agent','company'],axis=1,inplace=True)         # dropping two columns 'agent' and 'company'


# In[20]:


df['country'].value_counts()    # count of each sub-categories of 'country' features


# In[21]:


df.fillna(0,inplace=True)   # all the null values filling with 0


# In[22]:


df.isnull().sum()


# In[23]:


filter1=(df['children']==0)& (df['adults']==0) & (df['babies']==0)


# In[24]:


df[filter1]


# In[25]:


data=df[~filter1]    # removing all the rows where no. of children, adults and babies is 0


# In[26]:


data.shape   # 180 rows has been removed


# # Performing EDA

# In[27]:


# from where the most of the guests come from?


# In[28]:


data[data['is_canceled']==0]     # Taking the rows where the booking do not get canceled


# In[29]:


data[data['is_canceled']==0]['country'].value_counts()  # we can say 20977 people booked from Portugal


# In[30]:


len(data[data['is_canceled']==0])


# In[31]:


data[data['is_canceled']==0]['country'].value_counts()/75011


# In[32]:


# we can say 27% customer booked from Portugal(PRT)


# In[33]:


country_wise_data=data[data['is_canceled']==0]['country'].value_counts().reset_index()
country_wise_data.columns=['country','no_of_guests']
country_wise_data                           # converting into dataframe


# In[34]:


import plotly
import chart_studio.plotly as py
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)


# In[35]:


import plotly.express as px


# In[36]:


map_guest=px.choropleth(country_wise_data,
             locations=country_wise_data['country'],
             color=country_wise_data['no_of_guests'],
              hover_name=country_wise_data['country'],
              title='home country of guests')
             


# In[37]:


map_guest.show()


# In[38]:


#How much do guests pay for a room per night ?


# In[39]:


data2=data[data['is_canceled']==0]


# In[40]:


data2.columns


# In[41]:


plt.figure(figsize=(12,8))
sns.boxplot(x='reserved_room_type',y='adr' ,hue='hotel',data=data2)

plt.title('Price of room types per night and person')
plt.xlabel('room types')
plt.ylabel('price( EUR)')
plt.show()


# In[42]:


# for city hotel,guests pay extremely high price for room category G and for Resort hotel,guests pay extremely high
# price for room category H


# In[43]:


#Which are the most busy month ?


# In[44]:


data['hotel'].unique()


# In[45]:


data_resort=data[(data['hotel']=='Resort Hotel') & (data['is_canceled']==0)]
data_city = data[(data['hotel']=='City Hotel') & (data['is_canceled']==0)]


# In[46]:


data_resort.head(3)


# In[47]:


rush_resort=data_resort['arrival_date_month'].value_counts().reset_index()
rush_resort.columns=['month','no_of_guests']
rush_resort                     #  in which month how many guests came to the hotel?


# In[48]:


rush_city=data_city['arrival_date_month'].value_counts().reset_index()
rush_city.columns=['month','no_of_guests']
rush_city


# In[49]:


final_rush=rush_resort.merge(rush_city,on='month')   # merging two features 


# In[50]:


final_rush.columns=['month','no_of_guests_in_resort','no_of_guests_city']  # giving column names


# In[51]:


final_rush


# In[52]:


import sort_dataframeby_monthorweek as sd


# In[53]:


final_rush=sd.Sort_Dataframeby_Month(final_rush,'month')    # to get all the months in sorted order


# In[54]:


final_rush.columns


# In[55]:


px.line(data_frame=final_rush,x='month',y=['no_of_guests_in_resort', 'no_of_guests_city'])


# In[56]:


# In August, most of the guests booked hotel i.e. most intense month is August for rush.Also July, August,September are
# rush months


# In[57]:


#which month has highest adr(average daily rate)?


# In[58]:


data=sd.Sort_Dataframeby_Month(data,'arrival_date_month')


# In[59]:


sns.barplot(x='arrival_date_month',y='adr',data=data ,hue='is_canceled')
plt.xticks(rotation='vertical')
plt.show()


# In[60]:


# "Cancel booking" have the higher average daily rate than the "Not canceled booking"


# In[61]:


# month August has extremely high adr compare to the other months for City hotel and resort hotel both


# In[62]:


plt.figure(figsize=(12,8))
sns.boxplot(x='arrival_date_month',y='adr',data=data ,hue='is_canceled')
plt.xticks(rotation='vertical')
plt.ylim(0,800)
plt.show()


# In[63]:


# from boxplot we can say that higher adr could be one of the reason of cancel booking.


# In[64]:


#Lets analyse whether bookings were made only for weekdays or for weekends or for both ??


# In[65]:


data.columns


# In[66]:


pd.crosstab(index=data['stays_in_weekend_nights'],columns=data['stays_in_week_nights'])


# In[67]:


# As example we can say from the above table, approximately 9k guests stayed in hotel for 2 week night and 1 weekend night


# In[68]:


def week_function(row):
    feature1='stays_in_weekend_nights'
    feature2='stays_in_week_nights'
    
    if row[feature2]==0 and row[feature1] >0 :
        return 'stay_just_weekend'
    
    elif row[feature2]>0 and row[feature1] ==0 :
        return 'stay_just_weekdays'
    
    elif row[feature2]>0 and row[feature1] >0 :
        return 'stay_both_weekdays_weekends'
    
    else:
        return 'undefined_data'


# In[69]:


data2['weekend_or_weekday']=data2.apply(week_function,axis=1)


# In[70]:


data2.head(2)


# In[71]:


data2['weekend_or_weekday'].value_counts()


# In[72]:


# So 37551 guests stayed in Hotel for weekdays and weekends both.


# In[73]:


type(sd)


# In[74]:


data2=sd.Sort_Dataframeby_Month(data2,'arrival_date_month')


# In[75]:


data2.groupby(['arrival_date_month','weekend_or_weekday']).size()


# In[76]:


group_data=data2.groupby(['arrival_date_month','weekend_or_weekday']).size().unstack().reset_index()


# In[77]:


sorted_data=sd.Sort_Dataframeby_Month(group_data,'arrival_date_month') # To sort "Month feature"


# In[78]:


sorted_data.set_index('arrival_date_month',inplace=True)  # changing the index to the column 'arrival_date_month'


# In[79]:


sorted_data


# In[80]:


# So as a example, In January 1550 guests stayed in Hotel for weekdays as well as wekends.


# In[81]:


sorted_data.plot(kind='bar',stacked=True,figsize=(15,10))       # visualizing above data


# In[82]:


#How to create some more features ?


# In[83]:


# This function takes rows as a input and gives output 1 for which families has atleast one member
def family(row):
    if (row['adults']>0) &  (row['children']>0 or row['babies']>0) :
        return 1
    else:
        return 0


# In[84]:


data['is_family']=data.apply(family,axis=1)  # applying the above function with dataframe "data"


# In[85]:


data['total_customer'] = data['adults'] + data['babies'] + data['children']


# In[86]:


data['total_nights']=data['stays_in_week_nights'] + data['stays_in_weekend_nights']


# In[87]:


data.head(3)


# In[88]:


data.columns


# In[89]:


data['deposit_type'].unique()


# In[90]:


dict1={'No Deposit':0, 'Non Refund':1, 'Refundable': 0}


# In[91]:


data['deposit_given']=data['deposit_type'].map(dict1)


# In[92]:


data.columns


# In[93]:


data.drop(columns=['adults', 'children', 'babies', 'deposit_type'],axis=1,inplace=True)  # dropping features


# In[94]:


data.columns


# In[95]:


#how to apply Feature encoding on data?


# In[96]:


data.head(6)


# In[97]:


data.dtypes


# In[98]:


cate_features=[col for col in data.columns if data[col].dtype=='object']  # Categoricalfeatures 


# In[99]:


num_features=[col for col in data.columns if data[col].dtype!='object']   # numerical features


# In[100]:


num_features


# In[101]:


cate_features


# In[102]:


data_cat=data[cate_features]


# In[103]:


data.groupby(['hotel'])['is_canceled'].mean().to_dict()


# In[104]:


# Average number of guests canceled booking for City Hotel is  0.4178593534858457 and for Resort Hotel is 0.277673733363298


# In[105]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[106]:


data_cat['cancellation']=data['is_canceled']


# In[107]:


data_cat.head()


# In[108]:


cols=data_cat.columns


# In[109]:


cols


# In[110]:


cols=cols[0:-1]   # excluding last feature


# In[111]:


cols


# In[112]:


# mean encoding for every categorical features
for col in cols:
    dict2=data_cat.groupby([col])['cancellation'].mean().to_dict()
    data_cat[col]=data_cat[col].map(dict2)


# In[113]:


data_cat.head(3)   #  we converted all categorical features to numerical features by mean encoding


# In[114]:


#Handle Outliers


# In[115]:


data[num_features]


# In[116]:


dataframe=pd.concat([data_cat,data[num_features]],axis=1)  # concatening data_cat and num_features


# In[117]:


dataframe.columns


# In[118]:


dataframe.drop(['cancellation'],axis=1,inplace=True)


# In[119]:


dataframe.head(3)


# In[120]:


sns.distplot(dataframe['lead_time'])


# In[121]:


# applying log of those column whose distributuion has skewness (i.e. which columns has some outlier values)
def handle_outlier(col):
    dataframe[col]=np.log1p(dataframe[col])


# In[122]:


handle_outlier('lead_time')


# In[123]:


sns.distplot(dataframe['lead_time'])


# In[124]:


# After applying log we can see less number of outliers in the distribution


# In[125]:


# adr(Average Daily Rate)


# In[126]:


sns.distplot(dataframe['adr'])


# In[127]:


dataframe[dataframe['adr']<0]


# In[128]:


handle_outlier('adr')


# In[129]:


dataframe['adr'].isnull().sum()  # negative value has not been handled by log function.


# In[130]:


sns.distplot(dataframe['adr'].dropna())  # plotting distribution after removing the row with negative "adr" value


# In[131]:


#Select important Features using Co-relation & univariate analysis.


# In[132]:


sns.FacetGrid(data,hue='is_canceled',xlim=(0,500)).map(sns.kdeplot,'lead_time',shade=True).add_legend()


# In[133]:


# For feature "lead_time" ,distribution of is_canceled=0 and is_canceled=1 are not overlapped.


# In[134]:


corr=dataframe.corr()                # correlation between every features


# In[135]:


corr               


# In[136]:


corr['is_canceled'].sort_values(ascending=False)


# In[137]:


corr['is_canceled'].sort_values(ascending=False).index


# In[138]:


# High correlation  causes  overfitting model and Low correlation causes low accuracy for the ML model
# So it is good to drop thee features.


# In[139]:


features_to_drop=['reservation_status', 'reservation_status_date','arrival_date_year',
       'arrival_date_week_number', 'stays_in_weekend_nights',
       'arrival_date_day_of_month']


# In[140]:


dataframe.drop(features_to_drop,axis=1,inplace=True)    # dropping features


# In[141]:


dataframe.shape


# In[142]:


#How to find Important features for model building?


# In[143]:


dataframe.head(2)


# In[144]:


dataframe.isnull().sum()


# In[145]:


dataframe.dropna(inplace=True)


# In[146]:


## separate dependent & independent features


# In[147]:


x=dataframe.drop('is_canceled',axis=1)


# In[148]:


y=dataframe['is_canceled']


# In[149]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


# In[150]:


feature_sel_model=SelectFromModel(Lasso(alpha=0.005)) # creating model


# In[151]:


feature_sel_model.fit(x,y)     # fitting model


# In[152]:


feature_sel_model.get_support()      # Showing "True" means that features have  selected for the model


# In[153]:


cols=x.columns


# In[154]:


cols


# In[155]:


# let's print the number of selected features

selected_feature=cols[feature_sel_model.get_support()]


# In[156]:


selected_feature               


# In[157]:


x=x[selected_feature]


# In[158]:


x


# In[159]:


#Lets build ML model.


# In[160]:


from sklearn.model_selection import train_test_split


# In[161]:


X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.25)


# In[162]:


X_train.shape


# In[163]:


from sklearn.linear_model import LogisticRegression


# In[164]:


logreg=LogisticRegression()


# In[165]:


logreg.fit(X_train,y_train)


# In[166]:


pred=logreg.predict(X_test)


# In[167]:


pred


# In[168]:


from sklearn.metrics import confusion_matrix


# In[169]:


confusion_matrix(y_test,pred)


# In[170]:


from sklearn.metrics import accuracy_score


# In[171]:


accuracy_score(y_test,pred)


# In[172]:


# How to cross-validate model?


# In[173]:


from sklearn.model_selection import cross_val_score


# In[174]:


score=cross_val_score(logreg,x,y,cv=10) # using logistic regression


# In[175]:


score           # accuracy of 10 iterations


# In[176]:


score.mean()    # mean of accuracy of 10 iterations


# In[177]:


#playing with multiple algorithms


# In[178]:


from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[179]:


models=[]

models.append(('LogisticRegression',LogisticRegression()))
models.append(('Naive_bayes',GaussianNB()))
models.append(('Random Forest',RandomForestClassifier()))
models.append(('Decision_tree',DecisionTreeClassifier()))
models.append(('KNN',KNeighborsClassifier()))


# In[180]:


for name,model in models:
    print(name)
    model.fit(X_train,y_train)
    
    predictions=model.predict(X_test)
    
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(predictions,y_test)
    print(cm)
    
    from sklearn.metrics import accuracy_score
    acc=accuracy_score(predictions,y_test)
    print(acc)
    print('\n')


# In[ ]:




