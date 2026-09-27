#!/usr/bin/env python
# coding: utf-8

# # Churn Analysis

# In[1]:


#import the required libraries
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.ticker as mtick  
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('Customer-Churn.csv')


# In[3]:


type(df)


# In[4]:


df


# In[5]:


df.head()


# In[6]:


df.shape


# In[8]:


df.columns


# In[9]:


# Checking the data types of all the columns
df.dtypes


# In[10]:


# Check the descriptive statistics of numeric variables
df.describe()


# SeniorCitizen is actually a categorical hence the 25%-50%-75% distribution is not propoer
# 
# 75% customers have tenure less than 55 months
# 
# Average Monthly charges are USD 64.76 whereas 25% customers pay more than USD 89.85 per month

# In[11]:


df['Churn'].value_counts()


# In[12]:


df['Churn'].value_counts()/len(df)*100


# In[9]:


df['Churn'].value_counts().plot(kind='barh')


# In[10]:


df['Churn'].value_counts().plot(kind='bar')


# In[11]:


df['Churn'].value_counts().plot(kind='pie')


# In[21]:


df['Churn'].value_counts().plot(kind='barh', figsize=(4, 2))
plt.xlabel("Count")
plt.ylabel("Target Variable")
plt.title("Count of TARGET Variable per category");


# In[10]:


100*df['Churn'].value_counts()/len(df['Churn'])


# In[21]:


df['Churn'].value_counts()


# * Data is highly imbalanced, ratio = 73:27<br>
# * So we analyse the data with other features while taking the target values separately to get some insights.

# In[22]:


# Concise Summary of the dataframe, as we have too many columns, we are using the verbose = True mode
df.info() 


# In[23]:


df.isnull().sum()


# In[24]:


missing = pd.DataFrame((df.isnull().sum())*100/df.shape[0]).reset_index()
plt.figure(figsize=(16,5))
ax = sns.pointplot('index',0,data=missing)
plt.xticks(rotation =90,fontsize =7)
plt.title("Percentage of Missing values")
plt.ylabel("PERCENTAGE")
plt.show()


# ### Missing Data - Initial Intuition
# 
# * Here, we don't have any missing data.
# 
# General Thumb Rules:
# 
# * For features with less missing values- can use regression to predict the missing values or fill with the mean of the values present, depending on the feature.
# * For features with very high number of missing values- it is better to drop those columns as they give very less insight on analysis.
# * As there's no thumb rule on what criteria do we delete the columns with high number of missing values, but generally you can delete the columns, if you have more than 30-40% of missing values. But again there's a catch here, for example, Is_Car & Car_Type, People having no cars, will obviously have Car_Type as NaN (null), but that doesn't make this column useless, so decisions has to be taken wisely.

# ## Data Cleaning
# 

# **1.** Create a copy of base data for manupulation & processing

# In[28]:


new_df = df.copy()


# **2.** Total Charges should be numeric amount. Let's convert it to numerical data type

# In[29]:


new_df.TotalCharges = pd.to_numeric(new_df.TotalCharges, errors='coerce')


# In[30]:


new_df.info()


# In[31]:


new_df.isnull().sum()


# In[20]:


missing = pd.DataFrame((new_df.isnull().sum())*100/new_df.shape[0]).reset_index()
plt.figure(figsize=(16,5))
ax = sns.pointplot('index',0,data=missing)
plt.xticks(rotation =90,fontsize =7)
plt.title("Percentage of Missing values")
plt.ylabel("PERCENTAGE")
plt.show()


# **3.** As we can see there are 11 missing values in TotalCharges column. Let's check these records 

# In[32]:


new_df.loc[new_df ['TotalCharges'].isnull() == True]


# **4. Missing Value Treatement**

# Since the % of these records compared to total dataset is very low ie 0.15%, it is safe to ignore them from further processing.

# In[33]:


#Removing missing values 
new_df.dropna(how = 'any', inplace = True)
#new_df.fillna(0)


# In[34]:


new_df.head(5)


# In[35]:


new_df.shape


# **5.** Divide customers into bins based on tenure e.g. for tenure < 12 months: assign a tenure group if 1-12, for tenure between 1 to 2 Yrs, tenure group of 13-24; so on...

# In[39]:


# Get the max tenure
print(new_df['tenure'].max()) #72


# In[40]:


labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

print(labels)


# In[41]:


# Group the tenure in bins of 12 months
labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

new_df['tenure_group'] = pd.cut(new_df.tenure, range(1, 80, 12), right=False, labels=labels)


# In[42]:


new_df['tenure_group'].value_counts()


# **6.** Remove columns not required for processing

# In[43]:


#drop column customerID and tenure
new_df.drop(columns= ['customerID','tenure'], axis=1, inplace=True)


# In[44]:


new_df.head()


# ## Data Exploration
# **1. ** Plot distibution of individual predictors by churn

# ### Univariate Analysis

# In[45]:


new_df.head(5)


# In[46]:


new_df.Churn.value_counts()/len(new_df)*100


# In[47]:


sns.countplot(data=new_df, x='SeniorCitizen', hue='Churn')


# In[49]:


sns.countplot(data=new_df, x='gender', hue='Churn')


# In[34]:


for i, predictor in enumerate(new_df.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(data=new_df, x=predictor, hue='Churn')


# In[50]:


new_df.SeniorCitizen.value_counts()


# In[52]:


new_df1_target0=new_df[new_df["Churn"]=='No']
new_df1_target1=new_df[new_df["Churn"]=='Yes']


# In[33]:


new_df1_target1.SeniorCitizen.value_counts()


# In[53]:


new_df.gender.value_counts()


# In[54]:


new_df1_target1.gender.value_counts()


# In[34]:


pd.crosstab(new_df.PaymentMethod, new_df.Churn)


# **2.** Convert the target variable 'Churn'  in a binary numeric variable i.e. Yes=1 ; No = 0

# ## Univariate Insights [TO DO]
# 
# - Senior Citizens are more likely to churn
# - Monthly customers are more likely to churn

# In[35]:


new_df['Churn'] = np.where(new_df.Churn == 'Yes',1,0)


# In[36]:


new_df.head()


# **3.** Convert all the categorical variables into dummy variables

# In[37]:


# For Machine Learning (Predictive Modelling), we need to perform Feature Encoding
new_df_dummies = pd.get_dummies(new_df)
new_df_dummies.head()


# **9. ** Relationship between Monthly Charges and Total Charges

# In[38]:


sns.lmplot(data=new_df_dummies, x='MonthlyCharges', y='TotalCharges', fit_reg=False)


# In[40]:


new_df_dummies['MonthlyCharges'].corr(new_df_dummies['TotalCharges'])


# Total Charges increase as Monthly Charges increase - as expected.

# **10. ** Churn by Monthly Charges and Total Charges

# In[41]:


Mth = sns.kdeplot(new_df_dummies.MonthlyCharges[(new_df_dummies["Churn"] == 0) ],color="Red", shade = True)
Mth = sns.kdeplot(new_df_dummies.MonthlyCharges[(new_df_dummies["Churn"] == 1) ],ax =Mth, color="Blue", shade= True)
Mth.legend(["No Churn","Churn"],loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title('Monthly charges by churn')


# **Insight:** Churn is high when Monthly Charges ar high

# In[42]:


Tot = sns.kdeplot(new_df_dummies.TotalCharges[(new_df_dummies["Churn"] == 0) ],color="Red", shade = True)
Tot = sns.kdeplot(new_df_dummies.TotalCharges[(new_df_dummies["Churn"] == 1) ],ax =Tot, color="Blue", shade= True)
Tot.legend(["No Churn","Churn"],loc='upper right')
Tot.set_ylabel('Density')
Tot.set_xlabel('Total Charges')
Tot.set_title('Total charges by churn')


# **Surprising insight ** as higher Churn at lower Total Charges
# 
# However if we combine the insights of 3 parameters i.e. Tenure, Monthly Charges & Total Charges then the picture is bit clear :- Higher Monthly Charge at lower tenure results into lower Total Charge. Hence, all these 3 factors viz **Higher Monthly Charge**,  **Lower tenure** and **Lower Total Charge** are linkd to **High Churn**.

# **11. Build a corelation of all predictors with 'Churn' **

# In[39]:


plt.figure(figsize=(20,8))
new_df_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')


# **Derived Insight: **
# 
# **HIGH** Churn seen in case of  **Month to month contracts**, **No online security**, **No Tech support**, **First year of subscription** and **Fibre Optics Internet**
# 
# **LOW** Churn is seens in case of **Long term contracts**, **Subscriptions without internet service** and **The customers engaged for 5+ years**
# 
# Factors like **Gender**, **Availability of PhoneService** and **# of multiple lines** have alomost **NO** impact on Churn
# 
# This is also evident from the **Heatmap** below

# In[40]:


plt.figure(figsize=(12,12))
sns.heatmap(new_df_dummies.corr(), cmap="Paired")


# ### Bivariate Analysis

# In[41]:


new_df1_target0=new_df.loc[new_df["Churn"]==0] #Active Customers
new_df1_target1=new_df.loc[new_df["Churn"]==1] #Churned Customers


# In[42]:


len(new_df1_target0)


# In[43]:


len(new_df1_target1)


# In[47]:


def uniplot(df,col,title,hue =None):
    
    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.titlepad'] = 30
    
    
    temp = pd.Series(data = hue)
    fig, ax = plt.subplots()
    width = len(df[col].unique()) + 7 + 4*len(temp.unique())
    fig.set_size_inches(width , 8)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title(title)
    ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue,palette='bright') 
        
    plt.show()


# In[49]:


uniplot(new_df1_target1,col='Partner',title='Distribution of Gender for Churned Customers',hue='gender')


# In[93]:


uniplot(new_df1_target0,col='Partner',title='Distribution of Gender for Non Churned Customers',hue='gender')


# In[48]:


uniplot(new_df1_target1,col='PaymentMethod',title='Distribution of PaymentMethod for Churned Customers',hue='gender')


# In[36]:


uniplot(new_df1_target1,col='Contract',title='Distribution of Contract for Churned Customers',hue='gender')


# In[37]:


uniplot(new_df1_target1,col='TechSupport',title='Distribution of TechSupport for Churned Customers',hue='gender')


# In[38]:


uniplot(new_df1_target1,col='SeniorCitizen',title='Distribution of SeniorCitizen for Churned Customers',hue='gender')


# # CONCLUSION

# These are some of the quick insights from this exercise:
# 
# 1. Electronic check medium are the highest churners
# 2. Contract Type - Monthly customers are more likely to churn because of no contract terms, as they are free to go customers.
# 3. No Online security, No Tech Support category are high churners
# 4. Non senior Citizens are high churners
# 
# Note: There could be many more such insights, so take this as an assignment and try to get more insights :)

# 1. Senior Citizens are more likely to churn, they have a churn rate of ~42%
# 2. People with no Partners are more likely to churn, they have a churn rate of ~35%
# 3. Monthly customers are more likely to churn, they have a churn rate of ~41%, people with 2 year contract are very less likely to churn
# 4. Almost 44% of the customers paying via Electronic check are churners
# 5. People with less tenure i.e 1-12 are very high churners
# 6. Higher Monthly Charge, Lower tenure and Lower Total Charge are linked to High Churn
# 7. Females who don't have partners are more likely to churn compared to their male counterparts
# 8. Females using credit card are very high churners as compared to their counterparts

# In[55]:


telco_data_dummies.to_csv('tel_churn.csv')

