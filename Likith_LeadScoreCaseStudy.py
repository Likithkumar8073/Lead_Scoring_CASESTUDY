#!/usr/bin/env python
# coding: utf-8

# ### Goal of the case study

# * The goal of the case study is to build a logistic regression model to predict lead score between 0 to 100. High lead score means conversion or 'hot' and low means no conversion or 'cold'

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, r2_score, plot_roc_curve
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ### Read the leads data in .csv format as pandas dataframe

# In[2]:


leaddf = pd.read_csv(r"C:\Users\riduv\Downloads\Leads.csv")


# ### Visualize rows and columns

# In[3]:


leaddf.head(3)


# ### Check initial shape

# In[4]:


leaddf.shape


# ### Data cleaning 

# ### check empty column names

# In[5]:


leaddf.columns.isnull().sum()


# **Impression:**
# no empty columns names were identified

# ### Check duplicated rows and columns

# In[6]:


leaddf[leaddf.duplicated()].shape


# In[7]:


leaddf.loc[:,leaddf.columns.duplicated()].shape


# **Impression:**
# No duplicated rows were identified

# ### Check datatype and properties of columns

# In[8]:


leaddf.info()


# In[9]:


leaddf.describe()


# ### Check and fill/remove empty values rows/columns 
# 
# Calculate percentage of null values in each column of the two dataframe and remove those columns with greater than 40 percentage of null values
# 

# In[10]:


## replacing the word 'select' with nan values. so, null colums can be further removed

leaddf = leaddf.replace('Select', np.nan)


# In[11]:


find_col_leaddf = (leaddf.isnull().sum()/leaddf.shape[0])*100
nan_cols_leaddf = find_col_leaddf[(find_col_leaddf != 0) & (find_col_leaddf >= 45)].index

print("Current application df columns with greater than 45% of null values: ", nan_cols_leaddf.shape[0])


# #### Columns with percentage null values

# In[12]:


find_col_leaddf


# #### Columns with greater then 40% null values

# In[13]:


nan_cols_leaddf


# ### Drop columns

# In[14]:


leaddf.drop(['Prospect ID', 'Lead Number'], axis=1, inplace=True)


# In[15]:


leaddf.drop(nan_cols_leaddf, axis=1, inplace=True)


# **Impression**: removed 'Prospect ID', 'Lead Number' as these are identification numbers with no information for model. 'How did you hear about X Education', 'Lead Profile' 'Lead Quality', 'Asymmetrique Activity Index', 'Asymmetrique Profile Index', 'Asymmetrique Activity Score',  'Asymmetrique Profile Score' columns where dropped with greater than 45% null values.

# In[16]:


## set of columns after removal with percentage of null values

(leaddf.isnull().sum()/leaddf.shape[0])*100


# ### Check final shape of dataframe

# In[17]:


leaddf.shape


# ### Numerical and Categorical columns

# #### Categorical columns

# In[18]:


leaddf[leaddf.columns.difference(leaddf.describe().columns)].columns


# #### Numerical columns

# In[19]:


leaddf.describe().columns


# ### Handling outliers in numerical columns

# Ploted are some of the columns with outliers

# In[20]:


plt.figure(figsize=(10,2))
sns.boxplot(leaddf["TotalVisits"])
plt.show()


# In[21]:


## total visits outlier removal

Q3 = leaddf.TotalVisits.quantile(0.99)
leaddf = leaddf[(leaddf.TotalVisits <= Q3)]
Q1 = leaddf.TotalVisits.quantile(0.01)
leaddf = leaddf[(leaddf.TotalVisits >= Q1)]
sns.boxplot(y=leaddf['TotalVisits'])
plt.show()


# In[22]:


plt.figure(figsize=(10,2))
sns.boxplot(leaddf["Total Time Spent on Website"])
plt.show()


# In[23]:


plt.figure(figsize=(10,2))
sns.boxplot(leaddf["Page Views Per Visit"])
plt.show()


# In[24]:


## Page Views Per Visit column outlier removal

Q3 = leaddf["Page Views Per Visit"].quantile(0.99)
leaddf = leaddf[(leaddf["Page Views Per Visit"] <= Q3)]
Q1 = leaddf["Page Views Per Visit"].quantile(0.01)
leaddf = leaddf[(leaddf["Page Views Per Visit"] >= Q1)]
sns.boxplot(y=leaddf["Page Views Per Visit"])
plt.show()


# ### Check correlations in numerical columns

# In[25]:


sns.heatmap(leaddf.corr(), annot=True)
plt.show()


# ### Checking data imbalance

# In[26]:


leaddf.Converted.value_counts()


# In[27]:


leaddf.Converted.value_counts().plot.bar()
plt.xlabel("Converted")
plt.ylabel("COUNTS")
plt.title("Data Imbalance")
plt.show()


# **Impression** 
# * There is a slight data imbalance with 3346 rows of converted (1) and 5487 rows of non converted (0).

# ### Analysis of categorical features

# In[28]:


leaddf[leaddf.columns.difference(leaddf.describe().columns)].columns


# In[29]:


def Converted_or_not_percent(df, col):
    tot_dict = {0:{},1:{}}
    for i in df["Converted"].unique():
        nonan = df[col].unique()
        for j in nonan[~pd.isnull(nonan)]:
            perc = (df[(df["Converted"]==i) & (df[col]==j)].shape[0]/df[df[col]==j].shape[0])*100
            tot_dict[i][j] = perc
    tot_dict["Not_converted"],tot_dict["Converted"] = tot_dict.pop(0),tot_dict.pop(1)
    return tot_dict


# #### Column 'Country'

# In[30]:


leaddf['Country'].value_counts(dropna=False)


# In[31]:


## As observed majority of rows have india as city, so we can replace with 'India'

leaddf['Country'] = leaddf['Country'].replace(np.nan,'India')


# In[32]:


leaddf['Country'].value_counts(dropna=False)/leaddf['Country'].shape[0]


# In[33]:


### we can drop the column as around 97% of the rows has india as country

leaddf.drop(['Country'], axis=1, inplace=True)


# #### Column 'City'

# In[34]:


leaddf['City'].value_counts(dropna=False)


# In[35]:


## As observed majority of rows have mumbai, so we can replace with 'mumbai'

leaddf['City'] = leaddf['City'].replace(np.nan,'Mumbai')


# In[36]:


sns.countplot(x=leaddf["City"],hue=leaddf["Converted"])
plt.xticks(rotation=90)
plt.show()


# #### Column 'What is your current occupation'

# In[37]:


leaddf['What is your current occupation'].value_counts(dropna=False)


# In[38]:


leaddf['What is your current occupation'].value_counts(dropna=False)/leaddf['What is your current occupation'].shape[0]


# In[39]:


## As observed majority (60 %) of rows have Unemployed as occupation, so we can replace with 'Unemployed'

leaddf['What is your current occupation'] = leaddf['What is your current occupation'].replace(np.nan,'Unemployed')


# In[40]:


Converted_or_not_percent(leaddf, 'What is your current occupation')


# In[41]:


sns.countplot(x=leaddf['What is your current occupation'],hue=leaddf['Converted'])
plt.xticks(rotation=90)
plt.show()


# **Impression** 
# * Working professionals have higher chances of getting converted or hot lead. while, Unemployed have higher chances of not getting converted or cold

# 

# #### Column 'What is your current occupation'

# In[42]:


leaddf['Specialization'].value_counts(dropna=False)/leaddf['Specialization'].shape[0]


# In[43]:


leaddf['Specialization'] = leaddf['Specialization'].replace(np.nan, 'Not Specified')


# In[44]:


sns.countplot(x=leaddf['Specialization'],hue=leaddf['Converted'])
plt.xticks(rotation=90)
plt.show()


# In[45]:


# As there many specialization related to management, can combine them into category 'Management_Studies'

leaddf['Specialization'] = leaddf['Specialization'].replace(['Finance Management','Human Resource Management',
                                                           'Marketing Management','Operations Management',
                                                           'IT Projects Management','Supply Chain Management',
                                                    'Healthcare Management','Hospitality Management',
                                                           'Retail Management'] ,'Management_Studies')  


# In[46]:


sns.countplot(x=leaddf['Specialization'],hue=leaddf['Converted'])
plt.xticks(rotation=90)
plt.show()


# **Impression**
# * It can be observed that majority of 'hot' leads have specialization related to management

# #### Column 'Tags'

# In[47]:


leaddf['Tags'].value_counts(dropna=False)


# In[48]:


leaddf['Tags'] = leaddf['Tags'].replace(np.nan, 'Not Specified')


# In[49]:


sns.countplot(x=leaddf['Tags'],hue=leaddf['Converted'])
plt.xticks(rotation=90)
plt.show()


# In[50]:


## less than 50 occurences are grouped into one category

d = leaddf['Tags'].value_counts(dropna=False)
to_be_droped_cols = d[d < 50].index

leaddf['Tags'] = leaddf['Tags'].replace(to_be_droped_cols, 'Other_Tags')


# In[51]:


sns.countplot(x=leaddf['Tags'],hue=leaddf['Converted'])
plt.xticks(rotation=90)
plt.show()


# #### Column 'What matters most to you in choosing a course'

# In[52]:


leaddf['What matters most to you in choosing a course'].value_counts(dropna=False)/leaddf['What matters most to you in choosing a course'].shape[0]


# In[53]:


# As this column provides no new information, we can drop this column

leaddf.drop(['What matters most to you in choosing a course'], axis=1, inplace=True)


# #### Column ''Lead Origin''

# In[54]:


leaddf['Lead Origin'].value_counts(dropna=False)/leaddf['Lead Origin'].shape[0]


# In[55]:


sns.countplot(x=leaddf['Lead Origin'],hue=leaddf['Converted'])
plt.xticks(rotation=90)
plt.show()


# In[56]:


Converted_or_not_percent(leaddf, 'Lead Origin')


# **Impression**
# * Lead add form showed a higher convertion, while, API showed lower convertion or hot leads

# #### Column ''Lead Source''

# In[57]:


leaddf['Lead Source'].value_counts(dropna=False)


# In[58]:


leaddf['Lead Source'] = leaddf['Lead Source'].replace(np.nan,'Others')
leaddf['Lead Source'] = leaddf['Lead Source'].replace('google','Google')
leaddf['Lead Source'] = leaddf['Lead Source'].replace('Facebook','Social Media')
leaddf['Lead Source'] = leaddf['Lead Source'].replace(['bing','Click2call','Press_Release',
                                                     'youtubechannel','welearnblog_Home',
                                                     'WeLearn','blog','Pay per Click Ads',
                                                    'testone','NC_EDM'] ,'Others')


# In[59]:


leaddf['Lead Source'].value_counts(dropna=False)


# In[60]:


sns.countplot(x=leaddf['Lead Source'],hue=leaddf['Converted'])
plt.xticks(rotation=90)
plt.show()


# In[61]:


Converted_or_not_percent(leaddf, 'Lead Source')


# **Impression**
# * maximum lead come from google
# * 'Reference','Welingak Website' and others showed higher conversion rates.
# * maximum cold leads from Olark Chat, organic search, direct traffic, and google

# #### Column 'A free copy of Mastering The Interview' and 'Last Notable Activity'

# In[62]:


leaddf['A free copy of Mastering The Interview'].value_counts(dropna=False)


# In[63]:


leaddf['Last Notable Activity'].value_counts(dropna=False)


# In[64]:


## less than 30 occurences are grouped into one category

d = leaddf['Last Notable Activity'].value_counts(dropna=False)
to_be_droped_cols = d[d < 60].index

leaddf['Last Notable Activity'] = leaddf['Last Notable Activity'].replace(to_be_droped_cols, 'Other_Notable_activity')


# In[65]:


sns.countplot(x=leaddf['Last Notable Activity'],hue=leaddf['Converted'])
plt.xticks(rotation=90)
plt.show()


# In[66]:


## Check null values in any columns

leaddf.isnull().sum()


# #### Column 'Do Not Email'

# In[67]:


leaddf['Do Not Email'].value_counts(dropna=False)


# #### Imbalanced columns that can be dropped

# In[68]:


leaddf['Do Not Call'].value_counts(dropna=False)


# In[69]:


leaddf['Magazine'].value_counts(dropna=False)


# In[70]:


leaddf['Newspaper Article'].value_counts(dropna=False)


# In[71]:


leaddf['Search'].value_counts(dropna=False)


# In[72]:


leaddf['Newspaper'].value_counts(dropna=False)


# In[73]:


leaddf['Digital Advertisement'].value_counts(dropna=False)


# In[74]:


leaddf['X Education Forums'].value_counts(dropna=False)


# In[75]:


leaddf['Receive More Updates About Our Courses'].value_counts(dropna=False)


# In[76]:


leaddf['Through Recommendations'].value_counts(dropna=False)


# In[77]:


leaddf['Get updates on DM Content'].value_counts(dropna=False)


# In[78]:


leaddf['Update me on Supply Chain Content'].value_counts(dropna=False)


# In[79]:


leaddf['I agree to pay the amount through cheque'].value_counts(dropna=False)


# In[80]:


## drop the above imbalanced columns

to_be_drop_cols = ['Do Not Call', 'Magazine', 'Newspaper Article', 'Search', 'Newspaper', 
                   'Digital Advertisement','X Education Forums', 'Receive More Updates About Our Courses',
                   'Through Recommendations','Get updates on DM Content', 'Update me on Supply Chain Content', 
                   'Do Not Call', 'I agree to pay the amount through cheque']

leaddf.drop(to_be_drop_cols, axis=1, inplace=True)


# ### Create dummy variable for categorical columns

# In[81]:


## final categorical columns
leaddf[leaddf.columns.difference(leaddf.describe().columns)].columns


# In[82]:


## final categorical columns
leaddf.describe().columns


# In[83]:


leaddf.shape


# In[84]:


## dummy for 'Last Activity'

leaddf = pd.get_dummies(leaddf, columns=['Last Activity'], prefix  = 'Last Activity')


# In[85]:


## dummy for 'Last Notable Activity'

leaddf = pd.get_dummies(leaddf, columns=['Last Notable Activity'], prefix  = 'Last Notable Activity')
leaddf.drop(['Last Notable Activity_Other_Notable_activity'], axis=1, inplace=True)


# In[86]:


## dummy for 'Specialization'

leaddf = pd.get_dummies(leaddf, columns=['Specialization'], prefix  = 'Specialization')
leaddf.drop(['Specialization_Not Specified'], axis=1, inplace=True)


# In[87]:


## dummy for 'What is your current occupation'

leaddf = pd.get_dummies(leaddf, columns=['What is your current occupation'], prefix  = 'What is your current occupation', drop_first=True)


# In[88]:


## dummy for 'Lead Origin'

leaddf = pd.get_dummies(leaddf, columns=['Lead Origin'], prefix  = 'Lead Origin', drop_first=True)


# In[89]:


## dummy for 'Lead Source'

leaddf = pd.get_dummies(leaddf, columns=['Lead Source'], prefix  = 'Lead Source')
leaddf.drop(['Lead Source_Others'], axis=1, inplace=True)


# In[90]:


## dummy for 'Tags'

leaddf = pd.get_dummies(leaddf, columns=['Tags'], prefix  = 'Tags')
leaddf.drop(['Tags_Not Specified'], axis=1, inplace=True)


# In[91]:


## dummy for 'A free copy of Mastering The Interview'

leaddf['A free copy of Mastering The Interview'] = leaddf['A free copy of Mastering The Interview'].apply(lambda x: 1 if x == 'Yes' else 0)
dummy = pd.get_dummies(leaddf, columns=['A free copy of Mastering The Interview'], prefix  = 'A free copy of Mastering The Interview')


# In[92]:


## dummy for 'A free copy of Mastering The Interview'

leaddf['Do Not Email'] = leaddf['Do Not Email'].apply(lambda x: 1 if x == 'Yes' else 0)
dummy = pd.get_dummies(leaddf, columns=['Do Not Email'], prefix  = 'Do Not Email')


# In[93]:


## dummy for 'City'

leaddf = pd.get_dummies(leaddf, columns=['City'], prefix  = 'City', drop_first=True)


# In[94]:


## Current shape after adding dummies

leaddf.shape


# ### Train test split

# #### remove target variable to split the data

# In[95]:


targ_y = leaddf.pop('Converted')
indep_x = leaddf

xtrain,xtest,ytrain,ytest = train_test_split(indep_x, targ_y, test_size=0.2, random_state=20)


# In[96]:


## shape of train and test dataset

xtrain.shape, xtest.shape


# ### Normalization or scaling of columns

# In[97]:


numerical_cols = xtrain.select_dtypes(include=['float64', 'int64']).columns

scaler = MinMaxScaler()

xtrain[numerical_cols] = scaler.fit_transform(xtrain[numerical_cols])


# In[98]:


xtrain.describe()


# In[99]:


xtrain


# ### Training the model

# we will only use essential columns for our final model, in the process eliminate will some columns.

# In[100]:


## create function for repetative tasks

def create_model(cols):
    xtrain_sm = sm.add_constant(xtrain[cols])
    lr = sm.GLM(ytrain, xtrain_sm, family = sm.families.Binomial())
    lr_model = lr.fit()
    print(lr_model.summary())
    return lr_model

def vifs(cols):
    df = xtrain[cols]
    vif = pd.DataFrame()
    vif['Features'] = df.columns
    vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    vif['VIF'] = round(vif['VIF'],2)
    print(vif.sort_values(by='VIF',ascending=False))


# ### Use RFE to get initial set of columns (Automated + Manual)

# In[101]:


log_reg_model = LogisticRegression()
log_reg_model.fit(xtrain,ytrain)


# we will select initial 15 set of columns

# In[102]:


rfe = RFE(log_reg_model,n_features_to_select=15, step=1)
rfe.fit(xtrain,ytrain)


# In[103]:


rfe.support_


# In[104]:


list(zip(xtrain.columns,rfe.support_,rfe.ranking_))


# In[105]:


## columns selected by RFE
xtrain.columns[rfe.support_]


# In[106]:


xtrain_after_rfe = xtrain[xtrain.columns[rfe.support_]]


# In[107]:


## shape after RFE

xtrain_after_rfe.shape


# In[108]:


xtrain_after_rfe.shape , ytrain.shape


# In[109]:


plt.figure(figsize = (15,10))
sns.heatmap(xtrain_after_rfe.corr(),cmap='YlGnBu', annot=True)


# ### Build model

# #### Model 1

# In[110]:


cols = xtrain.columns[rfe.support_]

model1 = create_model(cols)
vifs(cols)


# Columns 'Tags_Diploma holder (Not Eligible)', 'Tags_invalid number', 'Tags_Not doing further education' can be dropped as they are high p-value

# In[111]:


cols = cols.drop(['Tags_Diploma holder (Not Eligible)', 'Tags_invalid number', 'Tags_Not doing further education'],1)


# #### Model 2

# In[112]:


model2 = create_model(cols)
vifs(cols)


# Now these set of columns and model 2 can be considered as final model as the p-values of all independent variables are almost zero and no high VIF variables or multi-colinearity.

# ### Final Model

# In[113]:


## final set of columns

final_cols = cols

log_reg_final = LogisticRegression()
log_reg_final.fit(xtrain[final_cols],ytrain)

print(log_reg_final.intercept_,log_reg_final.coef_)


# ### Evaluation

# In[114]:


ypred = log_reg_final.predict_proba(xtrain[cols])


# In[115]:


pred_final = pd.DataFrame({'Actual_Converted':ytrain,'Pred_Converted_Prob':ypred[:,1]})


# In[116]:


pred_final['Pred_Converted'] = pred_final.Pred_Converted_Prob.map(lambda x: 1 if x > 0.5 else 0)


# In[117]:


pred_final


# #### Confusion matrix

# In[118]:


confusion_mat = confusion_matrix(pred_final.Actual_Converted, pred_final.Pred_Converted)
confusion_mat


# In[119]:


TP,TN,FP,FN = confusion_mat[1,1],confusion_mat[0,0],confusion_mat[0,1],confusion_mat[1,0]


# In[120]:


## Sensitivity of the logistic regression model

sensitivity = TP/(TP+FN)
print('Sensitivity: ',sensitivity)

## Specificity of the logistic regression model

specificity = TN/(TN+FP)
print('Specificity: ',specificity)


# #### Accuracy score on training data

# In[121]:


accuracy_score(pred_final.Actual_Converted, pred_final.Pred_Converted)


# #### ROC curve 

# In[122]:


plot_roc_curve(log_reg_final, xtrain[final_cols], ytrain)
plt.show()


# #### Decide Cut-off 

# In[123]:


df_for_cutoff = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    pred_final['Pred_Converted'] = pred_final.Pred_Converted_Prob.map(lambda x: 1 if x > i else 0)
    cm = confusion_matrix(pred_final.Actual_Converted, pred_final.Pred_Converted)
    speci = cm[0,0]/(cm[0,0]+cm[0,1])
    sensi = cm[1,1]/(cm[1,0]+cm[1,1])
    df_for_cutoff.loc[i] =[i,accuracy_score(pred_final.Actual_Converted, pred_final.Pred_Converted),sensi,speci]


# In[124]:


df_for_cutoff


# In[125]:


df_for_cutoff.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# As seen above, 0.3 can be considered as a good cut-off as **'accuracy', 'sensitivity', 'specificity'** have optimum values with **0.9141, 0.9011, 0.9170** respectively

# In[126]:


pred_final['Pred_Converted'] = pred_final.Pred_Converted_Prob.map(lambda x: 1 if x > 0.3 else 0)
pred_final['Lead Score'] = pred_final.Pred_Converted_Prob.map(lambda x: round(x*100, 2))
pred_final


# ### Scaling and prediction on test data 

# In[127]:


numerical_cols = xtest.select_dtypes(include=['float64', 'int64']).columns
xtest[numerical_cols] = scaler.transform(xtest[numerical_cols])


# In[128]:


xtest = xtest[final_cols]
xtest.head()


# In[129]:


ytest_pred = log_reg_final.predict_proba(xtest)


# In[130]:


pred_test_final = pd.DataFrame({'Actual_Converted':ytest,'Pred_Converted_Prob':ytest_pred[:,1]})


# In[131]:


pred_test_final['Pred_Converted'] = pred_test_final.Pred_Converted_Prob.map(lambda x: 1 if x > 0.5 else 0)


# In[132]:


pred_test_final.head()


# #### Accuracy and confusion matrix on test data

# In[133]:


accuracy_score(pred_test_final.Actual_Converted, pred_test_final.Pred_Converted)


# In[134]:


confusion_mat_test = confusion_matrix(pred_test_final.Actual_Converted, pred_test_final.Pred_Converted)
confusion_mat_test


# In[135]:


TP,TN,FP,FN = confusion_mat_test[1,1],confusion_mat_test[0,0],confusion_mat_test[0,1],confusion_mat_test[1,0]


# In[136]:


## Sensitivity of the logistic regression model

sensitivity = TP/(TP+FN)
print('Sensitivity: ',sensitivity)

## Specificity of the logistic regression model

specificity = TN/(TN+FP)
print('Specificity: ',specificity)


# **'accuracy', 'sensitivity', 'specificity'** on test data are **0.9241, 0.8621, 0.9621** respectively

# ### Final model results

# * **Train Data** - **'accuracy', 'sensitivity', 'specificity'** - **0.9141, 0.9011, 0.9170** 
# * **Test Data** - **'accuracy', 'sensitivity', 'specificity'** - **0.9241, 0.8621, 0.9621**
# 
# The model is giving promising results as shown by above evaluation metrics Hence, this should help make identify hot lead and significantly impact the business.
