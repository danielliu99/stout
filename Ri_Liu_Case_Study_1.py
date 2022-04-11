#!/usr/bin/env python
# coding: utf-8

# •	Describe the dataset and any issues with it. 
# 
# •	Generate a minimum of 5 unique visualizations using the data and write a brief description of your observations. Additionally, all attempts should be made to make the visualizations visually appealing
# 
# •	Create a feature set and create a model which predicts interest_rate using at least 2 algorithms. Describe any data cleansing that must be performed and analysis when examining the data.
# 
# •	Visualize the test results and propose enhancements to the model, what would you do if you had more time. Also describe assumptions you made and your approach.
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


pd.set_option('display.float_format',lambda x : '%.4f' % x)


# In[3]:


data_raw = pd.read_csv('loans_full_schema.csv')
data = data_raw.copy()


# In[4]:


data.shape


# In[5]:


data.head(5).append(data.tail(5))


# In[6]:


meta_dict = []
for col in data.columns:
    # if data[col].dtype == 'int64':
    #     ordinal_cols.append(col)
    type = data[col].dtype
    unique = 0
    if type == 'object':
        unique = len(data[col].unique())
    NAs = data[col].isnull().sum()
    NA_prop = NAs / len(data[col]) * 100
    dict_tmp = {'Name': col, 'Type': type, 'Unique': unique, 'NAs': NAs, 'NA%': NA_prop}
    meta_dict.append(dict_tmp)


# In[7]:


meta = pd.DataFrame(meta_dict)
meta = meta.set_index('Name').drop('interest_rate')


# In[8]:


meta.sort_values('NA%', ascending=False)


# In[9]:


# column names of columns having more than 50% missing values
high_NA = meta[meta['NA%'] > 50].index
high_NA


# In[10]:


meta_new = meta[meta['NA%'] <= 50]
# meta_new.sort_values('NA%', ascending=False)


# In[11]:


pd.DataFrame({'Number of variables': meta_new.groupby('Type').size()})


# In[12]:


meta_new.sort_values('NA%', ascending=False)


# In[13]:


# `emp_title` has too many unique text values, so I drop it
meta_new = meta_new.drop('emp_title')
meta_new.shape


# ### Statistical Attributes of Variables

# In[14]:


data[meta_new[meta_new.Type == 'float64'].index].describe().T


# In[15]:


for col in meta_new[meta_new.Type == 'float64'].index:
    plt.figure()
    sns.histplot(data[col],bins=10)
    plt.show()


# In[16]:


data[meta_new[meta_new.Type == 'int64'].index].describe().T


# In[17]:


for col in meta_new[meta_new.Type == 'int64'].index:
    plt.figure()
    sns.histplot(data[col], bins=20)
    plt.show()


# In[91]:


# for col in meta_new[meta_new.Type == 'object'].index:
#     print(data[col].value_counts())


# `num_accounts_120d_past_due` is all 0, so it will be dropped 
# 
# `current_accounts_delinq` and `num_accounts_30d_past_due` both has only 1 value that is not 0, so they will be dropped
# 
# There are some variables that has extremely unbalanced distribution or have outliers (but may not be influential), such as `annual_income`, `debt_to_income`, `paid_late_fees`, `num_historical_failed_to_pay`, `total_collection_amount_ever`. 
# 
# `term` has only two values 36 and 60, though it is type of int64. 

# In[19]:


meta_new = meta_new.drop(['num_accounts_120d_past_due', 'current_accounts_delinq', 'num_accounts_30d_past_due'])


# In[20]:


Y = data['interest_rate']
sns.histplot(Y, bins=30)


# ## Preliminary Feature Selection 

# ### Correlation of numerical features

# In[21]:


from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr 


# In[22]:


# missing value?
data_nonan = data.dropna(axis=0, how='any')


# In[23]:


correlation_dict = []
for col in meta_new[meta_new.Type == 'float64'].index:
    correlation, pvalue = pearsonr(data_nonan[col], data_nonan['interest_rate'])
    dict_tmp = {'Name': col, 'Correlation': correlation, 'P-value':pvalue}
    correlation_dict.append(dict_tmp)


# In[24]:


pd.DataFrame(correlation_dict).set_index('Name').sort_values('P-value')


# In[25]:


correlation_dict = []
for col in meta_new[meta_new.Type == 'int64'].index:
    correlation, pvalue = pearsonr(data_nonan[col], data_nonan['interest_rate'])
    dict_tmp = {'Name': col, 'Correlation': correlation, 'P-value':pvalue}
    correlation_dict.append(dict_tmp)


# In[26]:


pd.DataFrame(correlation_dict).set_index('Name').sort_values('P-value')


# ### XGBoost for numerical features

# In[27]:


import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_xgb = []
data_xgb = data[meta_new[meta_new.Type == 'float64'].index.append(meta_new[meta_new.Type == 'int64'].index)]
labels = data['interest_rate']

X = []
for row in data_xgb.values:
    row = [float(x) for x in row]
    X.append(row)

y = [float(x) for x in labels]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=0)

model = xgb.XGBRegressor(n_estimators=160, 
                         learning_rate=0.1, 
                         max_depth=5, 
                         silent=True, 
                         objective='reg:gamma')
                         

model.fit(X_train, y_train)
model.get_booster().feature_names =  list(data_xgb.columns)

fig,ax = plt.subplots(figsize=(15,10))
plot_importance(model, ax=ax)
plt.show()


# The features that have significant correlation with `interest_rate` and the features that have feature importance higher than the 'elbow point' are selected currently. 

# In[28]:


continuous_cols = ['paid_interest', 'annual_income', 'months_since_last_credit_inquiry', 'paid_principal', 'term', 'total_debit_limit', 'inquiries_last_12m', 'total_credit_limit', 'num_mort_accounts', 'accounts_opened_24m', 'balance', 'installment', 'paid_total', 'loan_amount']


# In[29]:


sns.heatmap(abs(data[continuous_cols].corr()))


# The shallow color cells indicate the strong correlations between two features. 
# 
# 'paid_total' with 'paid_principle', 'balance' and 'installment', 'loan_amount' with 'balance' and 'installment' are correlated pairs. 
# 
# `paid_total`, `installment` and `loan_amount` will be dropped as they have lower correlation or feature importance than their correlated ones. 
# 
# Now, we have 11 numerical features.

# In[30]:


continuous_cols = ['paid_interest', 'annual_income', 'months_since_last_credit_inquiry', 'paid_principal', 'total_debit_limit', 'inquiries_last_12m', 'total_credit_limit', 'num_mort_accounts', 'accounts_opened_24m', 'balance']
len(continuous_cols)


# In[31]:


sns.heatmap(abs(data[continuous_cols].corr()))


# In[32]:


data['term_new'] = 0
data.loc[data['term'] == 60, 'term_new'] = 1


# ### Categorical Features

# In[33]:


state_group = pd.DataFrame(data.groupby('state')['interest_rate'].mean()).reset_index().reset_index()


# In[34]:


fig,ax = plt.subplots(figsize=(10,8))
sns.scatterplot(state_group['index'], state_group.interest_rate, ax=ax)
def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], str(point['val']))

label_point(state_group['index'], state_group.interest_rate, state_group.state, plt.gca())  


# In[35]:


from scipy import stats


# In[36]:


stats.f_oneway(data[data.homeownership == 'MORTGAGE'].interest_rate, data[data.homeownership == 'RENT'].interest_rate, data[data.homeownership == 'OWN'].interest_rate)


# In[37]:


import pingouin as pg


# In[38]:


catgorical_cols = list(meta_new[(meta_new.Type == 'object') & (meta_new.Unique < 15)].index)


# In[39]:


anova_dict = []
for col in catgorical_cols:
    pvalue = pg.anova(data, 'interest_rate', col)['p-unc'][0]
    dict_tmp = {'Name': col, 'P-value': pvalue}
    anova_dict.append(dict_tmp)


# In[40]:


pd.DataFrame(anova_dict).set_index('Name')


# In[41]:


pg.pairwise_tukey(data, 'interest_rate', 'homeownership')


# In[42]:


pg.pairwise_tukey(data, 'interest_rate', 'verified_income')


# In[43]:


tukey_loanPurpose = pg.pairwise_tukey(data, 'interest_rate', 'loan_purpose').sort_values('p-tukey')


# In[44]:


tukey_loanPurpose[tukey_loanPurpose['p-tukey'] < 0.05]


# In[45]:


pg.pairwise_tukey(data, 'interest_rate', 'application_type').sort_values('p-tukey')


# In[46]:


pg.pairwise_tukey(data, 'interest_rate', 'grade').sort_values('p-tukey')


# In[47]:


pg.pairwise_tukey(data, 'interest_rate', 'loan_status').sort_values('p-tukey')


# In[48]:


pg.pairwise_tukey(data, 'interest_rate', 'initial_listing_status').sort_values('p-tukey')


# In[49]:


pg.pairwise_tukey(data, 'interest_rate', 'disbursement_method').sort_values('p-tukey')


# `state` to be recoded as 2 levels ('HI' or 'ND' and 'other')
# 
# `homeownership` to be rocoded as `Rent` and `other`
# 
# `verified_income` to be kept the 3 levels
# 
# `loan_purpose` to be recoded as `credit_card`, `debt_consolidation` and `other`
# 
# `grade` to be recoded as ordinal
# 
# `loan_status` to be recoded as `Current`, `Fully paid` and `other`
# 
# `application_type`, `initial_listing_status` and `disbursement_method` to be recoded as binary

# In[50]:


# state
data['state_new'] = 0
data.loc[(data['state'] == 'HI') | (data['state'] == 'ND'), 'state_new'] = 1


# In[51]:


# homeownership 
data['homeownership_new'] = 0
data.loc[data['homeownership'] == 'RENT', 'homeownership_new'] = 1


# In[52]:


# verified_income
data['verified_income_sourceVerified'] = 0
data.loc[data['verified_income'] == 'Source Verified', 'verified_income_sourceVerified'] = 1

data['verified_income_Verified'] = 0
data.loc[data['verified_income'] == 'Verified', 'verified_income_Verified'] = 1


# In[53]:


# loan purpose
data['loan_purpose_card'] = 0
data.loc[data['loan_purpose'] == 'credit_card', 'loan_purpose_card'] = 1

data['loan_purpose_consolid'] = 0
data.loc[data['loan_purpose'] == 'debt_consolidation', 'loan_purpose_consolid'] = 1


# In[54]:


# grade
data['grade_new'] = 0
data.loc[data['grade'] == 'A', 'grade_new'] = 1
data.loc[data['grade'] == 'B', 'grade_new'] = 2
data.loc[data['grade'] == 'C', 'grade_new'] = 3
data.loc[data['grade'] == 'D', 'grade_new'] = 4
data.loc[data['grade'] == 'E', 'grade_new'] = 5
data.loc[data['grade'] == 'F', 'grade_new'] = 6
data.loc[data['grade'] == 'G', 'grade_new'] = 7


# In[55]:


# loan_status
data['loan_status_current'] = 0
data.loc[data['loan_status'] == 'Current', 'loan_status_current'] = 1

data['loan_status_paid'] = 0
data.loc[data['loan_status'] == 'Fully Paid', 'loan_status_paid'] = 1


# In[56]:


# application_type
data['application_type_new'] = 0
data.loc[data['application_type'] == 'joint', 'application_type_new'] = 1


# In[57]:


# initial_listing_status
data['initial_listing_status_new'] = 0
data.loc[data['initial_listing_status'] == 'fractional', 'initial_listing_status_new'] = 1


# In[58]:


# disbursement_method
data['disbursement_method_new'] = 0
data.loc[data['disbursement_method'] == 'DirectPay', 'disbursement_method_new'] = 1


# ### XGBoost for categorical features

# In[59]:


categorical_cols = ['state_new', 'homeownership_new', 'verified_income_sourceVerified', 'verified_income_Verified', 'loan_purpose_card', 'loan_purpose_consolid', 'grade_new', 'loan_status_current', 'loan_status_paid', 'application_type_new', 'initial_listing_status_new', 'disbursement_method_new']
len(categorical_cols)


# In[60]:


data_xgb = []
data_xgb = data[categorical_cols]
labels = data['interest_rate']

X = []
for row in data_xgb.values:
    row = [float(x) for x in row]
    X.append(row)

y = [float(x) for x in labels]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=0)

model = xgb.XGBRegressor(n_estimators=300, 
                         learning_rate=0.1, 
                         max_depth=15, 
                         silent=True, 
                         objective='reg:gamma')
                         

model.fit(X_train, y_train)
model.get_booster().feature_names =  list(data_xgb.columns)

fig,ax = plt.subplots(figsize=(15,10))
plot_importance(model, ax=ax)
plt.show()


# According to feature importance, there are 8 recoded categorical features remained currently. 

# In[61]:


categorical_cols = ['homeownership_new', 'verified_income_sourceVerified', 'verified_income_Verified', 'loan_purpose_card', 'loan_purpose_consolid', 'grade_new', 'application_type_new', 'initial_listing_status_new', 'term_new']
len(categorical_cols)


# ## Current Feature Set

# In[62]:


feature_cols = continuous_cols+categorical_cols
len(feature_cols)


# In[63]:


meta_dict = []
for col in feature_cols:
    # if data[col].dtype == 'int64':
    #     ordinal_cols.append(col)
    type = data[col].dtype
    unique = 0
    if type == 'object':
        unique = len(data[col].unique())
    NAs = data[col].isnull().sum()
    NA_prop = NAs / len(data[col]) * 100
    dict_tmp = {'Name': col, 'Type': type, 'Unique': unique, 'NAs': NAs, 'NA%': NA_prop}
    meta_dict.append(dict_tmp)


# In[64]:


pd.DataFrame(meta_dict).set_index('Name')


# In[65]:


# for col in feature_cols:
#     plt.figure()
#     sns.histplot(data[col], bins=20)
#     plt.show()


# In[66]:


data = data.fillna({'months_since_last_credit_inquiry': 0})


# ## Modelling

# In[67]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from  sklearn.metrics import  mean_squared_error as MSE
from sklearn.metrics import r2_score


# In[68]:


data_new = data.copy()


# In[69]:


for col in continuous_cols:
    # print(sum(data[col] < np.quantile(data[col], 0.999)))
    data_new = data_new.loc[data_new[col] < np.quantile(data_new[col], 0.99), :]


# In[70]:


X = data[feature_cols].values
y = data['interest_rate'].values


# ### Baseline Model

# In[71]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=10)


# In[72]:


reg = LinearRegression()
reg.fit(X_train, y_train)


# In[73]:


y_pred = reg.predict(X_test)


# In[74]:


reg.score(X_test, y_test)


# In[75]:


mse=MSE(y_pred,y_test)
print('RMSE: ', np.sqrt(mse))


# In[76]:


r_2 = r2_score(y_pred = y_pred, y_true = y_test)
print('R-squared: ', r_2)


# ### Lasso

# In[77]:


from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV


# In[78]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=10)


# In[79]:


lasso = LassoCV(alphas = [0,0.1,0.01, 0.001, 0.0001])
lasso.fit(X_train, y_train)


# In[80]:


lasso.alpha_


# In[81]:


lasso.score(X_test, y_test)


# In[82]:


y_pred = lasso.predict(X_test)


# In[83]:


mse=MSE(y_pred=y_pred, y_true=y_test)
print('RMSE: ', np.sqrt(mse))


# In[84]:


lasso.score(X_train, y_train)


# In[85]:


feat_importance = pd.DataFrame({'column': data[feature_cols].columns, 'coef':list(lasso.coef_)})
feat_importance['importance'] = np.abs(feat_importance.coef)
feat_importance = feat_importance.sort_values(by='importance', ascending=False)
feat_importance


# In[86]:


# feature_cols = list(feat_importance.loc[feat_importance.importance > 0.001, 'column'].values)
feature_cols


# In[ ]:





# ### Regression Tree

# In[87]:


from sklearn.tree import DecisionTreeRegressor


# In[88]:


tree = DecisionTreeRegressor(random_state=10)
tree.fit(X_train, y_train)


# In[89]:


tree.score(X_test, y_test)


# In[90]:


y_pred = tree.predict(X_test)
MSE(y_pred=y_pred, y_true=y_test)


# In[ ]:




