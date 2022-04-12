#!/usr/bin/env python
# coding: utf-8

# # 1 Read Data

# In[85]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[86]:


import warnings
warnings.filterwarnings("ignore")


# In[87]:


pd.set_option('display.float_format',lambda x : '%.4f' % x)


# In[88]:


data_raw = pd.read_csv('loans_full_schema.csv')
data = data_raw.copy()


# In[89]:


data.shape


# In[90]:


data.head(5).append(data.tail(5))


# # 2 Describe Data

# In[91]:


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


# In[92]:


meta = pd.DataFrame(meta_dict)
meta = meta.set_index('Name').drop('interest_rate')


# In[93]:


meta.sort_values('NA%', ascending=False).head(15)


# In[94]:


# column names of columns having more than 50% missing values
high_NA = meta[meta['NA%'] > 50].index
high_NA


# In[95]:


meta_new = meta[meta['NA%'] <= 50]
# meta_new.sort_values('NA%', ascending=False)


# In[96]:


pd.DataFrame({'Number of variables': meta_new.groupby('Type').size()})


# In[97]:


meta_new.sort_values('NA%', ascending=False).head(10)


# In[98]:


# `emp_title` has too many unique text values, so I drop it
meta_new = meta_new.drop('emp_title')
meta_new.shape


# # 3 Exploratory Data Analysis

# ### Statistical Attributes of Variables

# In[99]:


data[meta_new[meta_new.Type == 'float64'].index].describe().T


# In[100]:


for col in meta_new[meta_new.Type == 'float64'].index:
    plt.figure()
    sns.histplot(data[col],bins=10)
    plt.show()


# In[101]:


data[meta_new[meta_new.Type == 'int64'].index].describe().T


# In[102]:


for col in meta_new[meta_new.Type == 'int64'].index:
    plt.figure()
    sns.histplot(data[col], bins=20)
    plt.show()


# In[103]:


# for col in meta_new[meta_new.Type == 'object'].index:
#     print(data[col].value_counts())


# `num_accounts_120d_past_due` is all 0, so it will be dropped 
# 
# `current_accounts_delinq` and `num_accounts_30d_past_due` both has only 1 value that is not 0, so they will be dropped
# 
# There are some variables that has extremely unbalanced distribution or have outliers (but may not be influential), such as `annual_income`, `debt_to_income`, `paid_late_fees`, `num_historical_failed_to_pay`, `total_collection_amount_ever`. 
# 
# `term` has only two values 36 and 60, though it is type of int64. 

# In[104]:


meta_new = meta_new.drop(['num_accounts_120d_past_due', 'current_accounts_delinq', 'num_accounts_30d_past_due'])


# In[105]:


Y = data['interest_rate']
sns.histplot(Y, bins=30)


# ## Feature Selection 

# ### Correlation of numerical features

# In[106]:


from scipy.stats import pearsonr 


# In[107]:


data_nonan = data.dropna(axis=0, how='any')


# In[108]:


correlation_dict = []
for col in meta_new[meta_new.Type == 'float64'].index:
    correlation, pvalue = pearsonr(data_nonan[col], data_nonan['interest_rate'])
    dict_tmp = {'Name': col, 'Correlation': correlation, 'P-value':pvalue}
    correlation_dict.append(dict_tmp)


# In[109]:


pd.DataFrame(correlation_dict).set_index('Name').sort_values('P-value')


# In[110]:


correlation_dict = []
for col in meta_new[meta_new.Type == 'int64'].index:
    correlation, pvalue = pearsonr(data_nonan[col], data_nonan['interest_rate'])
    dict_tmp = {'Name': col, 'Correlation': correlation, 'P-value':pvalue}
    correlation_dict.append(dict_tmp)


# In[111]:


pd.DataFrame(correlation_dict).set_index('Name').sort_values('P-value')


# ### XGBoost for numerical features

# In[112]:


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

# In[113]:


continuous_cols = ['paid_interest', 'annual_income', 'months_since_last_credit_inquiry', 
                   'paid_principal', 'term', 'total_debit_limit', 'inquiries_last_12m', 
                   'total_credit_limit', 'num_mort_accounts', 'accounts_opened_24m', 'balance', 
                   'installment', 'paid_total', 'loan_amount']


# In[114]:


sns.heatmap(abs(data[continuous_cols].corr()))


# The shallow color cells indicate the strong correlations between two features. 
# 
# 'paid_total' with 'paid_principle', 'balance' and 'installment', 'loan_amount' with 'balance' and 'installment' are correlated pairs. 
# 
# `paid_total`, `installment` and `loan_amount` will be dropped as they have lower correlation or feature importance than their correlated ones. 
# 
# Now, we have 11 numerical features.

# In[115]:


continuous_cols = ['paid_interest', 'annual_income', 'months_since_last_credit_inquiry', 
                   'paid_principal', 'total_debit_limit', 'inquiries_last_12m', 'total_credit_limit', 
                   'num_mort_accounts', 'accounts_opened_24m', 'balance']
len(continuous_cols)


# In[116]:


sns.heatmap(abs(data[continuous_cols].corr()))


# In[117]:


data['term_new'] = 0
data.loc[data['term'] == 60, 'term_new'] = 1


# ### Categorical Features

# In[118]:


state_group = pd.DataFrame(data.groupby('state')['interest_rate'].mean()).reset_index().reset_index()


# In[119]:


fig,ax = plt.subplots(figsize=(10,8))
sns.scatterplot(state_group['index'], state_group.interest_rate, ax=ax)
def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], str(point['val']))

label_point(state_group['index'], state_group.interest_rate, state_group.state, plt.gca())  


# In[120]:


from scipy import stats


# In[121]:


stats.f_oneway(data[data.homeownership == 'MORTGAGE'].interest_rate, 
               data[data.homeownership == 'RENT'].interest_rate, 
               data[data.homeownership == 'OWN'].interest_rate)


# In[122]:


import pingouin as pg


# In[123]:


catgorical_cols = list(meta_new[(meta_new.Type == 'object') & (meta_new.Unique < 15)].index)


# In[124]:


anova_dict = []
for col in catgorical_cols:
    pvalue = pg.anova(data, 'interest_rate', col)['p-unc'][0]
    dict_tmp = {'Name': col, 'P-value': pvalue}
    anova_dict.append(dict_tmp)


# In[125]:


pd.DataFrame(anova_dict).set_index('Name')


# In[126]:


pg.pairwise_tukey(data, 'interest_rate', 'homeownership')


# In[127]:


pg.pairwise_tukey(data, 'interest_rate', 'verified_income')


# In[128]:


tukey_loanPurpose = pg.pairwise_tukey(data, 'interest_rate', 'loan_purpose').sort_values('p-tukey')


# In[129]:


tukey_loanPurpose[tukey_loanPurpose['p-tukey'] < 0.05]


# In[130]:


pg.pairwise_tukey(data, 'interest_rate', 'application_type').sort_values('p-tukey')


# In[131]:


pg.pairwise_tukey(data, 'interest_rate', 'grade').sort_values('p-tukey')


# In[132]:


pg.pairwise_tukey(data, 'interest_rate', 'loan_status').sort_values('p-tukey')


# In[133]:


pg.pairwise_tukey(data, 'interest_rate', 'initial_listing_status').sort_values('p-tukey')


# In[134]:


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

# In[135]:


# state
data['state_new'] = 0
data.loc[(data['state'] == 'HI') | (data['state'] == 'ND'), 'state_new'] = 1


# In[136]:


# homeownership 
data['homeownership_new'] = 0
data.loc[data['homeownership'] == 'RENT', 'homeownership_new'] = 1


# In[137]:


# verified_income
data['verified_income_sourceVerified'] = 0
data.loc[data['verified_income'] == 'Source Verified', 'verified_income_sourceVerified'] = 1

data['verified_income_Verified'] = 0
data.loc[data['verified_income'] == 'Verified', 'verified_income_Verified'] = 1


# In[138]:


# loan purpose
data['loan_purpose_card'] = 0
data.loc[data['loan_purpose'] == 'credit_card', 'loan_purpose_card'] = 1

data['loan_purpose_consolid'] = 0
data.loc[data['loan_purpose'] == 'debt_consolidation', 'loan_purpose_consolid'] = 1


# In[139]:


# grade
data['grade_new'] = 0
data.loc[data['grade'] == 'A', 'grade_new'] = 1
data.loc[data['grade'] == 'B', 'grade_new'] = 2
data.loc[data['grade'] == 'C', 'grade_new'] = 3
data.loc[data['grade'] == 'D', 'grade_new'] = 4
data.loc[data['grade'] == 'E', 'grade_new'] = 5
data.loc[data['grade'] == 'F', 'grade_new'] = 6
data.loc[data['grade'] == 'G', 'grade_new'] = 7


# In[140]:


# loan_status
data['loan_status_current'] = 0
data.loc[data['loan_status'] == 'Current', 'loan_status_current'] = 1

data['loan_status_paid'] = 0
data.loc[data['loan_status'] == 'Fully Paid', 'loan_status_paid'] = 1


# In[141]:


# application_type
data['application_type_new'] = 0
data.loc[data['application_type'] == 'joint', 'application_type_new'] = 1


# In[142]:


# initial_listing_status
data['initial_listing_status_new'] = 0
data.loc[data['initial_listing_status'] == 'fractional', 'initial_listing_status_new'] = 1


# In[143]:


# disbursement_method
data['disbursement_method_new'] = 0
data.loc[data['disbursement_method'] == 'DirectPay', 'disbursement_method_new'] = 1


# ### XGBoost for categorical features

# In[144]:


categorical_cols = ['state_new', 'homeownership_new', 'verified_income_sourceVerified', 
                    'verified_income_Verified', 'loan_purpose_card', 
                    'loan_purpose_consolid', 'grade_new', 'loan_status_current', 
                    'loan_status_paid', 'application_type_new', 'initial_listing_status_new', 
                    'disbursement_method_new']
len(categorical_cols)


# In[145]:


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

# In[146]:


categorical_cols = ['homeownership_new', 'verified_income_sourceVerified', 'verified_income_Verified', 
                    'loan_purpose_card', 'loan_purpose_consolid', 'grade_new', 'application_type_new', 
                    'initial_listing_status_new', 'term_new']
len(categorical_cols)


# ## Current Feature Set

# In[147]:


feature_cols = continuous_cols+categorical_cols
len(feature_cols)


# In[148]:


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


# In[149]:


pd.DataFrame(meta_dict).set_index('Name')


# In[150]:


# for col in feature_cols:
#     plt.figure()
#     sns.histplot(data[col], bins=20)
#     plt.show()


# In[ ]:





# In[151]:


from sklearn.impute import SimpleImputer


# In[152]:


mean_imputer = SimpleImputer(strategy = 'mean')


# In[153]:


data['months_since_last_credit_inquiry'] = mean_imputer.fit_transform(data[['months_since_last_credit_inquiry']])


# ## 4 Modelling

# In[154]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from  sklearn.metrics import  mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV


# In[155]:


data_new = data.copy()


# In[156]:


X = data[feature_cols].values
y = data['interest_rate'].values


# ### Baseline Model

# In[157]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=10)


# In[158]:


reg = LinearRegression()
reg.fit(X_train, y_train)


# In[159]:


y_pred = reg.predict(X_test)


# In[160]:


lr_r2 = reg.score(X_test, y_test)
print('R-squared: ', lr_r2)


# In[161]:


lr_rmse = np.sqrt(MSE(y_pred,y_test))
print('RMSE: ', lr_rmse)


# In[ ]:





# ## Feature Engineering

# In[162]:


for col in continuous_cols:
    plt.figure()
    sns.histplot(data[col])
    plt.show()


# ### Binning for unbalanced features

# In[163]:


# total_credit_limit_cut = pd.factorize(pd.cut(data.total_credit_limit, 
#     list(np.quantile(data.total_credit_limit, [0, 0.33, 0.67, 0.9, 1])), 
#     labels=[0, 1, 2, 3]))[0]
to_be_cut = ['paid_principal', 'annual_income', 'total_debit_limit']
for col in to_be_cut:
    new_colname = col + '_' + 'cut'
    new_col = pd.factorize(pd.cut(data.total_credit_limit, 
    list(np.quantile(data.total_credit_limit, [0, 0.33, 0.67, 0.9, 1])), 
    labels=[0, 1, 2, 3]))[0]
    data[new_colname] = new_col 
    continuous_cols.remove(col)
    continuous_cols = continuous_cols + [new_colname]


# In[ ]:


data = data.fillna({'annual_income_cut': 0, 
    'paid_principal_cut': 0, 'total_debit_limit_cut': 0, 
    'total_credit_limit_cut': 0})


# #### Feature set

# In[245]:


feature_cols


# ### Interaction terms

# In[188]:


from sklearn.preprocessing import PolynomialFeatures


# In[190]:


# poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# data_poly = pd.DataFrame(data=poly.fit_transform(data[feature_cols]), 
#                             columns=poly.get_feature_names_out(feature_cols))

data_poly = data[feature_cols]
for cate in categorical_cols:
    for col in continuous_cols:
        new_colname = cate + '_' + col
        new_col = data[cate] * data[col]
        data_poly[new_colname] = new_col


# In[191]:


data_poly.describe().T.sort_values('std')


# In[192]:


X = data_poly.values
y = data['interest_rate'].values


# In[167]:


X = data[feature_cols]
y = data['interest_rate'].values


# In[193]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=10)


# In[178]:


feature_cols = continuous_cols+categorical_cols


# ### Lasso

# In[194]:


from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV


# In[195]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=10)
alphas = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
lasso_dict = []
for alpha in alphas:
    lasso = LassoCV(alphas=[alpha])
    lasso.fit(X_train, y_train)
    tmp_r_2 = lasso.score(X_test, y_test)
    y_pred = lasso.predict(X_test)
    tmp_rmse = np.sqrt(MSE(y_pred=y_pred, y_true=y_test))
    dict_tmp = {'alpha': alpha, 'R-squared': tmp_r_2, 'RMSE': tmp_rmse}
    lasso_dict.append(dict_tmp)


# In[197]:


lasso_dict = pd.DataFrame(lasso_dict).sort_values('alpha', ascending=True)


# In[198]:


plt.figure(12, figsize=(10,5))
plt.subplot(121)
sns.pointplot(lasso_dict.alpha, lasso_dict['R-squared'])
plt.subplot(122)
sns.pointplot(lasso_dict.alpha, lasso_dict.RMSE)


# In[199]:


lasso = LassoCV(alphas=[0.01])
lasso.fit(X_train, y_train)


# In[200]:


lasso_r2 = lasso.score(X_test, y_test)
print('R-squared: ', lasso_r2)


# In[201]:


y_pred = lasso.predict(X_test)
lasso_rmse = np.sqrt(MSE(y_pred=y_pred, y_true=y_test))
print('RMSE: ', lasso_rmse)


# In[204]:


plt.scatter(y_test,y_pred-y_test,
            c='limegreen',
            edgecolor='white',
            marker='s',
            s=35,
            alpha=0.6,
            label='Test Data')
plt.xlabel("Y_true")
plt.ylabel("Residuals")


# In[205]:


feat_importance = pd.DataFrame({'column': data_poly.columns, 'coef':list(lasso.coef_)})
feat_importance['importance'] = np.abs(feat_importance.coef)
feat_importance = feat_importance.sort_values(by='importance', ascending=False)
feat_importance.head(15)


# ### Regression Tree

# In[206]:


from sklearn.tree import DecisionTreeRegressor


# In[228]:


# X = data_poly[feat_cols]
# y = data['interest_rate'].values


# In[210]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=10)
# min_samples_leaf = [1,3,5,10,15,20]
# max_depth = [10,50,100,150]
min_samples_leaf = [10,20,30,40,50]
max_depth = [20,50,100,150,200]
tree_dict = []
for leaf in min_samples_leaf:
    for depth in max_depth:        
        param = {'min_samples_leaf':[leaf], 'max_depth':[depth]}
        tree = GridSearchCV(DecisionTreeRegressor(), param_grid=param, cv=5)
        tree.fit(X_train, y_train)
        tmp_r_2 = tree.score(X_test, y_test)
        y_pred = tree.predict(X_test)
        tmp_rmse = np.sqrt(MSE(y_pred=y_pred, y_true=y_test))
        dict_tmp = {'min_samples_leaf': leaf, 'max_depth': depth, 'R-squared': tmp_r_2, 'RMSE': tmp_rmse}
        tree_dict.append(dict_tmp)


# In[211]:


tree_dict = pd.DataFrame(tree_dict)


# In[212]:


plt.figure(12, figsize=(10,5))
plt.subplot(121)
sns.pointplot(tree_dict.min_samples_leaf, tree_dict['R-squared'], tree_dict.max_depth)
plt.subplot(122)
sns.pointplot(tree_dict.min_samples_leaf, tree_dict['RMSE'], tree_dict.max_depth)


# In[218]:


# Use the best parameters to fit the model
param = {'min_samples_leaf':[40], 'max_depth':[50]}
tree = GridSearchCV(DecisionTreeRegressor(), param_grid=param, cv=5)
tree.fit(X_train, y_train)


# In[219]:


tree.best_params_


# In[220]:


y_pred = tree.predict(X_test)


# In[221]:


tree_r2 = tree.score(X_test, y_test)
print('R-squared: ', tree_r2)


# In[222]:


y_pred = tree.predict(X_test)
tree_rmse = np.sqrt(MSE(y_pred=y_pred, y_true=y_test))
print('RMSE: ', tree_rmse)


# In[224]:


plt.scatter(y_test,y_pred-y_test,
            c='limegreen',
            edgecolor='white',
            marker='s',
            s=35,
            alpha=0.6,
            label='Test Data')
plt.xlabel("Y_true")
plt.ylabel("Residuals")


# ### Random Forest

# In[229]:


from sklearn.ensemble import RandomForestRegressor


# In[226]:


# To save time on fitting, use less feature set for random forest temporarily
# It has had better performance than the other models
X = data[feature_cols]
y = data['interest_rate'].values


# In[237]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=10)
n_estimators = [100, 500, 1500]
max_depth = [30,50,100]
forest_dict = []
for n in n_estimators:
    for depth in max_depth:        
        param = {'n_estimators':[n], 'max_depth':[depth]}
        forest = GridSearchCV(RandomForestRegressor(), param_grid=param, cv=5, n_jobs=-1)
        forest.fit(X_train, y_train)
        tmp_r_2 = forest.score(X_test, y_test)
        y_pred = forest.predict(X_test)
        tmp_rmse = np.sqrt(MSE(y_pred=y_pred, y_true=y_test))
        dict_tmp = {'n_estimators': n, 'max_depth': depth, 'R-squared': tmp_r_2, 'RMSE': tmp_rmse}
        forest_dict.append(dict_tmp)


# In[238]:


forest_dict = pd.DataFrame(forest_dict)


# In[239]:


plt.figure(12, figsize=(20,10))
plt.subplot(121)
sns.pointplot(forest_dict.max_depth, forest_dict['R-squared'], forest_dict.n_estimators)
plt.subplot(122)
sns.pointplot(forest_dict.max_depth, forest_dict['RMSE'], forest_dict.n_estimators)


# In[240]:


param = {'n_estimators':[1500], 'max_depth':[100]}
forest = GridSearchCV(RandomForestRegressor(), param_grid=param, cv=5, n_jobs=4)
forest.fit(X_train, y_train)


# In[241]:


forest_r2 = forest.score(X_test, y_test)
print('R-squared: ', forest_r2)


# In[242]:


y_pred = forest.predict(X_test)
forest_rmse = np.sqrt(MSE(y_pred=y_pred, y_true=y_test))
print('RMSE: ', forest_rmse)


# In[243]:


y_pred_train = forest.predict(X_train)


# In[244]:


plt.scatter(y_test,y_pred-y_test,
            c='limegreen',
            edgecolor='white',
            marker='s',
            s=35,
            alpha=0.6,
            label='Test Data')
plt.xlabel("Y_true")
plt.ylabel("Residuals")


# Summary:
# 
# The original data has 10, 000 rows and 55 columns, including 1 target column. The target column 'interest_rate' is not normally distributed. And in feature columns, there are some features that distribute extremely unbalanced, in which are small proportion of values that are far from the majority. Like `annual income` and `debt to income`, which is understandable that people having exceedingly high income or debt is a small proportion. But it's not appropriate to conclude that they are influential outliers to be removed. I used binning to reduce the influence of these points to the regression model. 
# 
# The features that are selected initially are basically based on correlation or importance, for the attributes of regression and tree models, and a large proportion of features are dropped. Though some interaction features are derived, there might be some important information not involved in current features. 
# 
# If there's more time, it is necessary to get a deeper understanding of all columns in the data. It might also be helpful by creating different feature set for regression and tree model. 

# 
