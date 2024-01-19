#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Needed libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


# In[23]:


# Loading dataset
data = pd.read_csv("/Users/khushitaneja/Downloads/cc_approvals.csv", header=None)
data.head()


# In[ ]:





# In[24]:


# Spliting data into train and test sets
data_train, data_test = train_test_split(data, test_size=0.33, random_state=42)


# In[25]:


# Replacing '?' with null values in data
data_train_nans_replaced = data_train.replace("?", np.NaN)
data_test_nans_replaced = data_test.replace("?", np.NaN)


# In[26]:


# Imputing the numerical null values with mean
data_train_imputed = data_train_nans_replaced.fillna(data_train_nans_replaced.mean(numeric_only=True))
data_test_imputed = data_test_nans_replaced.fillna(data_test_nans_replaced.mean(numeric_only=True))


# In[27]:


# Iterate over each column of data_train_imputed
for col in data_train_imputed.columns:
    # Check if the column is non-numeric - to impute missing values in a different way
    if data_train_imputed[col].dtypes == "object":
        # Impute with the most frequent value
        data_train_imputed = data_train_imputed.fillna(
            data_train_imputed[col].value_counts().index[0]
        )
        data_test_imputed = data_test_imputed.fillna(
            data_test_imputed[col].value_counts().index[0]
        )


# In[28]:


# Converting the categorical features in the train and test sets to get dummy variables
cat_data_train_encoding = pd.get_dummies(data_train_imputed)
cat_data_test_encoding = pd.get_dummies(data_test_imputed)


# In[29]:


# aligning the index of the test set with the train set
cat_data_test_encoding = cat_data_test_encoding.reindex(
    columns=cat_data_train_encoding.columns, fill_value=0
)


# In[30]:


# Segregate features and labels into separate variables
X_train, y_train = (
    cat_data_train_encoding.iloc[:, :-1].values,
    cat_data_train_encoding.iloc[:, [-1]].values,
)
X_test, y_test = (
    cat_data_test_encoding.iloc[:, :-1].values,
    cat_data_test_encoding.iloc[:, [-1]].values,
)


# In[31]:


# Scaling features using min-max scaling
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.transform(X_test)


# In[32]:


# Instantiating a LogisticRegression classifier with default parameter values and then fitting it
logreg = LogisticRegression()
logreg.fit(rescaledX_train, y_train)

# Use logreg to predict test values
y_pred = logreg.predict(rescaledX_test)

# Checking the confusion matrix to see the model performance
print(confusion_matrix(y_test, y_pred))


# In[ ]:


# Confusion matrix shows 0 miscalculations


# In[ ]:




