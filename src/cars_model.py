# Databricks notebook source
### IMPORT: ------------------------------------
import scipy.stats as stats 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') # To supress warnings
 # set the background for the graphs
from scipy.stats import skew
plt.style.use('ggplot')
import missingno as msno # to get visualization on missing values
from sklearn.model_selection import train_test_split # Sklearn package's randomized data splitting function
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_rows', 300)
pd.set_option('display.max_colwidth',400)
pd.set_option('display.float_format', lambda x: '%.5f' % x) # To supress numerical display in scientific notations
import statsmodels.api as sm
print("Load Libraries- Done")

# COMMAND ----------

import mlflow.sklearn
import mlflow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
import os
import shutil
import pprint
import tempfile
from random import random, randint
from sklearn import metrics

# COMMAND ----------

import pandas as pd
cars = pd.read_csv('/dbfs/FileStore/shared_uploads/blasa.matthew@yahoo.com/cars_final_model.csv')

# COMMAND ----------

X = cars.drop(["Price", "Price_log"], axis=1)
y = cars[["Price_log", "Price"]]

# COMMAND ----------

cars.head()

# COMMAND ----------

# DBTITLE 1, Creating dummy variables
def encode_cat_vars(x):
    x = pd.get_dummies(
        x,
        columns=x.select_dtypes(include=["object", "category"]).columns.tolist(),
        drop_first=True,
    )
    return x

# COMMAND ----------

#Dummy variable creation is done before spliting the data , so all the different categories are covered
#create dummy variable
X = encode_cat_vars(X)
X.head()

# COMMAND ----------

# DBTITLE 1,Test Train Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train.reset_index()
print("X_train:",X_train.shape)
print("X_test:",X_test.shape)
print("y_train:",y_train.shape)
print("y_test:",y_test.shape)

# COMMAND ----------

# Statsmodel api does not add a constant by default. We need to add it explicitly.
X_train = sm.add_constant(X_train)
# Add constant to test data
X_test = sm.add_constant(X_test)


def build_ols_model(train):
    # Create the model
    olsmodel = sm.OLS(y_train["Price_log"], train)
    return olsmodel.fit()

# COMMAND ----------

# DBTITLE 1,Declare Experiment Name

experiment_name = "/Users/cuckoodu4@hotmail.com/cars_bugu"
mlflow.set_experiment(experiment_name)

#add connection to current node 

# COMMAND ----------

#Experiment ID needs to be explicitly declared 
with mlflow.start_run(run_name='OLS_Regression') as run:
        # Get the run and experimentid
        run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id
        
        params = None
        
        # Create linear regression object
        regr = linear_model.LinearRegression() if params is None else linear_model.LinearRegression(**params)

        # Train the model using the training sets
        regr.fit(X_train, y_train)

        # Make predictions using the testing set
        y_pred = regr.predict(X_test)
        
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rsme = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # The coefficient of determination: 1 is perfect prediction
        print(f'Coefficients: {regr.coef_}')
        print(f'Mean squared error: {mse}')
        print(f'Root mean square error: {rsme}')
        print(f'R2 score = {r2}')
     
       
        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rsme)
        mlflow.log_metric("r2_score", r2)
                      
        # Log the model
        mlflow.sklearn.log_model(
            sk_model=regr,
            artifact_path="sklearn-model")

# COMMAND ----------



# COMMAND ----------

def model_pref(olsmodel, x_train, x_test):

    # Insample Prediction
    y_pred_train_pricelog = olsmodel.predict(x_train)
    y_pred_train_Price = y_pred_train_pricelog.apply(math.exp)
    y_train_Price = y_train["Price"]

    # Prediction on test data
    y_pred_test_pricelog = olsmodel.predict(x_test)
    y_pred_test_Price = y_pred_test_pricelog.apply(math.exp)
    y_test_Price = y_test["Price"]

    print(
        pd.DataFrame(
            {
                "Data": ["Train", "Test"],
                "RMSE": [
                    rmse(y_pred_train_Price, y_train_Price),
                    rmse(y_pred_test_Price, y_test_Price),
                ],
                "MAE": [
                    mae(y_pred_train_Price, y_train_Price),
                    mae(y_pred_test_Price, y_test_Price),
                ],
                "MAPE": [
                    mape(y_pred_train_Price, y_train_Price),
                    mape(y_pred_test_Price, y_test_Price),
                ],
            }
        )
    )



# COMMAND ----------

# MAGIC %md 
# MAGIC # Test Assumptions

# COMMAND ----------

# DBTITLE 1,Assumption #1: No Multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor


def checking_vif(train):
    vif = pd.DataFrame()
    vif["feature"] = train.columns

    # calculating VIF for each feature
    vif["VIF"] = [
        variance_inflation_factor(train.values, i) for i in range(len(train.columns))
    ]
    return vif


# COMMAND ----------

vif

# COMMAND ----------

# Check VIF
print(checking_vif(X_train))

# COMMAND ----------

X_train1=X_train.drop(['Engine'],axis=1)
X_test1=X_test.drop(['Engine'],axis=1)
olsmodel2= build_ols_model(X_train1)

print(olsmodel2.summary())

# Checking model performance
model_pref(olsmodel2, X_train1, X_test1)

# COMMAND ----------

print(checking_vif(X_train1))

# COMMAND ----------

# DBTITLE 1,Checking Assumption 2: Mean of residuals should be 0
residuals = olsmodel2.resid
np.mean(residuals)

# COMMAND ----------

# DBTITLE 1,Checking Assumption 3: No Heteroscedasticity
import statsmodels.stats.api as sms
from statsmodels.compat import lzip

name = ["F statistic", "p-value"]
test = sms.het_goldfeldquandt(residuals, X_train1)
lzip(name, test)

# COMMAND ----------

# DBTITLE 1,Checking Assumption 4: Linearity of variables
# predicted values
fitted = olsmodel2.fittedvalues

# sns.set_style("whitegrid")
sns.residplot(fitted, residuals, color="purple", lowess=True)
plt.xlabel("Fitted Values")
plt.ylabel("Residual")
plt.title("Residual PLOT")
plt.show()

# COMMAND ----------

sns.distplot(residuals)

# COMMAND ----------

# Plot q-q plot of residuals
import pylab
import scipy.stats as stats

stats.probplot(residuals, dist="norm", plot=pylab)
plt.show()

# COMMAND ----------


