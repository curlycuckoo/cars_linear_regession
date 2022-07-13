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


def import_data():
  
    '''
    This function loads the used cars csv from the databricks cluster to create a pandas dataframe 
    
    Parameters: None 
    
    Returns: 
    Dataframe (object) of csv in the file path as a Pandas Dataframe 
    '''
    
    df = pd.read_csv('/dbfs/FileStore/shared_uploads/blasa.matthew@yahoo.com/cars_final_model.csv')
    return df

# COMMAND ----------

experiment_name = "/Users/cuckoodu4@hotmail.com/cars_bugu"
mlflow.set_experiment(experiment_name)

# COMMAND ----------

data = import_data()

# COMMAND ----------

X = data.drop(["Price", "Price_log"], axis=1)
y = data[["Price_log", "Price"]]

# COMMAND ----------

def encode_cat_vars(x):
    x = pd.get_dummies(
        x,
        columns=x.select_dtypes(include=["object", "category"]).columns.tolist(),
        drop_first=True,
    )
    return x

# COMMAND ----------

X = encode_cat_vars(X)
X.head()

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train.reset_index()

# COMMAND ----------

X_train = sm.add_constant(X_train)
# Add constant to test data
X_test = sm.add_constant(X_test)


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


