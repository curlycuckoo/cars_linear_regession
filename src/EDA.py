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

# DBTITLE 1,EDA
cars.info()

# COMMAND ----------

cars.describe()

# COMMAND ----------

cars.head()

# COMMAND ----------

<p style = "font-size : 20px ; color: blue;font-family:TimesNewRoman"><b>Observations</b></p>

    
- Years is left skewed. Years ranges from 1996- 2019 . Age of cars 2 year old to 25 years old

- Kilometer driven , median is ~53k Km and mean is ~58K. Max values seems to be 6500000. This is very high , and seems to be outlier. Need to analyze further.

- Mileage is almost Normally distrubuited

- Engine is right skewed and has outliers on higher  and lower end

- Power and Price are also right skewed.

- Price 160 Lakh is too much for a used car. Seems to be an outlier.

# COMMAND ----------

plt.style.use('ggplot')
#select all quantitative columns for checking the spread
numeric_columns = cars.select_dtypes(include=np.number).columns.tolist()
plt.figure(figsize=(20,25))

for i, variable in enumerate(numeric_columns):
                     plt.subplot(10,3,i+1)
                       
                     sns.distplot(cars[variable],kde=False,color='blue')
                     plt.tight_layout()
                     plt.title(variable)

# COMMAND ----------

<p style = "font-size : 15px ; color: blue;font-family:TimesNewRoman">
    <b>Observations</b></p>
    
  
- Year is left skewed and has outilers on lower side., This column can be dropped
- Kilometer_driven is right skewed.
- Mileage is almost Normally distrubuted. Has few outliers on upper and lower side. need to check further.
- Engine ,power and price are  right skewed and has outliers on upper side.
- Age of car is right skewed.


# COMMAND ----------

cat_columns=['Location','Fuel_Type','Transmission', 'Owner_Type', 'Brand'] #cars.select_dtypes(exclude=np.number).columns.tolist()

plt.figure(figsize=(15,21))

for i, variable in enumerate(cat_columns):
                     plt.subplot(4,2,i+1)
                     order = cars[variable].value_counts(ascending=False).index    
                     ax=sns.countplot(x=cars[variable], data=cars , order=order ,palette='viridis')
                     for p in ax.patches:
                           percentage = '{:.1f}%'.format(100 * p.get_height()/len(cars[variable]))
                           x = p.get_x() + p.get_width() / 2 - 0.05
                           y = p.get_y() + p.get_height()
                           plt.annotate(percentage, (x, y),ha='center')
                     plt.xticks(rotation=90)
                     plt.tight_layout()
                     plt.title(variable)

# COMMAND ----------

<p style = "font-size : 20px ; color: blue;font-family:TimesNewRoman">
    <b>Observations</b></p>
   
   **Car Profile**
    
-  ~71 % cars available for sell have manual Transmission.
- ~82 % cars are First owned cars.
- ~39% of car available for sale are from  Maruti & Hyundai brands.
-  ~53% of car being sold/avialable for purchase  have fuel type as Diesel .
- Mumbai has highest numbers of car availabe for purchase whereas Ahmedabad has least
- Most of the cars are 5 seaters.
- Car being sold/available for purchase are in  2 - 23 years old
- ~ 71% car are lower price range car.

# COMMAND ----------

numeric_columns= numeric_columns = cars.select_dtypes(include=np.number).columns.tolist()
plt.figure(figsize=(13,17))

for i, variable in enumerate(numeric_columns):
                     plt.subplot(5,2,i+1)
                     sns.scatterplot(x=cars[variable],y=cars['Price']).set(title='Price vs '+ variable)
                     #plt.xticks(rotation=90)
                     plt.tight_layout()

# COMMAND ----------

# DBTITLE 1,Handling missing values
cars.isnull().sum()

# COMMAND ----------

# DBTITLE 1,Calculating missing values in each row
# counting the number of missing values per row
num_missing = cars.isnull().sum(axis=1)
num_missing.value_counts()

# COMMAND ----------

#Investigating how many missing values per row are there for each variable
for n in num_missing.value_counts().sort_index().index:
    if n > 0:
        print("*" *30,f'\nFor the rows with exactly {n} missing values, NAs are found in:')
        n_miss_per_col = cars[num_missing == n].isnull().sum()
        print(n_miss_per_col[n_miss_per_col > 0])
        print('\n\n')

# COMMAND ----------

This confirms that certain columns tend to be missing together or all nonmissing together. So will try to fill the missing values , as much as possible.

# COMMAND ----------

cars[num_missing==7]

# COMMAND ----------

col=['Engine','Power','Mileage']
cars[col].isnull().sum()

# COMMAND ----------

cars.groupby(['Name','Year'])['Engine'].median().head(30)

# COMMAND ----------

cars['Engine']=cars.groupby(['Name','Year'])['Engine'].apply(lambda x:x.fillna(x.median()))
cars['Power']=cars.groupby(['Name','Year'])['Power'].apply(lambda x:x.fillna(x.median()))
cars['Mileage']=cars.groupby(['Name','Year'])['Mileage'].apply(lambda x:x.fillna(x.median()))

# COMMAND ----------

col=['Engine','Power','Mileage']
cars[col].isnull().sum()

# COMMAND ----------

cars.groupby(['Brand','Model'])['Engine'].median().head(10)

# COMMAND ----------

#chosing Median to fill the the missing value as there are many outliers, 
#grouping by model and year to get  more granularity and more accurate Engine and then fillig with median
cars['Engine']=cars.groupby(['Brand','Model'])['Engine'].apply(lambda x:x.fillna(x.median()))


# COMMAND ----------

#chosing Median to fill the the missing value as there are many outliers, 
#grouping by model to get more granularity and more accurate Engine
cars['Power']=cars.groupby(['Brand','Model'])['Power'].apply(lambda x:x.fillna(x.median()))

# COMMAND ----------

#chosing Median to fill the the missing value as there are many outliers, 
#grouping by model to get more granularity and more accurate Engine
cars['Mileage']=cars.groupby(['Brand','Model'])['Mileage'].apply(lambda x:x.fillna(x.median()))


# COMMAND ----------

col=['Engine','Power','Mileage']
cars[col].isnull().sum()


# COMMAND ----------

cars.groupby(['Model','Year'])['Engine'].agg({'median','mean','max'}).sort_values(by='Model',ascending='True').head(10)

# COMMAND ----------

cars.groupby(['Brand','Engine'])['Power'].agg({'mean','median','max'}).head(10)

# COMMAND ----------

cars['Seats'].isnull().sum()

# COMMAND ----------

cars['Seats']=cars.groupby(['Name'])['Seats'].apply(lambda x:x.fillna(x.median()))

# COMMAND ----------

cars['Seats'].isnull().sum()

# COMMAND ----------

cars['Seats']=cars.groupby(['Model'])['Seats'].apply(lambda x:x.fillna(x.median()))

# COMMAND ----------

cars['Seats'].isnull().sum()

# COMMAND ----------

cars[cars['Seats'].isnull()==True].head(10)

# COMMAND ----------

#most of cars are 5 seater so fillrest of 23 by 5
cars['Seats']=cars['Seats'].fillna(5)

# COMMAND ----------

cars['Seats'].isnull().sum()

# COMMAND ----------


cars["Location"] = cars["Location"].astype("category")
cars['Brand'] =cars['Brand'].astype("category")

# COMMAND ----------

cars.info()

# COMMAND ----------

# DBTITLE 1,Processing New Price
#For better granualarity grouping has there would be same car model present so filling with a median value brings it more near to real value
cars['new_price_num']=cars.groupby(['Name','Year'])['new_price_num'].apply(lambda x:x.fillna(x.median()))

# COMMAND ----------

cars.new_price_num.isnull().sum()

# COMMAND ----------

cars['new_price_num']=cars.groupby(['Name'])['new_price_num'].apply(lambda x:x.fillna(x.median()))

# COMMAND ----------

cars.new_price_num.isnull().sum()

# COMMAND ----------

cars['new_price_num']=cars.groupby(['Brand','Model'])['new_price_num'].apply(lambda x:x.fillna(x.median()))

# COMMAND ----------

cars.new_price_num.isnull().sum()

# COMMAND ----------

cars['new_price_num']=cars.groupby(['Brand'])['new_price_num'].apply(lambda x:x.fillna(x.median()))

# COMMAND ----------

cars.drop(['New_Price'],axis=1,inplace=True)

# COMMAND ----------

cars.new_price_num.isnull().sum()

# COMMAND ----------

cars.groupby(['Brand'])['new_price_num'].median().sort_values(ascending=False)

# COMMAND ----------

cars.isnull().sum()

# COMMAND ----------


cols1 = ["Power","Mileage","Engine"]

for ii in cols1:
    cars[ii] = cars[ii].fillna(cars[ii].median())

# COMMAND ----------

#dropping remaining rows
#cannot further fill this rows so dropping them

cars.dropna(inplace=True,axis=0)

# COMMAND ----------

cars.isnull().sum()

# COMMAND ----------

cars.head()

# COMMAND ----------

cars.isnull().sum()

# COMMAND ----------

df.shape 

# COMMAND ----------

cars.groupby(['Brand'])['Price'].agg({'median','mean','max'})

# COMMAND ----------

#using business knowledge to create class 
Low=['Maruti', 
     'Hyundai',
     'Ambassdor',
     'Hindustan',
     'Force',
     'Chevrolet',
     'Fiat',
     'Tata',
     'Smart',
     'Renault',
     'Datsun',
     'Mahindra',
     'Skoda',
     'Ford',
     'Toyota',
     'Isuzu',
     'Mitsubishi','Honda']

High=['Audi',
      'Mini Cooper',
      'Bentley',
      'Mercedes-Benz',
      'Lamborghini',
      'Volkswagen',
      'Porsche',
      'Land Rover',
      'Nissan',
      'Volvo',
      'Jeep',
      'Jaguar',
      'BMW']# more than 30lakh

# COMMAND ----------

def classrange(x):
    if x in Low:
        return "Low"
    elif x in High:
        return "High"
    else: 
        return x

# COMMAND ----------

cars['Brand_Class'] = cars['Brand'].apply(lambda x: classrange(x))

# COMMAND ----------

cars['Brand_Class'].unique()

# COMMAND ----------

cars['Engine']=cars['Engine'].astype(int)
cars['Brand_Class']=cars["Brand_Class"].astype('category')

# COMMAND ----------

# DBTITLE 1,Bivariate & Multivariate Analysis
plt.figure(figsize=(10,8))
sns.heatmap(cars.corr(),annot=True ,cmap="YlGnBu" )
plt.show()

# COMMAND ----------

 Observations

    Engine has strong positive correlation to Power [0.86].
    Price has positive correlation to Engine[0.66] as well Power [0.77].
    Mileage is negative correlated to Engine,Power,Price.,Ageofcar
    Price has negative correlation to age of car.
    Kilometer driven doesnt impact Price



# COMMAND ----------

sns.pairplot(data=cars , corner=True)
plt.show()

# COMMAND ----------

 Observations

    Same observation about correlation as seen in heatmap.

    Kilometer driven doesnot have impact on Price .

    As power increase mileage decrease.

    Car with recent make sell at higher prices.

    Engine and Power increase , price of the car seems to increase.

Variables that are correlated with Price variable
Price Vs Engine Vs Transmission

# COMMAND ----------

# understand relation ship of Engine vs Price and Transmimssion
plt.figure(figsize=(10,7))

plt.title("Price VS Engine based on Transmission")
sns.scatterplot(y='Engine', x='Price', hue='Transmission', data=cars)

# COMMAND ----------

# DBTITLE 1,Price Vs Power vs Transmission
 #understand relationship betweem Price and Power
plt.figure(figsize=(10,7))
plt.title("Price vs Power based on Transmission")
sns.scatterplot(y='Power', x='Price', hue='Transmission', data=cars)

# COMMAND ----------

# Understand the relationships  between mileage and Price
sns.scatterplot(y='Mileage', x='Price', hue='Transmission', data=cars)

# COMMAND ----------

# DBTITLE 1,Price Vs Year Vs Transmission
# Impact of years on price 
plt.figure(figsize=(10,7))
plt.title("Price based on manufacturing Year of model")
sns.lineplot(x='Year', y='Price',hue='Transmission',
             data=cars)

# COMMAND ----------

# DBTITLE 1,Price Vs Year VS Fuel Type
# Impact of years on price 
plt.figure(figsize=(10,7))
plt.title("Price Vs Year VS FuelType")
sns.lineplot(x='Year', y='Price',hue='Fuel_Type',
             data=cars)

# COMMAND ----------

# DBTITLE 1,Year Vs Price Vs Owner_Type
plt.figure(figsize=(10,7))
plt.title("Price Vs Year VS Owner_Type")
sns.lineplot(x='Year', y='Price',hue='Owner_Type',
             data=cars)

# COMMAND ----------

cars[(cars["Owner_Type"]=='Third') & (cars["Year"].isin([2010]))].sort_values(by='Price',ascending =False)

# COMMAND ----------

cars.describe()

# COMMAND ----------

# DBTITLE 1,Price Vs Mileage vs Fuel_type
# Understand relationships  between price and mileage
plt.figure(figsize=(10,7))
plt.title("Price Vs Mileage")
sns.scatterplot(y='Price', x='Mileage', hue='Fuel_Type', data=cars)

# COMMAND ----------

# DBTITLE 1,Price Vs Seat
#Price and seats 
plt.figure(figsize=(20,15))
sns.set(font_scale=2)
sns.barplot(x='Seats', y='Price', data=cars)
plt.grid()

# COMMAND ----------

# DBTITLE 1,Price Vs Location
#Price and LOcation 
plt.figure(figsize=(20,15))
sns.set(font_scale=2)
sns.barplot(x='Location', y='Price', data=cars)
plt.grid()

# COMMAND ----------

# DBTITLE 1,Price Vs Brand
#Price and band 
plt.figure(figsize=(20,15))
sns.set(font_scale=2)
sns.boxplot(x='Price', y='Brand', data=cars)
plt.grid()

# COMMAND ----------

sns.relplot(data=cars, y='Price',x='Mileage',hue='Transmission',aspect=1,height=5)

# COMMAND ----------

sns.relplot(data=cars, y='Price',x='Year',col='Owner_Type',hue='Transmission',aspect=1,height=5)

# COMMAND ----------

sns.relplot(data=cars, y='Price',x='Engine',col='Transmission',aspect=1,height=6,hue="Fuel_Type")

# COMMAND ----------

sns.relplot(data=cars, y='Price',x='Ageofcar',col='Transmission',aspect=1,height=6)

# COMMAND ----------

# DBTITLE 1,Insights based on EDA
 Observations

    Expensive cars are in Coimbatore and Banglore.
    2 Seater cars are more expensive.
    Deisel Fuel type car are more expensive compared to other fuel type.
    As expected, Older model are sold cheaper compared to latest model
    Automatic transmission vehicle have a higher price than manual transmission vehicles.
    Vehicles with more engine capacity have higher prices.
    Price decreases as number of owner increases.
    Automatic transmission require high engine and power.
    Prices for Cars with fuel type as Deisel has increased with recent models
    Engine,Power, how old the car his, Mileage,Fuel type,location,Transmission effect the price.



# COMMAND ----------

# check distrubution if skewed. If distrubution is skewed , it is advice to use log transform
cols_to_log = cars.select_dtypes(include=np.number).columns.tolist()
for colname in cols_to_log:
    sns.distplot(cars[colname], kde=True)
    plt.show()

# COMMAND ----------

# DBTITLE 1,Distrubtions are right skewed , using Log transform can help in normalization
def Perform_log_transform(df,col_log):
    """#Perform Log Transformation of dataframe , and list of columns """
    for colname in col_log:
        df[colname + '_log'] = np.log(df[colname])
    #df.drop(col_log, axis=1, inplace=True)
    df.info()

# COMMAND ----------

#This needs to be done before the data is split
Perform_log_transform(cars,['Kilometers_Driven','Price'])

# COMMAND ----------


cars.drop(['Name','Model','Year','Brand','new_price_num'],axis=1,inplace=True)

# COMMAND ----------

cars.info()

# COMMAND ----------

cars.to_csv('cars_final_model.csv')

# COMMAND ----------

cars.shape
