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

# COMMAND ----------

# DBTITLE 1,Read and Understand data
import pandas as pd
df = pd.read_csv('/dbfs/FileStore/shared_uploads/blasa.matthew@yahoo.com/used_cars_data_main.csv')

# COMMAND ----------

df.info()

# COMMAND ----------

df.head()

# COMMAND ----------

# DBTITLE 1,Data Type Conversion
def convert_string(col_name):
  df[col_name]=df[col_name].astype("string")
  #return df[col_name]

# COMMAND ----------

df['Name']=df['Name'].astype("string")

# COMMAND ----------

convert_string('Location')

# COMMAND ----------

convert_string('Fuel_Type')

# COMMAND ----------

convert_string('Transmission')

# COMMAND ----------

convert_string('Owner_Type')

# COMMAND ----------

convert_string('Mileage')

# COMMAND ----------

convert_string('Engine')

# COMMAND ----------

convert_string('Power')

# COMMAND ----------

#Reading the csv file  used car data.csv 
cars=df.copy()
print(f'There are {cars.shape[0]} rows and {cars.shape[1]} columns') # fstring 

# COMMAND ----------

# inspect data, print top 5 
cars.head(5)

# COMMAND ----------

# bottom 5 rows:
cars.tail(5)

# COMMAND ----------

#get the size of dataframe
print ("Rows     : " , cars.shape[0])  #get number of rows/observations
print ("Columns  : " , cars.shape[1]) #get number of columns
print ("#"*40,"\n","Features : \n\n", cars.columns.tolist()) #get name of columns/features
print ("#"*40,"\nMissing values :\n\n", cars.isnull().sum().sort_values(ascending=False))
print( "#"*40,"\nPercent of missing :\n\n", round(cars.isna().sum() / cars.isna().count() * 100, 2)) # looking at columns with most Missing Values
print ("#"*40,"\nUnique values :  \n\n", cars.nunique())  #  count of unique values

# COMMAND ----------

cars.info()

# COMMAND ----------

# Making a list of all categorical variables
cat_col = [
    "Fuel_Type",
    "Location",
    "Transmission",
    "Seats",
    "Year",
    "Owner_Type",
    
]
# Printing number of count of each unique value in each column
for column in cat_col:
    print(cars[column].value_counts())
    print("#" * 40)

# COMMAND ----------

# DBTITLE 1,Data Preprocessing
# MAGIC %md
# MAGIC ### Processing Engine,Power ,Mileage columns

# COMMAND ----------

# MAGIC %md
# MAGIC Datatype for Engine ,Power and Mileage  are object because of unit assigned ,so striping  units.

# COMMAND ----------

#np.random.seed(9)
cars[['Engine','Power','Mileage']].sample(10)

# COMMAND ----------

typeoffuel=['CNG','LPG']
cars.loc[cars.Fuel_Type.isin(typeoffuel)].head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Power has some values as "nullbhp" .Mileage also has some observations as 0. For fuel type and CNG and LPG mileage is measured in km/kg where as for other type it is measured in kmpl. Since  those units are in  km for both of them no need of conversion . Dropping units from mileages,Engine and Power.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Mileage

# COMMAND ----------

cars[cars.Mileage.isnull()==True]

# COMMAND ----------

cars = cars.dropna(subset=['Mileage'])

# COMMAND ----------

cars['Mileage'].isna().sum()

# COMMAND ----------

cars["Mileage"].isna().sum()

# COMMAND ----------

cars["Mileage"] = cars["Mileage"].str.rstrip(" kmpl")
cars["Mileage"] = cars["Mileage"].str.rstrip(" km/g")


# COMMAND ----------

cars.info()

# COMMAND ----------

cars.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Engine 

# COMMAND ----------

#remove units
cars["Engine"] = cars["Engine"].str.rstrip(" CC")

# COMMAND ----------

cars["Engine"].isna().sum()

# COMMAND ----------

cars = cars.dropna(subset=["Engine"])

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Power

# COMMAND ----------

#remove bhp and replace null with nan
cars["Power"] = cars["Power"].str.rstrip(" bhp")
cars["Power"]= cars["Power"].replace(regex="null", value = np.nan)

# COMMAND ----------

cars["Power"].isna().sum()

# COMMAND ----------

cars = cars.dropna(subset=["Power"])

# COMMAND ----------

#verify the data
num=['Engine','Power','Mileage']
cars[num].sample(20)

# COMMAND ----------

# MAGIC %md
# MAGIC I had seen some values in Power and Mileage as 0.0 so verifying data for Engine, Power, Mileage. Will check once again after converting datatype

# COMMAND ----------

cars.query("Power == '0.0'")['Power'].count()

# COMMAND ----------

cars.query("Mileage == '0.0'")['Mileage'].count()


# COMMAND ----------

# MAGIC %md
# MAGIC Converting this observations to Nan so we will remember to handle them when handling missing values.

# COMMAND ----------

cars.loc[cars["Mileage"]=='0.0','Mileage']=np.nan

# COMMAND ----------

cars.loc[cars["Engine"]=='0.0','Engine'].count()

# COMMAND ----------

cars[num].nunique()

# COMMAND ----------

cars[num].isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ###  Processing Seats

# COMMAND ----------

cars.query("Seats == 0.0")['Seats']

# COMMAND ----------

#seats cannot be 0 so changing it to nan and will be handled in missing value
cars.loc[3999,'Seats'] =np.nan

# COMMAND ----------

# MAGIC %md
# MAGIC ###  Processing  New Price
# MAGIC We know that `New_Price` is the price of a new car of the same model in INR Lakhs.(1 Lakh = 100, 000)
# MAGIC 
# MAGIC This column clearly has a lot of missing values. We will impute the missing values later. For now we will only extract the numeric values from this column.

# COMMAND ----------

cars.head()

# COMMAND ----------

# Create a new column after splitting the New_Price values.
import re

new_price_num = []

# Regex for numeric + " " + "Lakh"  format
regex_power = "^\d+(\.\d+)? Lakh$"

for observation in df["New_Price"]:
    if isinstance(observation, str):
        if re.match(regex_power, observation):
            new_price_num.append(float(observation.split(" ")[0]))
        else:
            # To detect if there are any observations in the column that do not follow [numeric + " " + "Lakh"]  format
            # that we see in the sample output
            print(
                "The data needs furthur processing.mismatch ",
                observation,
            )
    else:
        # If there are any missing values in the New_Price column, we add missing values to the new column
        new_price_num.append(np.nan)

# COMMAND ----------

cars.head()

# COMMAND ----------

# DBTITLE 1,Feature Enginering
cars.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## converting datatype

# COMMAND ----------

cars.info()

# COMMAND ----------

cars["Name"]=cars["Name"].astype("string")

# COMMAND ----------

#converting object data type to category data type
cars["Fuel_Type"] = cars["Fuel_Type"].astype("category")
cars["Transmission"] = cars["Transmission"].astype("category")
cars["Owner_Type"] = cars["Owner_Type"].astype("category")
#converting datatype  

cars = cars.dropna(subset=["Mileage"])

cars["Mileage"] = cars["Mileage"].astype(float)
cars["Power"] = cars["Power"].astype(float)
cars["Engine"]=cars["Engine"].astype(float)

# COMMAND ----------

cars.info()

# COMMAND ----------

cars.describe().T

# COMMAND ----------

# MAGIC %md
# MAGIC ### Processing Years to Derive Age of car
# MAGIC Since year has 2014, 1996  etc. But this will not help to understand how old cars is and its effect on  price.
# MAGIC so creating  two new columns current year and Age . Current year would be 2021 and Age column would be Ageofcar= currentyear-year. And then drop currentyear columns

# COMMAND ----------

cars['Current_year']=2021
cars['Ageofcar']=cars['Current_year']-cars['Year']
cars.drop('Current_year',axis=1,inplace=True)
cars.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Processing Name column

# COMMAND ----------

# MAGIC %md
# MAGIC Brands do play an important role in Car selection and Prices. So extracting brand names from the Name.

# COMMAND ----------

cars2 = cars.copy()

# COMMAND ----------

#dropping rows with name as null
cars2 = cars2.dropna(subset=['Name'])

# COMMAND ----------

cars2.head()

# COMMAND ----------

cars2['Name'].dtype

# COMMAND ----------

cars.info()

# COMMAND ----------

cars['Brand'] = cars['Name'].astype(str)

# COMMAND ----------

cars2.head()

# COMMAND ----------

def get_name(car):
    return car.split(" ")[0]
def get_model(car):
    return car.split(" ")[1]

# COMMAND ----------

cars2['car_name'] = cars2['Name'].apply(lambda x: get_name(x)+' '+get_model(x))

# COMMAND ----------

cars2.head()

# COMMAND ----------

cars2['car_brand'] = cars2['car_name'].str.split(' ').str[0] #Separating Brand name from the Name
cars2['model'] = cars2['car_name'].str.split(' ').str[1]

# COMMAND ----------

cars2.head()

# COMMAND ----------

# DBTITLE 1,Standardize Car Brand Names
cars2.car_brand.unique()

# COMMAND ----------

cars2.loc[cars2.car_brand == 'ISUZU','car_brand']='Isuzu'
cars2.loc[cars2.car_brand=='Mini','car_brand']='Mini Cooper'
cars2.loc[cars2.car_brand=='Land','car_brand']='Land Rover'
cars2.loc[cars2.car_brand=='Mercedes-AMG','car_brand']='Mercedes-Benz'
cars2.loc[cars2.car_brand=='OpelCorsa','car_brand']='Opel'


# COMMAND ----------

# MAGIC %md
# MAGIC Brand names like ISUZU and Isuzu are same and needs to be corrected. Land, Mini seems to be incorrect. So correcting brand names.

# COMMAND ----------

cars2.info()

# COMMAND ----------

#changing brandnames
cars.loc[cars.Brand == 'ISUZU','Brand']='Isuzu'
cars.loc[cars.Brand=='Mini','Brand']='Mini Cooper'
cars.loc[cars.Brand=='Land','Brand']='Land Rover'
#cars['Brand']=cars["Brand"].astype("category")

# COMMAND ----------

cars2.model.isnull().sum()

# COMMAND ----------

cars2.groupby(cars.Brand).size().sort_values(ascending =False)

# COMMAND ----------

cars2.model.isnull().sum()

# COMMAND ----------

cars3=cars2.copy()

# COMMAND ----------

#drop row with no model
cars3.dropna(subset=['model'],axis=0,inplace=True)

# COMMAND ----------

cars3.model.isnull().sum()

# COMMAND ----------

cars3.model.nunique()

# COMMAND ----------

cars3.groupby('model')['model'].size().nlargest(100)

# COMMAND ----------

cars3.drop(['New_Price'], axis=1,inplace=True)

# COMMAND ----------

cars3.head()

# COMMAND ----------


