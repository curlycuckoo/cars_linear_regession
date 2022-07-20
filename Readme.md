
![](https://res.cloudinary.com/makotoevo/image/upload/v1657384524/Copy_of_Blue_Yellow_Modern_Creative_Entrepreneur_LinkedIn_Banner_2000_600_px_nchgma.png)



## About This Project

The goal of this project was to create a predictive model which can estimate the price of cars in the Indian used car market. Data from this model will be used to optimize sale pricing and assist marketers target potential used car customers. 

The project will seek to create a multiple linear regression model for the data, using the most important features. The model will prune features down using Principal Component Analysis to select the most important features for the car pricing model, based off thresholds. Multiple linear regression will be used to estimate the pricing for used cars in India.

The problem to be solved in this project is to determine if a linear regression model could be created from a list of Indian used car prices. The goal was to determine whether a model could be created from the data. 


## Use Cases


* Which car models fetch the highest prices when resold? Which make and model are sold the most?
* How does car age affect the profitability of resale? 
* How does vehicle milage affect the price of the car resale?  
* Which older vehicles generate the most profit on resale? 

## Repository Structure
    .
    ├── input                           # Data used to train and test model
    │   ├── train.csv                       # model training set (Databricks)
    │   ├── test.csv                        # model testing set (Databricks)
    ├── notebooks                       # Jupyter Notebooks explaining model
    │   ├── cars_regression.ipynb           # Jupyter Notebook of process 
    ├── models                          # Model metadata, models, and model config
    │   ├── MLmodel                         # MLflow metadata 
    │   ├── conda.yaml                      # Mlflow model configuration
    │   ├── model.pkl                       # Pickled model 
    ├── src                             # Source files 
    │   ├── EDA.py                          # Exploratory Data Analysis in Databricks 
    │   ├── cars_data_pipeline.py           # Data Pipeline and transformation 
    │   ├── cars_model.py                   # Model Creation
    │   ├── train.py                        # Model Creation and Experiment Tracking 
    ├── workspace.code-workspace        # VS code workspace configuration
    ├── Dockerfile
    ├── requirements.txt              
    ├── LICENSE
    └── README.md

## Data Info

Data was taken from www.kaggle.com/datasets/saisaathvik/used-cars-dataset-from-cardekhocom

The data contains 14 columns:
* Row Number - row number of the data set  
* Name (Car Make and Model) -  Name of the car which includes Brand name and Model name
* Location - The location in which the car is being sold or is available for purchase Cities
* Year - Manufacturing year of the car
* KM Driven - The total kilometers driven in the car by the previous owner(s) in KM
* Fuel Type - The type of fuel used by the car. (Petrol, Diesel, Electric, CNG, LPG)
* Transmission Type - The type of transmission used by the car. (Automatic / Manual)
* Owner Type - Type of ownership
* Mileage - The standard mileage offered by the car company in kmpl or km/kg
* Engine - The displacement volume of the engine in CC.
* Max Power - The maximum power of the engine in bhp.
* Seats - The number of seats in the car.
* New Price - The price of a new car of the same model in INR Lakhs.(1 Lakh = 100, 000)
* Selling Price - The price of the used car in INR Lakhs (1 Lakh = 100, 000)

Each of the 14 columns have the following data types: 
* Row Number - String
* Name (Car Make and Model) -  String
* Location - String
* Year - String
* KM Driven - String
* Fuel Type - String
* Transmission String
* Owner Type - String
* Mileage - String
* Engine - String
* Max Power - String
* Seats - String
* New Price - String
* Selling Price - String


## Installation

This project is focused for the Azure Databricks environment. If you would like to use the python files and CSVs, please import them into you databricks environement. 

Install my project:

```bash
  npm install my-project
  cd my-project
```

## Modeling Process

![](https://res.cloudinary.com/makotoevo/image/upload/v1658276228/cars_regerssion_flowchart_nkelpx.png)

The modeling process is broken down into four parts. 

* CSV file is loaded into Azure Blob Storage. 
* The next stage is creating a data pipeline and finishing data transformation.
* The third stage is training the linear regression model.
* The final stage is exporting the model and experiments into mlflow.

## Limitations

Residual plots in the EDA show that cars manufactured after 2019 have increased exponentially in price. This is due to economic fallout from the ongoing COVID-19 pandemic and global inflation. The selling listed in Ladakh for in 2020 or 2021 cars may be similar in price to before 2019 without inflation. This greatly affects the data, leading to higher than normal prices for cars due to many factors such as supply chain issues, parts cost, etc. These increased costs can make a model difficult to deploy, given the many unseen factors going into the price to produce the car. Further investigation will be required to determine the exact effects of inflation. 

Another limitation is the date the car was sold is unknown. We also do not know when the car was bought on Cardeko. We assumed that the cars were bought in 2022, which is likely skewing the data. The only year that we have is the year in which it was manufactured. This may result in data leakage, where future and past data of sales dates are being mixed together. If we can figure out what year the car was purchased, then we can segment the data by year to create an better performing model and reduce this. 

The last limitation is how the data was scraped. This is important, since web scraping relies heavily on pulling data from specific elements on the website. (Lak). If the Cardekho website underwent major website structure redesign of its JavaScript or HTML elements, then data might be missed. This could result in large amounts of data missing at random, due to this difference. This could mean that essential features may change overtime. Data scrapped on one occasion might be complete different from another time. Even if that data is the same car that got sold. This can cause major data integrity issues. These data integrity issues can lead to training data and machine learning models that are different each time.

  

## Recommendations

There are two future approaches that a data science team can take towards this data set. The first is getting clear use cases from business users to determine what sort of machine learning model(s) should be created to address the users needs. The second approach is ensuring quality data from those requirements, and getting all features to create a useable model. These would both would deliver value in the long run. 

Clear use cases are important here. When creating the model, it was unclear who the end users would be. A machine learning model should be specialized to address not only users needs, but to serve as a foundation for future predictive models (Zheng 76). It is important to create a foundation for business users. Assuming that data being used in the model is correct, then it might be important to specialize. This can mean individual models designed to address sales, marketing, or any other departments needs. By specializing to each clear use case for a department, greater value can be delivered across different business verticals. 

Defining good data quality is another approach. Data quality comes from use cases. Data quality is the condition of data: accuracy, completeness, consistency, reliability and whether it's up to date. Business users end use cases determine whether the data fulfill this. Specifically, for each model the data needs to be documented and scoped. This would include all the columns that would be needed to build the model, which was the problem during model building. 

Good preparation is necessary to create a model to deliver value from this dataset. Without it, no business value can be gained or generated.


## Follow Me:
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/du-juan/)
