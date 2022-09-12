#ZILLOW PROJECT

#imports
import warnings
warnings.filterwarnings("ignore")
# tabular data stuff: numpy and pandas
import numpy as np
import pandas as pd
# data viz:
import matplotlib.pyplot as plt
import seaborn as sns

import env
import os

# imputer from sklearn
from sklearn.impute import SimpleImputer

from math import sqrt
from scipy import stats
from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

#sql query
''' 
    This query pulls data from the zillow database from SQL.
    If this has already been done, the function will just pull from the zillow.csv
    '''
sql = """
SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt,fips
FROM properties_2017
 LEFT JOIN
        predictions_2017 USING (parcelid)
        join
        propertylandusetype USING (propertylandusetypeid)
        WHERE propertylandusedesc = 'Single Family Residential'
        AND transactiondate BETWEEN '2017-01-01' AND '2017-12-31'
"""

#connection set ip
def conn(db, user=env.username, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


#make function to acquire from sql
def new_zillow_data():
    df = pd.read_sql(sql,conn("zillow"))
    return df

def get_zillow_data():
    if os.path.isfile("zillow.csv"):
        #if csv is present locally, pull it from there
        df = pd.read_csv("zillow.csv", index_col = 0)
    else:
        #if not locally found, run sql querry to pull data
        df = new_zillow_data()
        df.to_csv("zillow.csv")
    return df

#this function takes the zillow data frame and cleans for modeling
def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df
 
    
def clean_zillow_data(): 


    ''' this function is for acquiring the zillow data, drops nulls, removes outliers,converts the fips data to categorical, 
    changing fips data to county and renaming them to Ventura, Orange & Los Angeles counties.'''

    df = get_zillow_data()
    df = remove_outliers(df,1.5,['calculatedfinishedsquarefeet', 'taxvaluedollarcnt', 'yearbuilt'])
    
    
        #convert fips to categorical
    df["fips"] = pd.Categorical(df.fips) 
    df['fips'] = df['fips'].astype(str).apply(lambda x: x.replace('.0',''))
        #rename columns
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                          'bathroomcnt':'bathrooms', 
                          'calculatedfinishedsquarefeet':'area',
                          'taxvaluedollarcnt':'assessed_value', 
                          'yearbuilt':'year_built',
                          'fips':'county'})
    df['county'].replace("6111",'Ventura', inplace=True) 
    df['county'].replace("6059",'Orange', inplace=True)
    df['county'].replace("6037",'Los_Angeles', inplace=True)
    df.to_csv("clean_zillow.csv")
    return df


#Write function to scale data for zillow data
def scale_data(train, validate, test, features_to_scale):
    """Scales the 3 data splits using MinMax Scaler. 
    Takes in train, validate, and test data splits as well as a list of the features to scale. 
    Returns dataframe with scaled counterparts on as columns"""
    
    
    # Make the thing to train data only
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(train[features_to_scale])
    
    # Fit the thing with new column names with _scaled added on
    scaled_columns = [col+"_scaled" for col in features_to_scale]
    
    # Transform the separate datasets using the scaler learned from train
    scaled_train = scaler.transform(train[features_to_scale])
    scaled_validate = scaler.transform(validate[features_to_scale])
    scaled_test = scaler.transform(test[features_to_scale])
    
    # Apply the scaled data to the original unscaled data
    train_scaled = pd.concat([train, pd.DataFrame(scaled_train,index=train.index, columns = scaled_columns)],axis=1)
    validate_scaled = pd.concat([validate, pd.DataFrame(scaled_validate,index=validate.index, columns = scaled_columns)],axis=1)
    test_scaled = pd.concat([test, pd.DataFrame(scaled_test,index=test.index, columns = scaled_columns)],axis=1)

    return train_scaled, validate_scaled, test_scaled













 





