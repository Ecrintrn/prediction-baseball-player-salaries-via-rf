# Prediction Baseball Player salaries via Random Forest

# Business Problem
In this section we are planing to predict the salaries of the baseball players by specific parameters and referred parameters described as:

# Dataset Story

This dataset was originally taken from the StatLib library which is maintained at Carnegie Mellon University. This is part of the data that was used in the 1988 ASA Graphics Section Poster Session. The salary data were originally from Sports Illustrated, April 20, 1987. The 1986 and career statistics were obtained from The 1987 Baseball Encyclopedia Update published by Collier Books, Macmillan Publishing Company, New York.

* AtBat Number of times at bat in 1986
* Hits Number of hits in 1986
* HmRun Number of home runs in 1986
* Runs Number of runs in 1986
* RBI Number of runs batted in in 1986
* Walks Number of walks in 1986
* Years Number of years in the major leagues
* CAtBat Number of times at bat during his career
* CHits Number of hits during his career
* CHmRun Number of home runs during his career
* CRuns Number of runs during his career
* CRBI Number of runs batted in during his career
* CWalks Number of walks during his career
* League A factor with levels A and N indicating player’s league at the end of 1986
* Division A factor with levels E and W indicating player’s division at the end of 1986
* PutOuts Number of put outs in 1986
* Assists Number of assists in 1986
* Errors Number of errors in 1986
* Salary 1987 annual salary on opening day in thousands of dollars
* NewLeague A factor with levels A and N indicating player’s league at the beginning of 1987

# Import Necessary Libraries

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

import warnings
warnings.filterwarnings('ignore')
```

# Import Dataset
```
hitters = pd.read_csv('/kaggle/input/hitters/Hitters.csv')
df = hitters.copy()
df.head()
```

# General Information About the Dataset
## Checking the Data Frame
Since we want to check the data to get a general idea about it, we create and use a function called check_df(dataframe, head=5, tail=5) that prints the referenced functions:

```
print(20*"#","HEAD",20*"#")
print(dataframe.head(head))
print(20*"#","Tail",20*"#")
print(dataframe.tail(head))
print(20*"#","Shape",20*"#")
print(dataframe.shape)
print(20*"#","Types",20*"#")
print(dataframe.dtypes)
print(20*"#","NA",20*"#")
print(dataframe.isnull().sum().sum())
print(dataframe.isnull().sum())
print(20*"#","Quartiles",20*"#")
print(dataframe.describe([0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]).T)
```

## Analysis of Categorical and Numerical Variables
```
cat_cols = [col for col in datraframe.columns if str(datraframe[col].dtypes) in ["category", "object", "bool"]]
num_but_cat = [col for col in datraframe.columns if datraframe[col].nunique()< cat_th and datraframe[col].dtypes in ["uint8", "int64", "float64"]]
cat_but_car = [col for col in datraframe.columns if datraframe[col].nunique() > car_th and str(datraframe[col].dtypes) in ["category", "object"]]
cat_cols = cat_cols + num_but_cat
num_cols= [col for col in datraframe.columns if datraframe[col].dtypes in ["uint8", "int64", "float64"]]
num_cols = [col for col in num_cols if col not in cat_cols]
```

We create another plot function called plot_num_summary(dataframe) to see the whole summary of numerical columns due to the high quantity of them:

![plot_num](https://github.com/user-attachments/assets/9b9bf897-f9de-42d1-82b2-d20f7fc2e2aa)

# Correlation Analysis

To analyze correlations between numerical columns we create a function called correlated_cols(dataframe):

![correlation](https://github.com/user-attachments/assets/1007480d-1f79-4536-94c8-cae74f3f4403)

# High Correlation Analysis


# Missing Value Analysis

We check the data to designate the missing values in it, dataframe.isnull().sum():

AtBat         0
Hits          0
HmRun         0
Runs          0
RBI           0
Walks         0
Years         0
CAtBat        0
CHits         0
CHmRun        0
CRuns         0
CRBI          0
CWalks        0
League        0
Division      0
PutOuts       0
Assists       0
Errors        0
Salary       59
NewLeague     0
dtype: int64
