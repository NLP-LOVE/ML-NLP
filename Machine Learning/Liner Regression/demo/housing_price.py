# Compatible with Python 2, 3
from __future__ import print_function

# Import libraries
import os
import numpy as np
import pandas as pd

# Set random seed
np.random.seed(36)

# Use matplotlib for plotting
import matplotlib
import seaborn
import matplotlib.pyplot as plot

from sklearn import datasets


# Load data
housing = pd.read_csv('kc_train.csv')
target=pd.read_csv('kc_train2.csv')  # Sale price
t=pd.read_csv('kc_test.csv')         # Test data

# Data preprocessing
housing.info()    # Check for missing values

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
minmax_scaler=MinMaxScaler()
minmax_scaler.fit(housing)   # Fit internally, parameters will change
scaler_housing=minmax_scaler.transform(housing)
scaler_housing=pd.DataFrame(scaler_housing,columns=housing.columns)

mm=MinMaxScaler()
mm.fit(t)
scaler_t=mm.transform(t)
scaler_t=pd.DataFrame(scaler_t,columns=t.columns)



# Linear regression model with gradient descent
from sklearn.linear_model import LinearRegression
LR_reg=LinearRegression()
# Fit model
LR_reg.fit(scaler_housing,target)


# Use MSE to evaluate model
from sklearn.metrics import mean_squared_error
preds=LR_reg.predict(scaler_housing)   # Predict on input data
mse=mean_squared_error(preds,target)   # Evaluate model with MSE

# Plot comparison
plot.figure(figsize=(10,7))       # Figure size
num=100
x=np.arange(1,num+1)              # Take 100 points for comparison
plot.plot(x,target[:num],label='target')      # Target values
plot.plot(x,preds[:num],label='preds')        # Predicted values
plot.legend(loc='upper right')  # Legend position
plot.show()


# Output test predictions
result=LR_reg.predict(scaler_t)
df_result=pd.DataFrame(result)
df_result.to_csv("result.csv")
