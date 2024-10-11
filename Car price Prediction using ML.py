import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn import metrics

###Data Collection and Processing###

# Load the dataset from a local file
dataset = 'Car Price Prediction Dataset.csv'  
df = pd.read_csv(dataset)

# Visualize the first 5 rows of the dataset 
print(df.head()) 

# Check and print the shape of the DataFrame
print("Shape of the dataset:", df.shape) 

# getting some information about the dataset
df.info()

# checking the number of missing values
print("Missing values in each Column:\n", df.isnull().sum())

# checking the distribution of categorical data
print("Fuel type distribution:\n", df['fueltype'].value_counts())
# Define features (X) and target (y)(spliting data)
X = df.drop(['CarName','price'],axis=1)
Y = df['price']
# Encode categorical features
X = pd.get_dummies(X, drop_first=True)
print(X)
print(Y)
# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=2)

# Train the model through(#Linear Regression#)
#linear regression model
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train,Y_train)

#prediction on Training data#
training_data_prediction = lin_reg_model.predict(X_train)
#R squared Error
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)
#Visualize the actu and pred car price
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Car Price")
plt.ylabel("Predicted Car Price")
plt.title(" Actual Car Prices vs Predicted Car Prices")
plt.show()

##prediction on Testing data##
testing_data_prediction = lin_reg_model.predict(X_test)
#R squared Error
error_score = metrics.r2_score(Y_test, testing_data_prediction)
print("R squared Error : ", error_score)
#Visualize the actu and pred car price
plt.scatter(Y_test, testing_data_prediction)
plt.xlabel("Actual Car Price")
plt.ylabel("Predicted Car Price")
plt.title(" Actual Car Prices vs Predicted Car Prices")
plt.show()


