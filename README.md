Required Libraries:

pandas, numpy, matplotlib.pyplot, and seaborn for data manipulation, mathematical operations, and visualization.
train_test_split from sklearn.model_selection to split the dataset.
LinearRegression from sklearn.linear_model for building a regression model.
StandardScaler from sklearn.preprocessing for normalizing features.
metrics from sklearn to evaluate model performance.

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

Loading the Dataset:

dataset = 'Car Price Prediction Dataset.csv'  
df = pd.read_csv(dataset) 
print(df.head()) 
print("Shape of the dataset:", df.shape) 
df.info()
print("Missing values in each Column:\n", df.isnull().sum())

You are reading a dataset from a CSV file using pandas.read_csv().
The first five rows of the dataset are printed for a quick preview of the data.
The dataset’s shape and data types are checked using df.info() to see which features are numeric or categorical.
Missing values in each column are checked with df.isnull().sum().

Exploring the Data:

print("Fuel type distribution:\n", df['fueltype'].value_counts())
X = df.drop(['CarName','price'],axis=1)
Y = df['price']
X = pd.get_dummies(X, drop_first=True)
print(X)
print(Y)

The distribution of the fueltype column is checked using df['fueltype'].value_counts().
Features (X) and target (Y) are defined. The CarName and price columns are dropped from the feature set, as they are not needed for training.
Categorical variables are encoded using one-hot encoding (pd.get_dummies()), and drop_first=True removes one category from each to avoid multicollinearity.

Splitting the Data:

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=2)

The dataset is split into training and testing sets using train_test_split() with 10% of the data used for testing.

Scaling the Data:

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train,Y_train)

StandardScaler is used to scale the features for better model performance.

Training with Linear Regression Model:

training_data_prediction = lin_reg_model.predict(X_train)
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)

The model is trained on the scaled training data using LinearRegression().fit().

Evaluating the Model:

Predictions on the training data are made using predict().
The R-squared score (metrics.r2_score()) is calculated to measure the model’s goodness-of-fit.
A scatter plot visualizes the relationship between the actual and predicted car prices.

plt.scatter(Y_train, training_data_prediction)

plt.xlabel("Actual Car Price")

plt.ylabel("Predicted Car Price")

plt.title(" Actual Car Prices vs Predicted Car Prices")

plt.show()

Testing the Model:

testing_data_prediction = lin_reg_model.predict(X_test)

error_score = metrics.r2_score(Y_test, testing_data_prediction)

print("R squared Error : ", error_score)

plt.scatter(Y_test, testing_data_prediction)

plt.xlabel("Actual Car Price")

plt.ylabel("Predicted Car Price")

plt.title(" Actual Car Prices vs Predicted Car Prices")

plt.show()

The model is tested on the testing data, and similar steps are followed to visualize the performance.
