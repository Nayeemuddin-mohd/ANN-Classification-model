# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:50:30 2019

@author: nayeemuddin.mohd
"""
# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
data=pd.read_csv("BankCustomers.csv")
x=data.iloc[:,3:13]
y=data.iloc[:,13]

# converting categorical feature into dummy variables
states=pd.get_dummies(x['Geography'],drop_first=True)
gender=pd.get_dummies(x['Gender'],drop_first=True)

# Concatenate the remaining dummies columns
x=pd.concat([x,states,gender],axis=1)

# Dropping the columns which contains categorical data
x=x.drop(['Geography','Gender'],axis=1)

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

# importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN
classifier=Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(activation='relu',input_dim=11,units=6,kernel_initializer='uniform'))

# Adding the second hidden layer
classifier.add(Dense(activation="relu",units=6,kernel_initializer="uniform"))

# Adding the output layer
classifier.add(Dense(activation="sigmoid",units=1,kernel_initializer="uniform"))

# Compile the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# fitting the ANN to the Training set
classifier.fit(x_train,y_train,batch_size=10,nb_epoch=100)

# Predicting the test set results
y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.5)

# making the confusion matrix and findout accuracy score
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
cm
acc=accuracy_score(y_test,y_pred)
acc   