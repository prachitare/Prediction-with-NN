#importing the dataset

import pandas as pd 
import numpy as np 

# Read in the csv file and convert them into the arrays. Arrays are the data format thet can be processed

db = pd.read_csv('houseprice.csv')

dataset = db.values
dataset

X = dataset[:, 0:10]
Y = dataset[:,  10]

#I want to use the code in ‘preprocessing’ within the sklearn package. Then, we use a function called the min-max scaler, which scales the dataset so that all the input features lie between 0 and 1 inclusive

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

#print(X_scale)

# split the dataset into the input features

from sklearn.model_selection import train_test_split
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)


#Unfortunately, this function only helps us split our dataset into two. Since we want a separate validation set and test set, we can use the same function to do the split again on val_and_test:

X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

#X_train (10 input features, 70% of full dataset)
#X_val (10 input features, 15% of full dataset)
#X_test (10 input features, 15% of full dataset)
#Y_train (1 label, 70% of full dataset)
#Y_val (1 label, 15% of full dataset)
#Y_test (1 label, 15% of full dataset)

print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

#applying the neural network to the following process model
# step 1 - Setting the the architecture

# let us import the important libraries from keras
from keras.models import Sequential
from keras.layers import Dense

# now we define our keras sequential model. How we want it

model = Sequential([Dense(32, activation = 'relu', input_shape = (10, )) , Dense(32, activation = 'relu'), Dense(1, activation = 'sigmoid'), ])
###
#Telling it which algorithm you want to use to do the optimization
#Telling it what loss function to use
#Telling it what other metrics you want to track apart from the loss function
###


# compiling the model with these settings we need to call the model like model.compile

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# sgd refers to the stochastic gradient descent
#training is pretty easy we just need to write one line code
trtm = model.fit(X_train, Y_train, batch_size = 32, epochs = 100, validation_data = (X_val, Y_val)) #trtm = training the model
# we are fitting the parameter to the model
#we specify the size of our mini-batch and how long we want to train it for (epochs)

# now you can evaluate with your test set

model.evaluate(X_test, Y_test)[1]
#The reason why we have the index 1 after the model.evaluate function is because the function returns the loss as the first element and the accuracy as the second element. To only output the accuracy, simply access the second element (which is indexed by 1, since the first element starts its indexing from 0).

#### STEP 2 - VISUALIZING LOSS AND ACCURACY

import matplotlib.pyplot as plt

#plt.plot(trtm.history['loss'])
#plt.plot(trtm.history['val_loss'])
#plt.title('Model loss')
#plt.ylabel('Loss')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Val'], loc='upper right')
#plt.show()

#We can do the same to plot our training accuracy and validation accuracy with the code below:

plt.plot(trtm.history['accuracy'])
plt.plot(trtm.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
#plt.show()

