"""
 Classification using Neural Network Approach with 10-fold Cross validation 

"""
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.cross_validation import KFold, cross_val_score


# read the input data set
input_data = pd.read_csv('sample.csv', header=None)

# calculate the total no of records and features in data set
row = input_data.shape[0]
column = input_data.shape[1]

# Data set values of all features are keep track for evaluation
X = input_data.iloc[:, 0:column-1].values
Y = input_data.iloc[:, column-1].values

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Splitting the dataset into the Training set and Test set
mask = np.random.rand(len(input_data)) < 0.8

X_train = input_data[mask].iloc[:, 0:column-1].values
X_test = input_data[~mask].iloc[:, 0:column-1].values

Y_train = input_data[mask].iloc[:, column-1].values
Y_test = input_data[~mask].iloc[:, column-1].values

# Label encoding
# Encoding the string values into different numeric value to represents different labels
# the target variable contains string value, therefore it requires to covert it into different numerical labels  
labelencoder_Y_train = LabelEncoder()
Y_train = labelencoder_Y_train.fit_transform(Y_train)

labelencoder_Y_test = LabelEncoder()
Y_test = labelencoder_Y_test.fit_transform(Y_test)


# Feature Scaling
# StandardScaler method is applied on train data as well as test data set for fitting and transforming 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Function create_model is defined to create Neural Network Model
def create_model():
    #Initializing Neural Network
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu', input_dim = 295))
    # Adding the second hidden layer
    classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu'))
    # Adding the output layer
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    # Compiling Neural Network
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return classifier


# calling function create_model() in order to apply the classification system
model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=10, verbose=0)

# Applying train and test data
estimator = model.fit(X_train, Y_train)

# Predicting the Test set results
#prediction = model.predict(X_test)
#prediction = (prediction > 0.5)

# Creating the Confusion Matrix
#cmMatrix = confusion_matrix(Y_test, prediction)
#print("Score:", cmMatrix)

#plt.scatter(Y_test, prediction)
#plt.xlabel("True Values")
#plt.ylabel("Predictions")


# evaluate using 10-fold cross validation
k_fold = KFold(len(X_train), n_folds=10, shuffle=True, random_state=None)
result = cross_val_score(model, X_train, Y_train, cv=k_fold)

print("Score: ", result)


