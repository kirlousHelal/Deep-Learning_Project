import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from random import seed
from random import random
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from GUI import *


class MLP:
    def __init__(self, Path, hiddenLayer, neuron, eta, epochs, bias, fun):
        self.data = pd.read_csv(Path)
        self.hiddenLayer = hiddenLayer
        self.neuron = neuron  # number of neuron in hidden layer
        self.eta = eta
        self.epochs = epochs
        self.bias = bias
        self.fun = fun
        self.numFeatures = 5
        self.numClasses = 3
        self.inputs = 5
        self.outputs = 3
        self.network = list()



    def PreProcessing(self):
        # Replace NULL values
        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        self.data.iloc[:, :] = imputer.fit_transform(self.data.iloc[:, :])

        # preprocessing
        pre = preprocessing.LabelEncoder()
        self.data['gender'] = pre.fit_transform(self.data['gender'])
        
        self.oneHotEncoding()

        # normalization min/max
        cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm","body_mass_g"]
        for i in cols:
          self.data[i] =(self.data[i]-self.data[i].min())/(self.data[i].max()-self.data[i].min())
          

        # split all data into 3 classes
        count = 50  # number of data for each class
        for i in range(self.numClasses):
            df = self.data[count * i: count * (i + 1)]
            df.to_csv(f'data{i + 1}.csv', index=False)

        d1 = pd.read_csv("data1.csv")
        d2 = pd.read_csv("data2.csv")
        d3 = pd.read_csv("data3.csv")

        # split three classes into 30 train(60%)  &  20 test  -  with shuffle
        d1_train,d1_test,  d2_train,d2_test,  d3_train,d3_test = train_test_split(d1, d2, d3, train_size=0.6, shuffle=False)

        # merge 5 feature into 1 table - for train data
        dd1 = d1_train.iloc[:, 1:6]  # select 5 columns
        dd1 = d2_train.iloc[:, 1:6]
        dd1 = d3_train.iloc[:, 1:6]
        self.TrainData = pd.concat([dd1, dd1, dd1], axis=0)
        self.TrainData = self.TrainData.values.tolist()
        
        dd1 = d1_train.iloc[:, 6:] # select expected output - [1, 0, 0] for class 1
        dd1 = d2_train.iloc[:, 6:]
        dd1 = d3_train.iloc[:, 6:]
        self.TrainExpected = pd.concat([dd1, dd1, dd1], axis=0)
        self.TrainExpected = self.TrainExpected.values.tolist()
        
        # for test data
        dd1 = d1_test.iloc[:, 1:6]
        dd2 = d2_test.iloc[:, 1:6]
        dd3 = d3_test.iloc[:, 1:6]
        self.TestData = pd.concat([dd1, dd2, dd3], axis=0)
        self.TestData = self.TestData.values.tolist()
        
        dd1 = d1_test.iloc[:, 6:]
        dd2 = d2_test.iloc[:, 6:]
        dd3 = d3_test.iloc[:, 6:]
        self.TestExpected = pd.concat([dd1, dd2, dd3], axis=0)
        self.TestExpected = self.TestExpected.values.tolist()
        
        return self.TrainData, self.TrainExpected  ,  self.TestData, self.TestExpected


    def oneHotEncoding(self):
        ohe = OneHotEncoder()
        featurearray = ohe.fit_transform(self.data[['species']]).toarray()
        feature_label = ohe.categories_
        feature_label = np.array(feature_label).ravel()
        features = pd.DataFrame(featurearray, columns=feature_label)
        self.data = pd.concat([self.data, features], axis=1)
        return self.data
    
    def Sigmoid(self, x, derivative=False):
        f = 1 / (1 + np.exp(-x))  # sigmoid
        if derivative:
            #f = 1 / (1 + np.exp(-x)) * (1 - 1 / (1 + np.exp(-x)))  # derivative of sigmoid
            f = x * (1 - x)
        return f

    def Tanh(self, x, derivative=False):
        f = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))  # tanh
        if derivative:
            f = 1 - f * f  # derivative of tanh
        return f


    def Activation(self, name, x, derivative=False):
        if name == 'tanh':
            return self.Tanh(x, derivative)
        return self.Sigmoid(x, derivative)
            
    #############################################
    # Initialize a network
    def initialize_network(self):
        self.network = list()
        first_hidden_layer = [{'weights':[random() for i in range(self.inputs + self.bias)]} for i in range(self.neuron)]
        self.network.append(first_hidden_layer)
        
        # hidden layers number of weights that enter in this layer = # of neurons of previous layer + 1
        for i in range(self.hiddenLayer - 1):
            hidden_layer = [{'weights':[random() for i in range(self.neuron + self.bias)]} for i in range(self.neuron)]
            self.network.append(hidden_layer)
        
        output_layer = [{'weights':[random() for i in range(self.neuron + self.bias)]} for i in range(self.outputs)]
        self.network.append(output_layer)
        return self.network


    # Calculate neuron activation for an input - np.dot(weights,inputs) - weights & inputs must have the same shape
    def net(self, weights, inputs):
        if self.bias == 1:
            return np.dot(weights[:-1], inputs) + weights[-1]
        else:
            return np.dot(weights, inputs)
        
    # Forward propagate input to a network output
    def forward_propagate(self, row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            #inputs.append(1)
            for neuron in layer:
                netvalue = self.net(neuron['weights'], inputs)
                neuron['output'] = self.Activation(self.fun, netvalue)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs
      
    
    # Backpropagate error and store in neurons
    def backward_propagate_error(self, t):  # t=[1, 0, 0] for class 1
        output = list()
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            # Hidden layers => delta[l] = W * delta * f’
            new_inputs = []
            if i != len(self.network) - 1:
                for j in range(len(layer)):
                    errorSum = 0

                    for n in self.network[i + 1]:  # loop for brevise layer to cal.. ∑ weight * delta
                        errorSum += (n['weights'][j] * n['delta'])

                    neuron = layer[j]
                    neuron['delta'] = errorSum * self.Activation(self.fun, neuron['output'], True)
                    new_inputs.append(neuron['output'])
                output.append(new_inputs)


            # Output layer => delta[j] = (t - ŷ) * neuron['output'] (1-neuron['output'])
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    neuron['delta'] = (t[j] - neuron['output']) * self.Activation(self.fun, neuron['output'], True)
                    new_inputs.append(neuron['output'])
                output.append(new_inputs)
        return self.network

    # Update weights for specific layer
    def update_weights(self, row):
        inputs = row
        for i in range(len(self.network)):    # loop for each layer (hidden then output)
            layer = self.network[i]
            new_inputs = []
            for j in range(len(layer)):       # loop for each neuron
                neuron = layer[j]
                #inputs.append(1)
                for k in range(len(inputs)):  # loop for weights of this neuron - 1 for bias
                    neuron['weights'][k] -= (self.eta * neuron['delta'] * inputs[k])
                new_inputs.append(neuron['output'])
            inputs = new_inputs
                        

    # Train a network with TrainData
    def train_network(self):
        #self.initialize_network()
        for i in range(self.epochs):
            errorSum = 0
            for j in range(len(self.TrainData)):
                row = self.TrainData[j]
                expected = self.TrainExpected[j]
                
                #predicted = self.forward_propagate(row)
                predicted = self.predict(row)
                print(expected ,' ',predicted)
                
                for j in range(len(expected)):
                    errorSum += (expected[j] - predicted[j]) ** 2

                self.backward_propagate_error(expected)
                self.update_weights(row)
            print('> Epoch:',i ,' ,  Accuracy =', self.Accuracy_Matrix(expected, predicted))
            #print('>epoch= %d  ,  error= %.3f' % (i, errorSum))

    # Test a network with TestData
    def test_network(self):
        AccuracyList = list()
        for j in range(len(self.TestData)):
            row = self.TestData[j]
            expected = self.TestExpected[j]
            
            predicted = self.predict(row)
            
            accuracy = self.Accuracy_Matrix(expected, predicted)

            AccuracyList.append(accuracy)
            #print('\nAccuracy Test =' , accuracy)
            print('expected=',expected ,'  , predicted=',predicted)
        print('\n Test Accuracy: %.3f %%' % (accuracy))
        

    # Make a prediction with a network
    def predict(self, row):
        outputs = self.forward_propagate(row)  # outputs = [0.22 , 0.99 , 0.55]
        index = outputs.index(max(outputs))    # index = 1
        outputs = [0,0]
        outputs.insert(index, 1)              # outputs = [0 , 1 , 0]
        return outputs                         

    
    
    #####################

    def Confusion_Matrix(self, count1, count2):
        row1 = [count1, 20 - count1]
        row2 = [20 - count2, count2]
        confusion_matrix = np.array([row1, row2])
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                    display_labels=['class 1', 'Class 2'])
        cm_display.plot()
        plt.show()

    def Accuracy_Matrix(self, expected, predicted):
        correct = 0
        for i in range(len(expected)):
            if expected[i] == predicted[i]:
                correct += 1
        return correct / float(len(expected)) * 100.0
    
##################



# MLP( Path, hiddenLayer, neuron, eta, epochs, bias )
seed(1)
p = MLP("penguins.csv", check_hidden.get(), check_neurons.get(), Check_eta.get(), check_epochs.get(), Check_bias.get(), activation.get())

TrainData, TrainExpected , TestData, TestExpected = p.PreProcessing()
weight = p.initialize_network()

print('Train')
p.train_network()

print('\n Test')
p.test_network()





