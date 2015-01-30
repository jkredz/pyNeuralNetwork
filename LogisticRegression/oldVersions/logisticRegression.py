## logisticRegression.py    Dana Hughes    version 1.0     24-November-2014
##
## Logistic Regression model, which predicts a single output value given an 
## input vector.
##
## Revisions:
##   1.0   Initial version, modified from LinearRegressionModel with
##         batch and stochastic gradient descent.
##   1.01  Modified output to be softmax vector

import numpy as np
import random

class LogisticRegressionModel:
   """
   """

   def __init__(self, numVariables):
      """
      Create a new Logistic Regression model with randomized weights
      """

      # Set the weights to zero, including an extra weight as the offset
      self.N = numVariables
      self.weights = np.zeros(numVariables + 1)
      for i in range(self.N):
          self.weights[i] = random.random()
     


   def sigmoid(self, z):
      """
      """

      return 1.0/(1.0+np.exp(-z))


   def cost(self, data, output):
      """
      Determine the cost (error) of the parameters given the data and labels
      """

      # Add the offset term to the data
      cost = 0.0

      for i in range(len(data)):
         prediction = self.predict(data[i])
         error = prediction - output[i]
         cost = cost - (output[i]*np.log(prediction) + (1-output[i])*np.log(1-prediction))

      return cost


   def gradient(self, data, output):
      """
      Determine the gradient of the parameters given the data and labels
      """

      gradient = [0] * (self.N + 1)

      errors = []
      for i in range(len(data)):
         prediction = self.predict(data[i])
         errors.append(prediction - output[i])

      gradient[0] = sum(errors)/2

      for j in range(self.N):
         gradient[j+1] = 0
         for i in range(len(errors)):
            gradient[j+1] = gradient[j+1] + data[i][j] * errors[i]
         gradient[j+1] = gradient[j+1]/2

      return gradient


   def train_batch(self, data, output, learning_rate = 0.1, convergence = 0.0001, maxEpochs = 10000):
      """
      Perform batch training using the provided data and labels
      """

      epoch = 0

      while self.cost(data, output) > convergence and epoch < maxEpochs:
         print "Epoch", epoch, "- Cost:", self.cost(data,output)
         epoch+=1
         gradient = np.array(self.gradient(data, output))
         self.weights -= learning_rate * gradient


   def train_stochastic(self, data, output, learning_rate = 0.1, convergence = 0.0001, maxEpochs = 10000):
      """
      Perform stochastic (on-line) training using the data and labels
      """

      epoch = 0

      while self.cost(data, output) > convergence and epoch < maxEpochs:
         print "Epoch", epoch, "- Cost:", self.cost(data,output)
         epoch+=1
         for i in range(len(data)):
            error = output[i] - self.predict(data[i])
            self.weights[0] += learning_rate * error
            for j in range(self.N):
               self.weights[j+1] += learning_rate * error * data[i][j]
 


   def predict(self, data):
      """
      Predict the class given the data
      """

      return self.sigmoid(self.weights[0] + np.sum(self.weights[1:] * np.array(data)))



      
