## logisticRegression.py    Dana Hughes    version 1.0     24-November-2014
##
## Logistic Regression model, which predicts a single output value given an 
## input vector.
##
## Revisions:
##   1.0   Initial version, modified from LinearRegressionModel with
##         batch and stochastic gradient descent.
##   1.01  Modified output to be softmax vector
##   1.02  Fixed the predictor.  Converges much better now!

import numpy as np
import random

class LogisticRegressionModel:
   """
   """

   def __init__(self, numVariables, numOutputs):
      """
      Create a new Logistic Regression model with randomized weights
      """

      # Set the weights to zero, including an extra weight as the offset
      self.N = numVariables
      self.M = numOutputs
      #theta trans:  create transposed weights matrix (rows = 4 variables + 1 offset, cols=3 output flowers)
      self.weights = np.zeros((numVariables + 1, numOutputs))
   #   for i in range(self.N+1):
   #       for j in range(self.M):
   #           self.weights[i,j] = random.random()
     


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
      # from 1 to # of training sets
      for i in range(len(data)):
         # make prediction for each row of training [1xM]
         prediction = self.predict(data[i])
         #from 1 to # of outputs M
         for j in range(self.M):
            # calculate Cost of predicting each output and continuously added:  (1-y)log(1-h(theta))+y*log(h(theta))
            cost = cost + ((1.0-output[i][j])*np.log(1.0-prediction[j]) + output[i][j]*np.log(prediction[j]))

      #total cost J(theta)=-1/m*sum(cost) [1]
      return -cost/len(data)


   def gradient(self, data, output):
      """
      Determine the gradient of the parameters given the data and labels
      """
      #create empty gradient matrix of [rows:N+1, cols:#outputs M]
      gradient = np.zeros((self.N + 1, self.M)) 
      #for each data set
      for k in range(len(data)):
         prediction = self.predict(data[k])
         #for each column in the dataset
         for j in range(self.M):
            #calculate the gradient(actual - prediction) for each output in the 0th row
            gradient[0,j] -= (output[k][j] - prediction[j])
            #for each parameter
            for i in range(self.N):
               #gradient of row's output is = x*(y-h(theta))
               gradient[i+1,j] -= data[k][i]*(output[k][j] - prediction[j])
      #gradient matrix/m [N+1 by M matrix]
      return gradient/len(data)

   #data:X matrix of data sets [n x N], output:Y vector of outputs [n x [1 x M]]
   def train_batch(self, data, output, learning_rate = 0.1, convergence = 0.0001, maxEpochs = 10000):
      """
      Perform batch training using the provided data and labels
      """

      epoch = 0

      while self.cost(data, output) > convergence and epoch < maxEpochs:
         #calculate cost
         print "Epoch", epoch, "- Cost:", self.cost(data,output)
         epoch+=1
         #calculate gradient matrix
         gradient = np.array(self.gradient(data, output))
         #new weights = - learning rate * gradient matrix [N+1 by M matrix]
         self.weights -= learning_rate * gradient



   def train_minibatch(self, data, output, learning_rate = 0.1, convergence = 0.0001, maxEpochs = 10000, numBatches = 10):
      """
      Perform batch training using the provided data and labels
      """

      epoch = 0
      batchSize = int(len(data)/numBatches)

      while self.cost(data, output) > convergence and epoch < maxEpochs:
  
         print "Epoch", epoch, "- Cost:", self.cost(data, output)
         epoch+=1

         for i in range(numBatches):
            batch_data = data[i*batchSize:(i+1)*batchSize]
            batch_output = output[i*batchSize:(i+1)*batchSize]

            gradient = np.array(self.gradient(batch_data, batch_output))
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
            gradient = np.array(self.gradient([data[i]], [output[i]]))
            self.weights -= learning_rate * gradient 
 


   def predict(self, data):
      """
      Predict the class probabilites given the data
      """
      #create zero vector with size same as number of outputs (1x3) representing the 3 different iris
      prediction = np.zeros(self.M)
      #prediction for each type of flower...
      for i in range(self.M):
         #for each column (type of flower predicted) : calculate the prediction using e^(weight 0 + sum(weights*X))
         prediction[i] = np.exp(self.weights[0,i] + np.sum(self.weights[1:,i]*np.array(data)))

      partition = sum(prediction)
      #turn prediction into a vector with probability for each class
      for i in range(self.M):
         prediction[i] = prediction[i]/partition

      return prediction



      
