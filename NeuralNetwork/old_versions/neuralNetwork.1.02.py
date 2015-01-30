## neuralNetwork.py    Dana Hughes    version 1.0     24-November-2014
##
## Fully connected neural network model, which predicts a output vector
## given an input vector.
##
## Revisions:
##   1.0   Initial version, modified from LogisticRegressionModel with
##         batch and stochastic gradient descent.

import numpy as np
import random

class NeuralNetwork:
   """
   """

   def __init__(self, layers):
      """
      Create a new Logistic Regression model with randomized weights
      """

      # Set the weights to zero, including an extra weight as the offset
      self.N = layers[0]
      self.M = layers[-1]
      self.numLayers = len(layers)

      # Set up the weight matrices
      self.weights = []
      
      for i in range(1,len(layers)):
         self.weights.append(np.zeros((layers[i], layers[i-1] + 1)))

      # Initialize the weights to small random values
      for i in range(len(self.weights)):
         for j in range(self.weights[i].shape[0]):
            for k in range(self.weights[i].shape[1]):
               self.weights[i][j,k] = 0.02*random.random() - 0.01
     

   def printNN(self):
      """
      Print NN information
      """

      print "Number of layers:", self.numLayers
      print "Number of inputs:", self.N
      print "Number of outputs:", self.M

      print

      print "Layers:"
      print "-------"

      print "Input (",self.N,") ->",

      for i in range(self.numLayers-1):
         print "Layer", i, "(", self.weights[i].shape, ") ->",

      print "Output (",self.M,")"

      print
      print "-------------------"
      print 
      
      np.set_printoptions(precision=3)
      np.set_printoptions(suppress=True)

      for i in range(self.numLayers-1):
         print "Weights - Layer",i
         print self.weights[i]
         print


   def sigmoid(self, z):
      """
      Activation function
      """

      return 1.0/(1.0+np.exp(-z))


   def softmax(self, z):
      """
      Returns the softmax of the given vector.
      """

      results = np.zeros(len(z))

      for i in range(len(z)):
         results[i] = exp(z)

      results = results/sum(results)

      return results


   def cost(self, data, output):
      """
      Determine the cost (error) of the parameters given the data and labels
      """

      # Add the offset term to the data
      cost = 0.0

      for i in range(len(data)):
         prediction = self.predict(data[i])

         for j in range(self.M):
            cost = cost + np.sum((prediction-output[i])**2)

      return cost/(2.0*len(data))


   def train_batch(self, data, output, learning_rate = 0.1, convergence = 0.0001, maxEpochs = 10000):
      """
      Perform batch training using the provided data and labels
      """

      epoch = 0

      while self.cost(data, output) > convergence and epoch < maxEpochs:
         print "Epoch", epoch, "- Cost:", self.cost(data,output)
         epoch+=1

         DW = [None]*(self.numLayers-1)
         for i in range(self.numLayers-1):
            DW[i] = np.zeros(self.weights[i].shape)

         for m in range(len(data)):

            # Forward propagation
            activations = self.forwardprop(data[m])

            # Backwards propagation
            deltas = self.backprop(data[m], output[m], activations)

            # Update the weights delta
            for i in range(self.numLayers-1):
               for j in range(len(activations[i])):
                  DW[i][:,j+1] += activations[i][j] * deltas[i]
               DW[i][:,0] += deltas[i]

         for i in range(len(self.weights)):
            self.weights[i] -= learning_rate*DW[i]/len(data)


   def train_minibatch(self, data, output, learning_rate = 0.1, convergence = 0.0001, maxEpochs = 10000, numBatches = 10):
      """
      Perform batch training using the provided data and labels
      """

      epoch = 0

      batch_size = int(len(data)/numBatches)

      while self.cost(data, output) > convergence and epoch < maxEpochs:
         print "Epoch", epoch, "- Cost:", self.cost(data,output)
         epoch+=1

         for i in range(numBatches):
            data_batch = []
            output_batch = []

            for j in range(batch_size):
               idx = random.randrange(0,len(data))
               data_batch.append(data[idx])
               output_batch.append(output[idx])
  

            DW = [None]*(self.numLayers-1)
            for i in range(self.numLayers-1):
               DW[i] = np.zeros(self.weights[i].shape)

            for m in range(len(data_batch)):

               # Forward propagation
               activations = self.forwardprop(data_batch[m])

               # Backwards propagation
               deltas = self.backprop(data_batch[m], output_batch[m], activations)

               # Update the weights delta
               for i in range(self.numLayers-1):
                  for j in range(len(activations[i])):
                     DW[i][:,j+1] += activations[i][j] * deltas[i]
                  DW[i][:,0] += deltas[i]

            for i in range(len(self.weights)):
               self.weights[i] -= learning_rate*DW[i]/len(data)


   def backprop(self, data, output, activations):
      """
      Perform backwards propagation, for a single data point
      """

      deltas = [None] * self.numLayers

      # First, cacluate the error between output activation and given output

      deltas[self.numLayers-1] = -(output - activations[self.numLayers-1])
      deltas[self.numLayers-1] *= activations[self.numLayers-1]*(1.0 - activations[self.numLayers-1])

      for i in range(self.numLayers-2, 0, -1):
         deltas[i] = np.zeros(self.weights[i].shape[1]-1)
         for j in range(self.weights[i].shape[1] - 1):
            deltas[i][j] = np.sum(self.weights[i][:,j+1] * deltas[i+1])
         deltas[i] *= activations[i]*(1.0-activations[i])

      return deltas[1:]


   def forwardprop(self, data):
      """
      Perform forward propagation, giving the activations at each layer
      """

      x = np.array(data)
      activations = [x]

      # Calculate the activation of each layer in the neural network
      for i in range(self.numLayers-1):
         activation = np.zeros(self.weights[i].shape[0])
         for j in range(self.weights[i].shape[0]):
            activation[j] = np.sum(self.weights[i][j,0] + self.weights[i][j,1:] * x)
            activation[j] = self.sigmoid(activation[j])

         activations.append(activation)
         x = activation

      return activations


   def predict(self,data):
      """
      """

      activations = self.forwardprop(data)
      return activations[-1]
