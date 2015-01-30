import numpy as np
import random

class RBM:
   """
   """

   def __init__(self, num_visible, num_hidden):
      """
      """

      self.num_visible = num_visible
      self.num_hidden = num_hidden

      # Weights is a matrix representing the weights between visible units
      # (rows) and hidden unit (columns)

      # Biases are column vectors with the number of hidden or visible units
      self.weights = np.zeros((num_visible, num_hidden))
      self.bias_visible = np.zeros((num_visible, 1))
      self.bias_hidden = np.zeros((num_hidden, 1))

      self.randomize_weights_and_biases(8*np.sqrt(6.0/(self.num_hidden + self.num_visible)))


   def randomize_weights_and_biases(self, value_range = 1):
      """
      Set all weights and biases to a value between [-range/2 and range/2]
      """

      for i in range(self.num_visible):
         for j in range(self.num_hidden):
            self.weights[i,j] = value_range*random.random() - value_range/2

      for i in range(self.num_visible):
         self.bias_visible[i,0] = value_range*random.random() - value_range/2

      for i in range(self.num_hidden):
         self.bias_hidden[i,0] = value_range*random.random() - value_range/2


   def sigmoid(self, z):
      """
      """

      return 1.0 / (1.0 + np.exp(-z))



   def get_probability_hidden(self, visible):
      """
      Returns the probability of setting hidden units to 1, given the 
      visible unit.
      """

      # h = sigmoid(W'v + c)
      return sigmoid(np.dot(self.weights.transpose(), visible) + self.bias_visible)


   def get_probability_visible(self, hidden):
      """
      Returns the probability of setting visible units to 1, given the
      hidden units.
      """

      return sigmoid(np.dot(self.weights, hidden) + self.bias_hidden)


   def sample_visible(self, hidden):
      """
      Generate a sample of the visible layer given the hidden layer.
      """

      P_visible = self.get_probability_visible(hidden)

      return [1.0 if random.random() < p else 0.0 for p in P_visible]


   def sample_hidden(self, visible):
      """
      Generate a sample of the hidden layer given the visible layer.
      """

      P_hidden = self.get_probability_hidden(visible)

      return [1.0 if random.random() < p else 0.0 for p in P_visible]


   # These are from deeplearning.net
   def propup(self, visible):
      """
      """

      net_input = np.dot(self.weights.transpose(), visible) + self.bias_hidden
      return net_input, self.sigmoid(net_input)


   def propdown(self, hidden):
      """
      """

      net_input = np.dot(self.weights, hidden) + self.bias_visible
      return net_input, self.sigmoid(net_input)
   

   def sample_h_given_v(self, v0):
      """
      """

      pre_sigmoid_h1, h1_mean = self.propup(v0)
      h1_sample = [1.0 if random.random() < p else 0.0 for p in h1_mean]
      return pre_sigmoid_h1, h1_mean, h1_sample

      
   def sample_v_given_h(self, h0):
      """
      """

      pre_sigmoid_v1, v1_mean = self.propdown(h0)
      v1_sample = [1.0 if random.random() < p else 0.0 for p in v1_mean]
      return pre_sigmoid_v1, v1_mean, v1_sample


   def contrastive_divergence(self, k=1):
      """
      """

