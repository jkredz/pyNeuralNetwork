# Networks.py                  12-May-2014               Dana Hughes
# version 1.0
#
# Various network architectures 
#
# References:
#
# M.T. Hagan, H.B. Demuth and M. Beale, "Neural Network Design"
#
# History:
#
# 1.0     Initial single layer perceptron code 
# 
# TODO:
#
# 1.  Create unit tests.
# 2.  Add licensing information to comments.   
# 3.  Add general, multilayer network
# 4.  Add general, single layer network
# 5.  Add Hamming network
# 6.  Add Hopfield network

import math
import TransferFunctions
import Neuron
import Layer

# Error messages
INPUT_SIZE_ERROR = "Incorrect number of inputs, expecting %d, received %d"

class Perceptron:
   """
   A perceptron is a single-layer neural network with symmetric hard limit transfer functions
   """

   def __init__(self, numberInputs, numberOutputs):
      """
      Create a new perceptron network
      """

      self.inputs = [1] + [0] * numberInputs
      self.outputs = [0] * numberOutputs
      self.neurons = [None] * numberOutputs
      for i in range(numberOutputs):
         self.neurons[i] = Neuron.Neuron(TransferFunctions.symmetricHardLimit, numberInputs + 1)


   def output(self, input):
      """
      Determine the output for the given input
      """

      assert len(input) == len(self.inputs)-1, INPUT_SIZE_ERROR % (len(self.inputs)-1, len(input))

      self.inputs[1:] = input
      for i in range(len(self.neurons)):
         self.neurons[i].setInputs(self.inputs)
      return [n.output() for n in self.neurons] 
