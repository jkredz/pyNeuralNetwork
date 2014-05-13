# Layer.py                   12-May-2014               Dana Hughes
# version 1.0
#
# A implementation of a layer of Neurons, as a class
#
# References:
#
# M.T. Hagan, H.B. Demuth and M. Beale, "Neural Network Design"
#
# History:
#
# 1.0     Initial code
#
# TODO:
#
# 1.  Create unit tests.
# 2.  Add licensing information to comments.   
#

import math
import TransferFunctions
import Neuron

# Error messages
NOT_LIST_ERROR = "Parameter Error: Expecting a list or tuple"
INPUT_LAYER_MISMATCH_ERROR = "Input Mismatch: Expecting %d inputs, %d provided" 
WEIGHTS_MISMATCH_ERROR = "Weights Mismatch: Expecting %d inputs, %d provided" 
INPUT_RANGE_ERROR = "Input index out of range: Max index is %d"
WEIGHT_RANGE_ERROR = "Weight index out of range: Max index is %d"

class Layer:
   """
   A single layer of neurons 
   """

   def __init__(self, numberNeurons = 1, numberInputs = 1, transferFunction = TransferFunctions.linear):
      """
      Create a new layer using the provided transfer function (linear by default), number of neurons (1 by default), 
      and number of inputs (1 by default).  Number of inputs does not include bias term.
      """

      self.numberNeurons = numberNeurons
      self.input = [1] + [0] * numberInputs
      self.neurons = [None] * numberNeurons
      for i in range(numberNeurons):
         neuron = Neuron.Neuron(transferFunction, numberInputs + 1)
         self.neurons[i] = neuron


   def input(self, input):
      """
      Sets the input.  Does not set the bias term (input at index 0)
      """

      # Ensure we were provided a list or tuple of the same length
      assert type(input) == list or type(input) == tuple, NOT_LIST_ERROR 
      assert len(input) == len(self.input), INPUT_LAYER_MISMATCH_ERROR % (len(self.input), len(input)) 

      # May come in as a tuple - convert to a list
      self.input[1:] = list(input)
      for neuron in self.neurons:
         self.setInputs(self.input)


   def output(self):
      """ 
      Calculate the output
      """

      output = [n.output() for n in self.neurons]

      return output 
