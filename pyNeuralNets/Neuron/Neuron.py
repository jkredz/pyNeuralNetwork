# Neuron.py                  12-May-2014               Dana Hughes
# version 1.1
#
# A implementation of a single Neuron, as a class
#
# References:
#
# M.T. Hagan, H.B. Demuth and M. Beale, "Neural Network Design"
#
# History:
#
# 1.0     Initial code
# 1.1     Added DelayNeuron and IntegratorNeuron
#
# TODO:
#
# 1.  Create unit tests.
# 2.  Add licensing information to comments.   
#

import math
import TransferFunctions

# Error messages
NOT_LIST_ERROR = "Parameter Error: Expecting a list or tuple"
INPUT_LAYER_MISMATCH_ERROR = "Input Mismatch: Expecting %d inputs, %d provided" 
WEIGHTS_MISMATCH_ERROR = "Weights Mismatch: Expecting %d inputs, %d provided" 
INPUT_RANGE_ERROR = "Input index out of range: Max index is %d"
WEIGHT_RANGE_ERROR = "Weight index out of range: Max index is %d"
INITIAL_VALUE_ERROR = "Number of initial values, %d, does not match delay, %d"


class Neuron:
   """
   A single neuron
   """

   def __init__(self, transferFunction = TransferFunctions.linear, numberInputs = 1):
      """
      Create a new neuron using the provided transfer function (linear by default) and number of inputs (1 by default)
      """

      self.transferFunction = transferFunction
      self.numberInputs = numberInputs
      self.inputs = [0] * numberInputs
      self.weights = [1] * numberInputs


   def setInputs(self, inputs):
      """
      Sets the inputs
      """

      # Ensure we were provided a list or tuple of the same length
      assert type(inputs) == list or type(inputs) == tuple, NOT_LIST_ERROR 
      assert len(inputs) == len(self.inputs), INPUT_LAYER_MISMATCH_ERROR % (len(self.inputs), len(inputs)) 

      # May come in as a tuple - convert to a list
      self.inputs = list(inputs)


   def setWeights(self, weights):
      """
      Sets the weights
      """
      
      # Ensure we were provided a list or tuple of the same length
      assert type(weights) == list or type(weights) == tuple, NOT_LIST_ERROR 
      assert len(weights) == len(self.weights), WEIGHT_MISMATCH_ERROR % (len(self.weights), len(weights))
      
      # May come in as a tuple - convert to a list
      self.weights = list(weights) 


   def setInput(self, index, input):
      """
      Set a particular input
      """

      # Ensure the index is in range
      assert index >= 0 and index < len(self.inputs), INPUT_RANGE_ERROR % len(inputs)

      self.inputs[index] = input


   def setWeight(self, index, weight):
      """
      Set a particular weight
      """

      # Ensure the index is in range
      assert index >= 0 and index < len(self.weights), WEIGHT_RANGE_ERROR % len(weights)

      self.weights[index] = weight


   def output(self):
      """ 
      Calculate the output
      """

      weightedInputs = [w*i for (w, i) in zip(self.weights, self.inputs)]
      return self.transferFunction(sum(weightedInputs))


class DelayNeuron(Neuron):
   """
   A delay neuron, for use in recurrent networks
   """

   def __init__(self, delay = 1, initialValues = [0]):
      """
      Create a new delay neuron with the given delay and initial values
      """ 
    
      assert delay == len(initialValues), INITIAL_VALUE_ERROR % (len(initialValues), delay)
 
      Neuron.__init__(self)
      self.delay = delay
      self.values = initialValues


   def output(self):
      """
      Provide the output and advance the input
      """

      output = self.values[0]
      self.values = self.values[1:] + self.inputs
      return output


class IntegratorNeuron(Neuron):
   """
   An integrator neuron
   """

   def __init__(self, initialValue = 0):
      """
      Create a new integrator neuron with the given initial value
      """

      Neuron.__init__(self)
      self.currentValue = initialValue


   def output(self):
      """
      Provide the output and update the integrator
      """

      output = self.currentValue
      self.currentValue += self.inputs[0]
      return output 
