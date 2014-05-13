# TransferFunctions.py         12-May-2014               Dana Hughes
# version 1.0
#
# A list of transfer functions available for neurons.  This list was compiled
# from Table 2.1 of the following
#
# M.T. Hagan, H.B. Demuth and M. Beale, "Neural Network Design"
#
# History:
#
# 1.0     Initial code
#
# TODO:
#
# 1.  Create unit tests of each function.
# 2.  Consider renaming of functions to match names in current literature.
# 3.  Add licensing information to comments.   
#

import math

def hardLimit(input):
   """
   Returns 1 if input is equal to or over 1, 0 otherwise
   """

   return 1 if input >= 0 else 0


def symmetricHardLimit(input):
   """
   Returns 1 if input is equal to or over 0, -1 otherwise
   """

   return 1 if input >= 0 else -1


def linear(input):
   """
   Identity transformation - simply returns the input
   """

   return input


def saturatingLinear(input):
   """
   Returns the input if between 0 and 1, 0 or 1 otherwise
   """

   if input < 0:
      return 0
   elif input <= 1:
      return input
   else:
      return 1


def symmetricSaturatingLinear(input):
   """
   Returns the saturating linear input, except with range from -1 to 1
   """

   if input < -1:
      return -1
   elif input <= 1:
      return input
   else:
      return 1


def logSigmoid(input):
   """
   Returns the sigmoid of the input
   """

   return 1/(1+math.exp(-input))


def hyperbolicLogSigmoid(input):
   """
   Returns the hyperbolic sigmoid of the input
   """

   return (math.exp(input) - math.exp(-input))/(math.exp(input) + math.exp(-input))


def positiveLinear(input):
   """
   Return 0 if negative, linear if positive
   """

   return 0 if input < 0 else input


