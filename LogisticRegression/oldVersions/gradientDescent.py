## gradientDescent.py
##
## Gradient descent algorithms
##

import numpy as np

def gradientDescent(position, gradientFunction, learningRate):
   """
   Return a new position given the gradient and learning rate
   """

   return position - gradientFunction(position)*learningRate
