import math
import numpy as np

def plateau(x, width = 0.2):
    return 0.5/math.tanh(width)*(math.tanh(x + width)-math.tanh(x - width))

def plateauFlat(x, width = 0.12):
    return 0.5/math.tanh(10)*(math.tanh(10/width*x + 10)-math.tanh(10/width*x - 10))

def deltaFunctionBarier(x, width = 0.2):
    return 100/math.sqrt(math.pi)*math.exp(-100/width*x**2)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoidBarier(x, width = 0.03, under_barier_scale = 1):
    return 1/(1+np.exp(10/width*(x-width/2))) - min(under_barier_scale*x,0)

def normTanh(x):
    return np.tanh(x)

def normTan(x):
    return 2/math.pi*np.arctan(x)