import math
import numpy as np
from scipy.special import expit 

def plateau(x, width = 0.2):
    return 0.5/math.tanh(width)*(math.tanh(x + width)-math.tanh(x - width))

def plateauFlat(x, width = 0.12, scale = 1):
    return 0.5/math.tanh(10*scale)*(math.tanh(10*scale/width*x + 10*scale)-math.tanh(10*scale/width*x - 10*scale))

def deltaFunctionBarier(x, width = 0.2):
    return 100/math.sqrt(math.pi)*math.exp(-100/width*x**2)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoidBarier(x, width = 0.03, under_barrier_scale = 1):
    # same as 1 / (1 + exp(z)), but stable
    z = (10/width) * (x - width/2)
    return float(expit(-z) - min(under_barrier_scale * x, 0))

def normTanh(x):
    return np.tanh(x)

def normTan(x):
    return 2/math.pi*np.arctan(x)