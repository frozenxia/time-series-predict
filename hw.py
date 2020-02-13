import numpy as np
import sys
import json

def load_data(fn):
    return json.load(open(fn,'r'))

def simple_exponential_smoothing(xsets):
    s = 0
    for x in xsets:
        
