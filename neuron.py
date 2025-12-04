import random
import sys
import numpy as np


class neuron:
    def __init__(self,input_size):

        self.inputs = []
        self.weights = [] 
        self.weights_sum = 0
        self.bias = 0.02
        self.error = 0
        self.m = []
        self.v = []

        for i in range(input_size):
            limit = np.sqrt(6 / (input_size+10))
            self.weights.append(random.uniform(-limit, limit))
            # self.weights.append(random.uniform(-0.5,0.5) * np.sqrt(2/input_size))
            self.m.append(0)
            self.v.append(0)
        
        self.mbias = 0
        self.vbias = 0

    
    def reLu(self,wsum):
        if wsum < 0: return 0
        else: return wsum 
    
    def reLu_dv(self,wsum):
        if(wsum > 0 ): return 1
        else : return 0

    def value(self,input):
        sum = self.bias
        self.inputs = input
        for i in range(len(self.weights)):
            sum = sum + self.weights[i]*input[i]
        self.weights_sum = sum

        return self.reLu(self.weights_sum)
    
    def get_sum(self,input):
        sum = self.bias
        self.inputs = input
        for i in range(len(self.weights)):
            sum = sum + self.weights[i]*input[i]
        self.weights_sum = sum

        return self.weights_sum

