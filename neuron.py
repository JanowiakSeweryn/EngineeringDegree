import random
import sys

class neuron:
    def __init__(self,input_size):
        self.weights = []
        self.weights_sum = 0
        self.bias = random.uniform(-0.02,0.02)

        self.error = 0

        for i in range(input_size):
            self.weights.append(random.uniform(-0.02,0.02))
    
    def reLu(self,wsum):
        if wsum < 0: return 0
        else: return wsum 
    
    def reLu_dv(self,wsum):
        if(wsum > 0 ): return 1
        else : return 0

    def value(self,input):
        sum = self.bias

        for i in range(len(self.weights)):
            sum = sum + self.weights[i]*input[i]
        self.weights_sum = sum
        return self.reLu(self.weights_sum)
        
