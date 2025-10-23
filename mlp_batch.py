import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
from neuron import neuron

class mlp:
    def __init__(self, hidden_layer_size):

        
        self.input = []
        self.hidden_layer_size = hidden_layer_size
        self.layer_w = []
        self.output = []
        self.presoftmax_output = []
        self.learning_rate = 0.001
        self.initial_learnig_rate = 0.1

        self.final_net_error = 0

        self.learning_rate_step = [] #visual the changes of learning rate

        self.epoch__max = 0
        self.epochs = [] #list of epochs to display in a seaborn plot

        self.ouput_errors = [] #list of errors at the last layer of nn
        self.rms_error= [] #list of total error in each epoch
        
        self.Loss = [] #cross-entropy-error

        self.total_neuron_error = [] #list of list of errors each neuron have (one error per sample)

        # --- initialize layers ---
    
        self.layers = [] #list of al layers
        self.ouput_neurons = [] #output layer [final result]


    #input change one per epoch or one per frame
    def input_change(self,input):
        self.input = input

    def init_layer_weights(self,input_sample,target_sample):

        layer_input_size = len(input_sample) #how many weights each neuron have (it depend on which layer)

        for s in self.hidden_layer_size: #for each layer
            layer_neurons = [] #single layer

            for i in range(s):
                layer_neurons.append(neuron(layer_input_size))

            self.layers.append(layer_neurons)
            layer_input_size = s

        for i in range(len(target_sample)):
            self.ouput_neurons.append(neuron(layer_input_size))
        
        self.layers.append(self.ouput_neurons)


    def disp(self):
        # for l in self.layers:
        #     for n in l:
        #         print(n.inputs)

        print(self.softmax(self.output))

    def sigmoid(self,a):
        p1 = 0.2
        p2 = self.epoch__max*0.65
        return 1/(1+np.exp(p1*(a - p2)))
    
    def CrossEntropyError(self,output, target):
        eps = 1e-15
        output = np.clip(output, eps, 1 - eps)
        return -np.sum(target * np.log((self.softmax(output))))

    def softmax(self,array):

        z = np.array(array)
        exp_z = np.exp(z-np.max(z))

        return exp_z/np.sum(exp_z)


    
    def PlotEntropyError(self):
        plt.plot(self.epochs,self.Loss)
        plt.show()

    def Backpropagate(self):
        
        a = self.learning_rate #learning rate
    
        layer_index = len(self.layers)
        full_neuron_errors = [] #list of n_error for each layer

        for l in reversed(self.layers): 

            n_error = [] #list of errors (each error for neuron) in a layer

            layer_index -= 1

            if layer_index == len(self.layers) - 1:
                for i ,neuron in enumerate(l):
                    neuron.error = self.ouput_errors[i]

                    n_error.append(neuron.error)

                        #update the weights of neuron 
                    for j in range(len(neuron.weights)):
                        neuron.weights[j] -=  a * neuron.error * neuron.inputs[j]
                    neuron.bias -= a * neuron.error 
                
            
            else:

                #for each neuron we compute neuron_error 
                for index,neuron in enumerate(l):   
                    
                    nerr_sum = 0
                    #for each neuron in next layer (with knows error)
                    for n in self.layers[layer_index + 1]:    
                        nerr_sum = nerr_sum + n.error * n.weights[index]
                    
                    neuron.error = nerr_sum * neuron.reLu_dv(neuron.weights_sum)
                    n_error.append(neuron.error)

                    #update the weights of neuron 
                    for j in range(len(neuron.weights)):
                        neuron.weights[j] -=  a * neuron.error * neuron.inputs[j]
                    neuron.bias -= a * neuron.error 
            
            full_neuron_errors.append(n_error)

    def rms(self,output,target):
        sum = 0
        for i in range(len(output)):
            sum = (output[i] - target[i])*(output[i] - target[i])

        return sum/len(output)
    
    def predict(self):
        
        prev_input = self.input

        for i in range(len(self.layers)):
        
            input = []
            if(i != len(self.layers) - 1):
                for n in self.layers[i]:
                    input.append(n.value(prev_input))
                prev_input = input
            else:
                for n in self.layers[i]:
                    input.append(n.get_sum(prev_input))
                prev_input = input
        
        self.output = prev_input
        # print(self.softmax(self.output))

    def Train(self,input_data,target_data,max_epoch):
        
        self.init_layer_weights(input_data[0],target_data[0])
        self.epoch__max = max_epoch
        epoch = 0

        while epoch < self.epoch__max:
            
            
            for i in range(len(input_data)):
                loss = []
                self.ouput_errors = []
                self.target = target_data[i]
                self.input = input_data[i]

                self.predict()

                for j in range(len(self.target)):
                    self.ouput_errors.append(self.output[j] - self.target[j])

                self.Backpropagate()
                loss.append(self.CrossEntropyError(self.output,self.target))
            
            self.epochs.append(epoch)
            epoch = epoch + 1
            self.Loss.append(loss[-1])


            print(f'{self.ouput_errors} \tCE:{loss[-1]} \tepoch: {epoch}')
        
        self.final_net_error = self.Loss[len(self.Loss)-1]


       


        


    
        





    
