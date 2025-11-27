#my own custom 

import numpy as np
import json
import os
import pickle

from neuron import neuron

module_dir = os.path.dirname(__file__)
NET_FILENAME = os.path.join(module_dir,"mlp_custom_weights.pkl" )

class mlp:
    def __init__(self, hidden_layer_size,adam=True):

        #NETWORK QUALITY PARAMETERS:
        self.batch_size = 100
        self.learning_rate = 0.001
        self.initial_learnig_rate = 0.1
        self.dynamic_learning_rate = False
        self.layer_initialized = False

        self.gesture_detected_index = 0 #gestures detected
        
        self.final_net_error = 0
        self.input = []
        self.hidden_layer_size = hidden_layer_size
        self.layer_w = []
        self.output = []
        self.adam_optimizer = adam
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.net_iteration = 0
        self.m = 0
        self.v = 0

        self.learning_rate_step = [] #visual the changes of learning rate

        self.epoch__max = 0
        self.epochs = [] #list of epochs to display in a seaborn plot

        self.ouput_errors = [] #list of errors at the last layer of nn
        self.rms_error= [] #list of total error in each epoch
        
        self.Loss = [] #cross-entropy-error

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
        print(self.softmax(self.output))

    def sigmoid(self,a):
        p1 = 0.2
        p2 = self.epoch__max*0.65
        return 1/(1+np.exp(p1*(a - p2)))
    
    # def CrossEntropyError(self,output, target):
    #     eps = 1e-15
    #     output = np.clip(output,eps,None)

    #     return -np.sum(target * np.log((self.softmax(output))))
    
    def CrossEntropyError(self, logits, target):
        probs = self.softmax(logits)
        return -np.sum(target * np.log(probs + 1e-15))

    def softmax(self,array):

        z = np.array(array)
        exp_z = np.exp(z-np.max(z))

        return exp_z/np.sum(exp_z)


    def Backpropagate(self):
        
        self.net_iteration +=1 
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

                    if self.adam_optimizer:
                        for j in range(len(neuron.weights)):
                            grad = neuron.error * neuron.inputs[j]
                            neuron.v[j] = self.beta2 * neuron.v[j] + (1-self.beta2)*grad*grad
                            neuron.m[j] = self.beta1 * neuron.m[j] + (1-self.beta1)*grad
                            
                            # Bias-corrected first and second moment estimates
                            m_hat = neuron.m[j] / (1 - np.power(self.beta1, self.net_iteration))
                            v_hat = neuron.v[j] / (1 - np.power(self.beta2, self.net_iteration))
                            
                            neuron.weights[j] -=  a * m_hat / (np.sqrt(v_hat) + self.epsilon)
                        
                        # Update bias with gradient
                        bias_grad = neuron.error
                        neuron.vbias = self.beta2 * neuron.vbias + (1 - self.beta2) * bias_grad * bias_grad
                        neuron.mbias = self.beta1 * neuron.mbias + (1 - self.beta1) * bias_grad
                        
                        # Bias-corrected bias moment estimates
                        mbias_hat = neuron.mbias / (1 - np.power(self.beta1, self.net_iteration))
                        vbias_hat = neuron.vbias / (1 - np.power(self.beta2, self.net_iteration))
                        
                        neuron.bias -= a * mbias_hat / (np.sqrt(vbias_hat) + self.epsilon)
                    else:
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

                    if self.adam_optimizer:
                        for j in range(len(neuron.weights)):
                            grad = neuron.error * neuron.inputs[j]
                            neuron.v[j] = self.beta2 * neuron.v[j] + (1-self.beta2)*grad*grad
                            neuron.m[j] = self.beta1 * neuron.m[j] + (1-self.beta1)*grad

                            # Bias-corrected first and second moment estimates
                            m_hat = neuron.m[j] / (1 - np.power(self.beta1, self.net_iteration))
                            v_hat = neuron.v[j] / (1 - np.power(self.beta2, self.net_iteration))

                            neuron.weights[j] -=  a * m_hat / (np.sqrt(v_hat) + self.epsilon)
                        
                        # Update bias with gradient
                        bias_grad = neuron.error
                        neuron.vbias = self.beta2 * neuron.vbias + (1 - self.beta2) * bias_grad * bias_grad
                        neuron.mbias = self.beta1 * neuron.mbias + (1 - self.beta1) * bias_grad
                        
                        # Bias-corrected bias moment estimates
                        mbias_hat = neuron.mbias / (1 - np.power(self.beta1, self.net_iteration))
                        vbias_hat = neuron.vbias / (1 - np.power(self.beta2, self.net_iteration))
                        
                        neuron.bias -= a * mbias_hat / (np.sqrt(vbias_hat) + self.epsilon)
                    else:
                        #update the weights of neuron 
                        for j in range(len(neuron.weights)):
                            neuron.weights[j] -=  a * neuron.error * neuron.inputs[j]
                        neuron.bias -= a * neuron.error 

            full_neuron_errors.append(n_error)

    def rms(self,output,target):
        sum = 0
        output = self.softmax(output)
        for i in range(len(output)):
            sum += (output[i] - target[i])*(output[i] - target[i])

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
        self.gesture_detected_index = np.argmax(self.softmax(self.output))

    def Train(self,input_data,target_data,max_epoch,learning_rate):
        self.learning_rate = learning_rate

        if not self.layer_initialized:
            self.init_layer_weights(input_data[0],target_data[0])
            self.layer_initialized = True

        self.epoch__max = max_epoch
        epoch = 0
        iter = 0
        self.final_net_error = 0
        self.Loss = []

        while epoch < self.epoch__max:
            

            loss = 0

            for b in range(self.batch_size):


                if iter >= len(input_data): iter = 0

                self.ouput_errors = []
                self.target = target_data[iter]
                self.input = input_data[iter]

                self.predict()

                probs = self.softmax(self.output)
                self.ouput_errors = probs - self.target

                # for j in range(len(self.target)):
                #     self.ouput_errors.append(self.output[j] - self.target[j])

                self.Backpropagate()
    
                loss += self.CrossEntropyError(self.output,self.target)
                iter += 1

            
            self.epochs.append(epoch)
            epoch = epoch + 1

            self.Loss.append(loss/self.batch_size)


            print(f'{self.ouput_errors} \tCE:{self.Loss[-1]} \tepoch: {epoch}')
        
        self.final_net_error = self.Loss[-1]


    def Validate(self,input_data,target_data,max_epoch,learning_rate):

        self.learning_rate = learning_rate
        self.epoch__max = max_epoch
        epoch = 0
        iter = 0

        self.epochs.clear()
        self.ouput_errors.clear()
        self.Loss.clear()

        while epoch < self.epoch__max:
            
            for b in range(self.batch_size):
                
                loss = 0
                if iter >= len(input_data): iter = 0

                self.ouput_errors = []
                self.target = target_data[iter]
                self.input = input_data[iter]

                self.predict()

                # for j in range(len(self.target)):
                #     self.ouput_errors.append(self.output[j] - self.target[j])
                probs = self.softmax(self.output)
                self.ouput_errors = probs - self.target
                
                self.Backpropagate()
                loss += self.CrossEntropyError(self.output,self.target)
                iter += 1
            
            self.epochs.append(epoch)
            epoch = epoch + 1
            self.Loss.append(loss/self.batch_size)


            print(f'{self.ouput_errors} \tCE:{self.Loss[-1]} \tepoch: {epoch}')
        
        self.final_net_error = self.Loss[-1]


    def save_weights(self, filename=NET_FILENAME):
          
        with open(filename, 'wb') as f:
            pickle.dump(self.layers, f)
        
        print(f"Network weights saved to {filename}")
    
    def load_weights(self, filename=NET_FILENAME):
        with open(filename, 'rb') as f:
            network_data = pickle.load(f)

        # self.layer_initialized = True
        self.layers = network_data
     
