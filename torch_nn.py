import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os


module_dir = os.path.dirname(__file__)
NET_FILENAME = os.path.join(module_dir,"weights.pth" )
NET_FILENAME_DYNAMIC =  os.path.join(module_dir,"weights_dynamics.pth" )


class mlp(nn.Module):
    def __init__(self, hidden_sizes, solver="adam"):
        super(mlp, self).__init__()
        
        self.gesture_detected_index = 0
        self.output_size = 0
        self.hidden_sizes = hidden_sizes
        self.input_size = 0
        self.final_net_error = 0
        self.solver = solver

        self.epochs = []
        self.loss_history = []

        self.input = []
        self.Layers_initialized = False

    def create_layers(self,input_size,output_size):
        self.input_size = input_size
        self.output_size = output_size

        layers = []
        in_size = self.input_size

        for h in self.hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h

        layers.append(nn.Linear(in_size, self.output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def assign_layers(self,layers):
        self.net = layers

    def Train(self, input_data, target, max_epoch,lr):

        self.input_size = len(input_data[0])
        self.output_size = len(target[0])
        
        if not self.Layers_initialized:
            if self.input_size != 0:
                self.create_layers(self.input_size,self.output_size)

        self.epochs = []
        self.Loss = []
        self.loss_history = []

        criterion = nn.CrossEntropyLoss()
        if self.solver == "adam":
            optimizer = optim.Adam(self.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(self.parameters(), lr=lr)

        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        target_tensor = torch.tensor([t.index(1) for t in target], dtype=torch.long)

        for epoch in range(max_epoch):
            outputs = self(input_tensor)
            loss = criterion(outputs, target_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if (epoch + 1) % 10 == 0 or epoch == 0:
                # print(f"Epoch [{epoch+1}/{max_epoch}], Loss: {loss.item():.4f}")

            self.epochs.append(epoch)
            self.loss_history.append(loss.item())

        self.Loss = self.loss_history
        self.final_net_error = self.Loss[-1]

        self.Layers_initialized = True
    
    def Validate(self, input_data, target, max_epoch, lr):
       
        self.epochs = []
        self.Loss = []
        self.loss_history = []

        criterion = nn.CrossEntropyLoss()

        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        target_tensor = torch.tensor([t.index(1) for t in target], dtype=torch.long)

        self.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            for epoch in range(max_epoch):
                outputs = self(input_tensor)
                loss = criterion(outputs, target_tensor)

                # if (epoch + 1) % 10 == 0 or epoch == 0:
                #     print(f"Validation Epoch [{epoch+1}/{max_epoch}], Loss: {loss.item():.4f}")

                self.epochs.append(epoch)
                self.loss_history.append(loss.item())

        self.Loss = self.loss_history
        self.final_net_error = self.Loss[-1]
        self.train()  # Set model back to training mode
            
    def input_change(self, input):
        self.input = input
        
    def disp(self):
        print(self.predict().squeeze(0).tolist())
    
    def save_weights(self,dynamic=False):
        if dynamic: torch.save(self.net.state_dict(),NET_FILENAME_DYNAMIC)
        else: torch.save(self.net.state_dict(),NET_FILENAME)
    
    def load_weights(self,dynamic=False):
        if dynamic: self.net.load_state_dict(torch.load(NET_FILENAME_DYNAMIC))
        else: self.net.load_state_dict(torch.load(NET_FILENAME))

    def predict(self):
        self.eval()
        with torch.no_grad():
            if isinstance(self.input[0], float):
                input_tensor = torch.tensor([self.input], dtype=torch.float32)
            else:
                input_tensor = torch.tensor(self.input, dtype=torch.float32)

            outputs = self(input_tensor)
            probs = F.softmax(outputs, dim=1)
            
        self.gesture_detected_index = np.argmax(probs.squeeze(0).tolist())
        return probs
    
    def get_acc(self, input_data, target_data):
        """
        Calculate accuracy on the given input and target data.
        Returns accuracy as a percentage.
        """
        tp = 0
        tn = 0
        
        for x, y in zip(input_data, target_data):
            self.input_change(x)
            self.predict()
            
            y_pred = self.gesture_detected_index
            y_true = np.argmax(y)
            
            if y_pred == y_true:
                tp += 1
            else:
                tn += 1
        
        return 100 * tp / (tp + tn)


