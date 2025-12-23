import numpy as np
import os
import pickle

module_dir = os.path.dirname(__file__)
KNN_FILENAME = os.path.join(module_dir, "knn_weights.pkl")


class ml_model:
    """
    K-Nearest Neighbors classifier with the same interface as torch_nn and mlp_custom.
    Note: The class is named 'mlp' to maintain compatibility with existing code.
    """
    
    def __init__(self, hidden_sizes=None, solver="adam"):
        """
        Initialize KNN classifier.
        
        Args:
            hidden_sizes: Not used in KNN, kept for interface compatibility
            solver: Not used in KNN, kept for interface compatibility
        """
        # Interface compatibility parameters (not used in KNN)
        self.hidden_sizes = hidden_sizes
        self.solver = solver
        
        # KNN specific parameters
        self.k = 5 if hidden_sizes is None else hidden_sizes[0] if isinstance(hidden_sizes, list) else hidden_sizes
        self.X_train = None
        self.y_train = None
        
        # Common interface parameters
        self.gesture_detected_index = 0
        self.output_size = 0
        self.input_size = 0
        self.final_net_error = 0
        self.input = []
        
        # Training history (minimal for KNN)
        self.epochs = []
        self.loss_history = []
        self.Loss = []
        
        # Flags
        self.Layers_initialized = False
        self.layer_initialized = False
        
    def create_layers(self, input_size, output_size):
        """
        Initialize KNN with data dimensions.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.Layers_initialized = True
        
    def init_layer_weights(self, input_sample, target_sample):
        """
        Initialize with data dimensions (compatibility method).
        """
        self.input_size = len(input_sample)
        self.output_size = len(target_sample)
        self.layer_initialized = True
        
    def input_change(self, input_data):
        """
        Set the current input for prediction.
        """
        self.input = input_data
        
    def _euclidean_distance(self, x1, x2):
        """
        Calculate Euclidean distance between two points.
        """
        return np.sqrt(np.sum((np.array(x1) - np.array(x2)) ** 2))
    
    def _get_neighbors(self, test_sample):
        """
        Find k nearest neighbors for a test sample.
        
        Returns:
            List of tuples (distance, label_index)
        """
        distances = []
        for i, train_sample in enumerate(self.X_train):
            dist = self._euclidean_distance(test_sample, train_sample)
            label_index = np.argmax(self.y_train[i])
            distances.append((dist, label_index))
        
        # Sort by distance and get k nearest
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.k]
        
        return neighbors
    
    def _majority_vote(self, neighbors):
        """
        Get the majority class from neighbors.
        
        Returns:
            Predicted class probabilities (one-hot encoded style)
        """
        # Count votes for each class
        votes = np.zeros(self.output_size)
        for _, label_index in neighbors:
            votes[label_index] += 1
        
        # Convert to probabilities
        probs = votes / self.k
        return probs
    
    def Train(self, input_data, target_data, max_epoch=1, lr=0.001):
        """
        'Train' the KNN model (just stores the training data).
        
        Args:
            input_data: Training input samples
            target_data: Training target labels (one-hot encoded)
            max_epoch: Not used in KNN, kept for interface compatibility
            lr: Not used in KNN, kept for interface compatibility
        """
        if not self.layer_initialized:
            self.init_layer_weights(input_data[0], target_data[0])
            self.layer_initialized = True
            
        # Store training data
        self.X_train = np.array(input_data)
        self.y_train = np.array(target_data)
        
        # Initialize dimensions
        self.input_size = len(input_data[0])
        self.output_size = len(target_data[0])
        
        # Simulate training for interface compatibility
        self.epochs = list(range(max_epoch))
        self.Loss = [0.0] * max_epoch  # KNN doesn't have a loss function
        self.loss_history = self.Loss.copy()
        self.final_net_error = 0.0
        
        print(f"KNN classifier 'trained' with {len(input_data)} samples, k={self.k}")
        
        self.Layers_initialized = True
        
    def Validate(self, input_data, target_data, max_epoch=1, lr=0.001):
        """
        Validate the KNN model by calculating accuracy.
        
        Args:
            input_data: Validation input samples
            target_data: Validation target labels (one-hot encoded)
            max_epoch: Not used in KNN, kept for interface compatibility
            lr: Not used in KNN, kept for interface compatibility
        """
        accuracy = self.get_acc(input_data, target_data)
        
        # Simulate validation for interface compatibility
        self.epochs = list(range(max_epoch))
        self.Loss = [100.0 - accuracy] * max_epoch  # Use error as pseudo-loss
        self.loss_history = self.Loss.copy()
        self.final_net_error = 100.0 - accuracy
        
        print(f"Validation Accuracy: {accuracy:.2f}%")
        
    def predict(self):
        """
        Predict class probabilities for the current input.
        
        Returns:
            Class probabilities as numpy array
        """
        if self.X_train is None:
            raise ValueError("Model not trained. Call Train() first.")
        
        if len(self.input) == 0:
            raise ValueError("No input data. Call input_change() first.")
        
        # Get k nearest neighbors
        neighbors = self._get_neighbors(self.input)
        
        # Get prediction probabilities
        probs = self._majority_vote(neighbors)
        
        # Update gesture detected index
        self.gesture_detected_index = np.argmax(probs)
        
        return probs
    
    def get_acc(self, input_data, target_data):
        """
        Calculate accuracy on the given input and target data.
        
        Args:
            input_data: Input samples
            target_data: Target labels (one-hot encoded)
            
        Returns:
            Accuracy as a percentage
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
    
    def disp(self):
        """
        Display the current prediction.
        """
        probs = self.predict()
        print(probs.tolist())
        
    def save_weights(self, dynamic=False):
        """
        Save the KNN model (training data and parameters).
        
        Args:
            dynamic: If True, save to dynamic weights file
        """
        filename = KNN_FILENAME.replace(".pkl", "_dynamic.pkl") if dynamic else KNN_FILENAME
        
        data = {
            'k': self.k,
            'X_train': self.X_train,
            'y_train': self.y_train,
            'input_size': self.input_size,
            'output_size': self.output_size
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"KNN model saved to {filename}")
        
    def load_weights(self, dynamic=False):
        """
        Load the KNN model (training data and parameters).
        
        Args:
            dynamic: If True, load from dynamic weights file
        """
        filename = KNN_FILENAME.replace(".pkl", "_dynamic.pkl") if dynamic else KNN_FILENAME
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found: {filename}")
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.k = data['k']
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.input_size = data['input_size']
        self.output_size = data['output_size']
        
        self.Layers_initialized = True
        self.layer_initialized = True
        
        print(f"KNN model loaded from {filename}")
        
    def forward(self, x):
        """
        Forward pass (compatibility method for PyTorch-style interface).
        """
        self.input_change(x)
        return self.predict()
