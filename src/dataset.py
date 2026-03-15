import numpy as np
import csv

class DatasetLoader:
    def __init__(self, filepath):
        """Initializes the loader with the path to the Iris dataset CSV."""
        self.filepath = filepath
        self.X = None
        self.y = None
        self.classes = None

    def load_csv(self):
        """Loads the CSV file, separates features from labels."""
        features = []
        labels = []
        
        with open(self.filepath, 'r') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip the header row if your CSV has one
            
            for row in reader:
                if not row: continue # Skip empty rows
                # First 4 columns are features (sepal/petal length/width)
                features.append([float(x) for x in row[:-1]])
                # Last column is the species label
                labels.append(row[-1])
                
        self.X = np.array(features)
        
        # Convert string labels (e.g., 'Iris-setosa') to integer indices
        self.classes = list(set(labels))
        self.y = np.array([self.classes.index(label) for label in labels])
        
        print(f"Loaded {self.X.shape[0]} samples with {self.X.shape[1]} features.")

    def one_hot_encode(self):
        """Converts integer labels to one-hot encoded vectors for the neural network."""
        num_classes = len(self.classes)
        one_hot_y = np.zeros((self.y.size, num_classes))
        one_hot_y[np.arange(self.y.size), self.y] = 1
        self.y = one_hot_y

    def normalize(self):
        """Applies Standard Scaling (Z-score normalization) to the features."""
        # Formula: z = (x - mean) / standard_deviation
        mean = np.mean(self.X, axis=0)
        std = np.std(self.X, axis=0)
        
        # Add a small epsilon to prevent division by zero just in case
        self.X = (self.X - mean) / (std + 1e-8)
        print("Features normalized.")

    def train_test_split(self, test_ratio=0.2, random_seed=None):
        """Splits the dataset into training and testing sets."""
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Create a shuffled array of indices
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        
        # Determine the split index
        test_size = int(self.X.shape[0] * test_ratio)
        
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        
        X_train, X_test = self.X[train_indices], self.X[test_indices]
        y_train, y_test = self.y[train_indices], self.y[test_indices]
        
        print(f"Split data into {X_train.shape[0]} training and {X_test.shape[0]} testing samples.")
        return X_train, X_test, y_train, y_test