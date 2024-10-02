import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import cv2
import os
from cv2 import resize
from random import shuffle

class SignLanguageModel:
    def __init__(self, data_path='DataSet', img_size=400, num_classes=26):
        self.data_path = data_path
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
    
    def load_data(self):
        """Load and preprocess the image data from the directory."""
        training_data = []
        subdirs = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # 'A' to 'Z'
        
        for label, subdir in enumerate(subdirs):
            subdir_path = os.path.join(self.data_path, subdir)
            if not os.path.exists(subdir_path):
                print(f"Warning: Directory {subdir_path} does not exist. Skipping.")
                continue

            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                img = cv2.imread(file_path)
                if img is None:
                    print(f"Warning: Could not read image {file_path}. Skipping.")
                    continue
                
                # Ensure image is resized
                try:
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    training_data.append([np.array(img), label])  # Append the image and the corresponding label
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
        
        print(f"Loaded {len(training_data)} samples in total.")
        shuffle(training_data)
        return training_data

    def prepare_data(self, training_data):
        """Prepare the data for training."""
        X = []
        y = []
        for features, label in training_data:
            resized_features = resize(features, (200, 200))  # Resize to 200x200
            X.append(resized_features)
            y.append(label)
        
        X = np.array(X).reshape(-1, 200, 200, 3)  # Reshape for RGB image
        y = np.array(y)
        
        print(f"Total samples: X.shape = {X.shape}, y.shape = {y.shape}")
        
        # Normalize the data
        X = X.astype('float32') / 255.0  # Use float32 instead of float64 to reduce memory usage
        
        # Convert labels to categorical
        y = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)
        
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=100)
        
        return X_train, X_val, y_train, y_val

    def build_model(self):
        """Build the CNN model."""
        self.model = Sequential()

        # First convolutional block
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
        self.model.add(MaxPooling2D((2, 2)))

        # Second convolutional block
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))

        # Third convolutional block
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))

        # Flattening the output and passing to the dense layers
        self.model.add(Flatten())  # This will now output the correct shape
        
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))  # Helps reduce overfitting
        self.model.add(Dense(self.num_classes, activation='softmax'))  # Output layer for 26 classes (A-Z)
        self.model.summary()

    def compile_model(self):
        """Compile the CNN model."""
        self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the CNN model without using data augmentation."""
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val)
        )
        return self.history

    def save_model(self, file_path='ASLMODEL.h5'):
        """Save the trained model to a file."""
        self.model.save(file_path)
        print(f'Model saved to {file_path}')

    def save_data(self, X_train, y_train, X_val, y_val, history_file='ASLMODEL.npy'):
        """Save the data and training history to files."""
        np.save('X_train.npy', X_train)
        np.save('Y_train.npy', y_train)
        np.save('X_val.npy', X_val)
        np.save('Y_val.npy', y_val)
        np.save(history_file, self.history.history)
        print(f'Training data, validation data, and training history saved.')

    def run(self):
        """Run the entire process: load data, prepare it, train the model, and save results."""
        training_data = self.load_data()  # Load the data
        X_train, X_val, y_train, y_val = self.prepare_data(training_data)  # Prepare the data
        self.build_model()  # Create the model
        self.compile_model()  # Compile the model
        self.train_model(X_train, y_train, X_val, y_val)  # Train the model
        self.save_data(X_train, y_train, X_val, y_val)  # Save the data and history
        self.save_model()  # Save the model

if __name__ == "__main__":
    sl_model = SignLanguageModel()
    sl_model.run()
