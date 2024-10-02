import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class SignLanguageModel:
    def __init__(self, model_path, x_test, y_test):
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        self.x_test = x_test
        self.y_test = y_test

    def get_true_label(self, index):
        """Return the true label (as an integer) for the given index."""
        return np.argmax(self.y_test[index])

    def predict_label(self, index):
        """Predict the label for the given test image."""
        prediction = self.model.predict(np.expand_dims(self.x_test[index], axis=0))
        return np.argmax(prediction)

    def visualize_prediction(self, index):
        """Visualize the true and predicted label for a given index."""
        # Get the true and predicted labels
        true_label = self.get_true_label(index)
        predicted_label = self.predict_label(index)
        
        # Plot the image with true and predicted labels
        plt.imshow(self.x_test[index])  # Display the 400x400 white image with landmarks
        plt.title(f"True Label: {true_label}, Predicted Label: {predicted_label}")
        
        # Save the plot as an image file before showing it
        plt.savefig('label_Accuracy1.png')
        print("Plot saved as 'label_Accuracy.png'")
        
        # Show the plot
        plt.show()
        
        # Calculate accuracy for the specific index
        accuracy = accuracy_score([true_label], [predicted_label])
        print(f"Accuracy for label at index {index}: {accuracy * 100:.2f}%")

# Example usage:
if __name__ == "__main__":
    # Load the test data (from validation set)
    X_val = np.load('X_val.npy')  # Load 400x400 white images with landmarks
    y_val = np.load('y_val.npy')  # Load corresponding labels

    # Create an instance of the SignLanguageModel class
    model_path = 'ASLMODEL.h5'
    sl_model = SignLanguageModel(model_path, X_val, y_val)

    # Test for a specific index, e.g., 0, 1, 2, or any index you'd like
    index = 9  # Replace with the index you want to test
    sl_model.visualize_prediction(index)
