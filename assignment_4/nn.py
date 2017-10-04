import numpy as np


class NN():
    """
    Class for neural network
    """

    def __init__(self, neuron_layers, lr=0.01):
        self.neuron_layers = neuron_layers
        self.number_of_layers = len(neuron_layers)
        self.weights = []
        self.biases = []
        self.learning_rate = lr

        # Initialize weights
        self.initialize_weights()

    def activation_function(self, x):
        """
        The activation function to use.
        For now let's use sigmoid as activation function
        """
        return 1.0 / (1.0 + np.exp(-x))

    def initialize_weights(self):
        """
        Initial weights
        """

        for i in range(1, self.number_of_layers):
            current_layer_neurons = self.neuron_layers[i]
            previous_layer_neurons = self.neuron_layers[i-1]

            # Create a numpy array to hold weights
            # Randomly initialize weights using uniform distribution in the
            # range [-0.5, 0.5]
            layer_weights = np.random.rand(
                previous_layer_neurons, current_layer_neurons) - 0.5

            # Store layer weights
            self.weights.append(layer_weights)

            # Randomly initialize biases in range [-0.5, 0.5]
            random_biases = np.random.rand(current_layer_neurons, 1) - 0.5
            self.biases.append(random_biases)

    def predict(self, x):
        """
        Feed forward the values to obtain the predicted value
        """
        x = np.array(x)

        # Set layer_activation to x initially
        layer_activation = x
        for layer in range(1, self.number_of_layers):
            layer_index = layer - 1

            # Obtain weights and biases
            weights = self.weights[layer_index]
            biases = self.biases[layer_index]

            # Assuming layer_activation is a row vector with dimension 1*m
            # Perform matrix multiplication and add bias
            z = np.dot(layer_activation, weights) + biases.T

            # Use the activation function
            layer_activation = self.activation_function(z)

        # Return activation obtained from final layer
        return layer_activation.T

    def cost_function(self, y_pred, y_target):
        """
        Calculate the cost function
        """

        # Total squared error
        squared_error = np.linalg.norm(y_target - y_pred)

    def train(self, input_list, output_list):
        """
        Train the neural network
        """

        input_array = np.array(input_list)
        output_array = np.array(output_list)

        # Keep training until stopping criteria is met
        for i in range(100000):
            for input_sample, output_sample in zip(input_array, output_array):
                activations = [input_sample.reshape(1, len(input_sample))]

                # Feed forward through the layers
                for layer in range(1, self.number_of_layers):
                    layer_index = layer - 1

                    # Obtain weights and biases
                    weights = self.weights[layer_index]
                    biases = self.biases[layer_index]

                    # Assuming layer_activation is a row vector with dimension
                    # 1*m
                    # Perform matrix multiplication and add bias
                    z = np.dot(activations[layer_index], weights) + biases.T

                    # Use the activation function
                    layer_activation = self.activation_function(z)

                    # Append this layers activation
                    activations.append(layer_activation)

                predicted_output = activations[-1]
                delta = (output_sample - predicted_output) * np.multiply(
                    predicted_output,
                    (1.0 - predicted_output)
                )

                # Update weights
                self.weights[-1] += self.learning_rate * delta * activations[-2].T
                self.biases[-1] += self.learning_rate * delta

                # Backpropagate delta through layers
                for layer in range(self.number_of_layers-2, 0, -1):

                    layer_index = layer - 1

                    layer_activation = activations[layer_index+1]

                    delta = np.dot(self.weights[layer_index+1], delta) * np.multiply(
                        layer_activation.T,
                        (1.0 - layer_activation.T)
                    )

                    self.weights[layer_index] += self.learning_rate * np.dot(
                        activations[layer_index].T,
                        delta.T
                    )
                    self.biases[layer_index] += self.learning_rate * delta


def main():
    # Create a neural network model
    model = NN([2, 10, 1])
    print(model.predict([0, 0]))
    print(model.predict([1, 0]))
    print(model.predict([0, 1]))
    print(model.predict([1, 1]))
    model.train([[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]])
    print(model.predict([0, 0]))
    print(model.predict([1, 0]))
    print(model.predict([0, 1]))
    print(model.predict([1, 1]))


if __name__ == "__main__":
    main()
