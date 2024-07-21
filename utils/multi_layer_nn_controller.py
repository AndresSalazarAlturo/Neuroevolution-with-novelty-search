import numpy as np

class ForwardNeuralNetwork:
    def __init__(self, params, num_input_neurons, output_size, inputs_size):
        """
        Initialize the neural network with predefined weights and biases.

        Args:
        weights1 (np.array): Weights for the first layer, shape (68, 3).
        biases1 (np.array): Biases for the first layer, shape (3,).
        weights2 (np.array): Weights for the second layer, connecting the 3 neurons to 2 outputs, shape (3, 2).
        biases2 (np.array): Biases for the second layer, shape (2,).
        """

        weights1_cut = inputs_size * num_input_neurons
        weights2_cut = weights1_cut + (num_input_neurons * output_size)
        biases1_cut = weights2_cut + num_input_neurons

        self.weights1 = np.array(params[:weights1_cut]).reshape(inputs_size, num_input_neurons)
        self.weights2 = np.array(params[weights1_cut:weights2_cut]).reshape(num_input_neurons, output_size)
        self.biases1 = np.array(params[weights2_cut:biases1_cut]).reshape(1, num_input_neurons)
        self.biases2 = np.array(params[biases1_cut:]).reshape(1, output_size)

        #~ print(f"biases 2: {self.biases2}")

    def relu(self, x):
        """
        ReLU activation function for non-linear transformation.
        """
        return np.maximum(0, x)

    def forward(self, input_data):
        """
        Perform a forward pass through the network using two layers.
        """
        # First layer processing
        z1 = np.dot(input_data, self.weights1) + self.biases1
        a1 = self.relu(z1)

        # Second layer processing
        z2 = np.dot(a1, self.weights2) + self.biases2
        output = self.relu(z2)  # Assuming ReLU activation for the output layer as well
        return output

#~ # Example usage
#~ num_input_neurons = 3
#~ output_size = 2
#~ inputs_size = 68
#~ params = np.random.rand(215)

#~ nn = SimpleNeuralNetwork(params, num_input_neurons, output_size, inputs_size)

#~ X_sample = np.random.rand(1, 68)  # Example input
#~ output = nn.forward(X_sample)
#~ print("Output of the neural network:", output)
