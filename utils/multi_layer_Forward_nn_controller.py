import numpy as np

class ForwardNeuralNetwork:
    def __init__(self, params, num_nn_neurons, inputs_size):
        """
        Initialize the neural network with predefined weights and biases.
        """

        ## Extract the number of neurons for each layer
        num_input_neurons = num_nn_neurons[0]
        hidden_layer1_neurons = num_nn_neurons[1]
        hidden_layer2_neurons = num_nn_neurons[2]
        output_neurons = num_nn_neurons[3]

        ## Get the location of weights and biases in the genotype
        input_layer_cut = inputs_size * num_input_neurons
        hidden_layer1_cut = input_layer_cut + (num_input_neurons * hidden_layer1_neurons)
        hidden_layer2_cut = hidden_layer1_cut + (hidden_layer1_neurons * hidden_layer2_neurons)
        output_layer_cut = hidden_layer2_cut + (hidden_layer2_neurons * output_neurons)

        input_bias = output_layer_cut + num_input_neurons
        hidden_layer1_bias = input_bias + hidden_layer1_neurons
        hidden_layer2_bias = hidden_layer1_bias + hidden_layer2_neurons
        # output_bias_cut = hidden_layer2_bias + output_neurons

        self.input_weights = np.array(params[:input_layer_cut]).reshape(inputs_size, num_input_neurons)
        self.hidden1_weights = np.array(params[input_layer_cut:hidden_layer1_cut]).reshape(num_input_neurons, hidden_layer1_neurons)
        self.hidden2_weights = np.array(params[hidden_layer1_cut:hidden_layer2_cut]).reshape(hidden_layer1_neurons, hidden_layer2_neurons)
        self.output_weigths = np.array(params[hidden_layer2_cut:output_layer_cut]).reshape(hidden_layer2_neurons, output_neurons)

        self.input_bias = np.array(params[output_layer_cut:input_bias]).reshape(1, num_input_neurons)
        self.hidden1_bias = np.array(params[input_bias:hidden_layer1_bias]).reshape(1, hidden_layer1_neurons)
        self.hidden2_bias = np.array(params[hidden_layer1_bias:hidden_layer2_bias]).reshape(1, hidden_layer2_neurons)
        self.output_bias = np.array(params[hidden_layer2_bias:]).reshape(1, output_neurons)

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
        z1 = np.dot(input_data, self.input_weights) + self.input_bias
        a1 = self.relu(z1)

        # Second layer processing
        z2 = np.dot(a1, self.hidden1_weights) + self.hidden1_bias
        a2 = self.relu(z2)

        # Third layer processing
        z3 = np.dot(a2, self.hidden2_weights) + self.hidden2_bias
        a3 = self.relu(z3)

        # Output layer processing
        z4 = np.dot(a3, self.output_weigths) + self.output_bias
        output = self.relu(z4)

        return output

#~ # Example usage
#~ num_nn_neurons = [5, 3, 5, 2]
#~ inputs_size = 68
#~ genotype_size = (inputs_size * num_nn_neurons[0]) + (num_nn_neurons[0] * num_nn_neurons[1]) + (num_nn_neurons[1] * num_nn_neurons[2]) + (num_nn_neurons[2] * num_nn_neurons[3]) +\
                #~ num_nn_neurons[0] + num_nn_neurons[1] + num_nn_neurons[2] + num_nn_neurons[3]

#~ params = np.random.rand(genotype_size)

#~ nn = ForwardNeuralNetwork(params, num_nn_neurons, inputs_size)

#~ X_sample = np.random.rand(1, 68)  # Example input
#~ output = nn.forward(X_sample)
#~ print("Output of the neural network:", output)
