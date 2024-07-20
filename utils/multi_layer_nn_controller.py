import numpy as np

class NeuralNetwork:
	def __init__(self, layers):
		self.layers = layers
		self.weights = []
		self.biases = []

        # Initialize weights and biases with random values for demonstration
		for i in range(len(layers) - 1):
			weight = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2. / layers[i])
			bias = np.zeros((1, layers[i + 1]))
			self.weights.append(weight)
			self.biases.append(bias)
		print(f"weights: {self.weights}")
		print(f"biases: {self.biases}")

	def relu(self, x):
		return np.maximum(0, x)

	def forward(self, input_data):
		"""
		Perform a forward pass through the network using ReLU activations.

		Args:
		input_data (np.array): The input data array where each row is a data point.

		Returns:
		np.array: Output from the network after the forward pass.
		"""
		activation = input_data
		for weight, bias in zip(self.weights, self.biases):
			z = np.dot(activation, weight) + bias
			activation = self.relu(z)
		return activation

# Example usage
#~ nn = NeuralNetwork([138, 64, 64, 2])  # Define the network architecture
nn = NeuralNetwork([3, 2])  # Define the network architecture
X_sample = np.random.rand(1, 3)  # Generate a random sample input - Sensors

output = nn.forward(X_sample)
print("Output of the neural network:", output)
