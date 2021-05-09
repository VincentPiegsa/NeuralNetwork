import numpy as np 
from scipy import special


class Utils:

	@staticmethod
	def sigmoid(x: float) -> float:

		return special.expit(x)

	@staticmethod
	def tanh(x: float) -> float:

		return np.tanh(x)


class Layer(object):

	def __init__(self, dimension: int, id: int, *args, **kwargs):

		self.dimension = dimension
		self.activation_function = lambda x: Utils.sigmoid(x)

		self.id = id

		self.input = []
		self.output = []
		self.error = []

	def __repr__(self):

		return f"Layer object: {self.dimension} Perceptrons"

	def process(self, input, weight_matrix):

		self.input = input
		self.output = self.activation_function(np.dot(weight_matrix, input))


class WeightMatrix(object):

	def __init__(self, dimensions: tuple, *args, **kwargs):

		self.dimensions = dimensions
		self.matrix = np.random.normal(0.0, pow(self.dimensions[1], -0.5), self.dimensions)

	def __repr__(self):

		return f"WeightMatrix object: {self.dimensions[0]} x {self.dimensions[1]} Weights"


class NeuralNetwork(object):

	def __init__(self, layer_dimensions: list, learning_rate: int, *args, **kwargs):
		
		self.layer_dimensions = layer_dimensions
		self.num_layers = len(self.layer_dimensions)

		self.layers = []
		self.weight_matrices = []

		for layer in range(self.num_layers):
			self.layers.append(Layer(self.layer_dimensions[layer], layer))

		for weight_matrix in range(self.num_layers - 1):
			dimensions = (self.layers[weight_matrix + 1].dimension, self.layers[weight_matrix].dimension)
			self.weight_matrices.append(WeightMatrix(dimensions))

		self.learning_rate = learning_rate

	def __repr__(self):

		return f"NeuralNetwork object: {self.num_layers} Layers"

	def train(self, input: np.array, target: np.array):

		input = np.array(input, ndmin=2).T
		target = np.array(target, ndmin=2).T
		
		self.layers[0].output = input

		for layer in range(1, self.num_layers):
			self.layers[layer].process(self.layers[layer - 1].output, self.weight_matrices[layer - 1].matrix)

		self.layers[-1].error = (target - self.layers[-1].output)

		for layer in range(self.num_layers - 2, -1, -1):
			self.layers[layer].error = np.dot(self.weight_matrices[layer].matrix.T, self.layers[layer + 1].error)

		for layer in range(self.num_layers - 1):
			self.weight_matrices[layer].matrix += self.learning_rate * np.dot((self.layers[layer + 1].error * self.layers[layer + 1].output * (1 - self.layers[layer + 1].output)), self.layers[layer].output.T)

	def query(self, input: np.array) -> np.array:

		input = np.array(input, ndmin=2).T
		
		self.layers[0].output = input

		for layer in range(1, self.num_layers):
			self.layers[layer].process(self.layers[layer - 1].output, self.weight_matrices[layer - 1].matrix)

		return self.layers[-1].output


if __name__ == '__main__':
	
	pass
