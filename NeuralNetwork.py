"""
Neural Network Concept
"""
import numpy as np 
from scipy import special


class Utils:

	"""
	Collection of useful functions
	"""
	
	@staticmethod
	def sigmoid(x: float) -> float:
		"""
		Sigmoid Activation Function (Logistic Growth)

		sigma(x) = 1 / (1 + exp(-x))
		sigma'(x) = sigma(x) * (1 - sigma(x))
		
		Args:
		    x (float): x
		
		Returns:
		    float: sigmoid(x)
		"""
		return special.expit(x)

	@staticmethod
	def tanh(x: float) -> float:
		"""
		Tangens Hyperbolicus Activation Function

		tanh(x) = 1 - 2 / (exp(2x) + 1)
		tanh'(x) = 1 / cosh(x)^2
		
		Args:
		    x (float): x
		
		Returns:
		    float: tanh(x)
		"""
		return np.tanh(x)


class Layer(object):

	"""
	Layer Class, contains basic layer structure
	
	Attributes:
	    activation_function (function): Activation Function
	    dimension (int): Number of Perceptrons
	    error (list): Error Vector (Gradient)
	    id (int): Position of Layer in Neural Network
	    input (list): Input Vector
	    output (list): Output Vector
	"""
	
	def __init__(self, dimension: int, id: int, *args, **kwargs):
		"""
		Initialize Layer
		
		Args:
		    dimension (int): Number of Perceptrons
		    id (int): Position of Layer in Neural Network
		    *args: Arguments
		    **kwargs: Keyword Arguments
		"""
		self.dimension = dimension
		self.activation_function = lambda x: Utils.sigmoid(x)

		self.id = id

		self.input = []
		self.output = []
		self.error = []

	def __repr__(self):
		"""
		String Representation
		
		Returns:
		    str: Description
		"""
		return f"Layer object: {self.dimension} Perceptrons"

	def process(self, input: np.array, weight_matrix: np.array):
		"""
		Process input data
		
		Args:
		    input (np.array): Input Vector
		    weight_matrix (np.array): Weight Matrix
		"""
		self.input = input
		self.output = self.activation_function(np.dot(weight_matrix, input))


class WeightMatrix(object):

	"""
	Weight Matrix Class, contains the weights between two layers
	
	Attributes:
	    dimensions (tuple): Dimensions of Matrix
	    matrix (np.array): Weight Matrix
	"""
	
	def __init__(self, dimensions: tuple, *args, **kwargs):
		"""
		Initialize Weight Matrix
		
		Args:
		    dimensions (tuple): Dimensions of Matrix
		    *args: Arguments
		    **kwargs: Keyword Arguments
		"""
		self.dimensions = dimensions
		self.matrix = np.random.normal(0.0, pow(self.dimensions[1], -0.5), self.dimensions)

	def __repr__(self):
		"""
		String Representation
		
		Returns:
		    str: Description
		"""
		return f"WeightMatrix object: {self.dimensions[0]} x {self.dimensions[1]} Weights"


class NeuralNetwork(object):

	"""
	Neural Network Class, contains the Layers, Weights and Learning Algorithms
	
	Attributes:
	    layer_dimensions (list): List of Perceptrons per Layer
	    layers (list): List of Layers
	    learning_rate (float): Learning Rate
	    num_layers (int): Number of Layers
	    weight_matrices (list): List of Weight Matrices
	"""
	
	def __init__(self, layer_dimensions: list, learning_rate: int, *args, **kwargs):
		"""
		Initialize Neural Network
		
		Args:
		    layer_dimensions (list): List of Perceptrons per Layer
		    learning_rate (int): Learning Rate
		    *args: Arguments
		    **kwargs: Keyword Arguments
		"""
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
		"""
		String Representation
		
		Returns:
		    str: Description
		"""
		return f"NeuralNetwork object: {self.num_layers} Layers"

	def train(self, input: np.array, target: np.array):
		"""
		Train the Neural Network with labelled input data
		
		Args:
		    input (np.array): Input Vector
		    target (np.array): Label Vector
		"""
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
		"""
		Process an input vector without label
		
		Args:
		    input (np.array): Input Vector
		
		Returns:
		    np.array: Output Vector
		"""
		input = np.array(input, ndmin=2).T
		
		self.layers[0].output = input

		for layer in range(1, self.num_layers):
			self.layers[layer].process(self.layers[layer - 1].output, self.weight_matrices[layer - 1].matrix)

		return self.layers[-1].output


if __name__ == '__main__':
	
	pass
