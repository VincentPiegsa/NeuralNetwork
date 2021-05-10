import numpy as np 
import pickle

from .Layer import Layer
from .WeightMatrix import WeightMatrix


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
			
	def batch_train(self, inputs: list, targets: list):
		"""
		Train the Neural Network with batches of labelled input data
		
		Args:
		    input (list): List with Input Vectors
		    target (list): List witch Label Vectors
		"""

		batch_size = len(inputs)
		cumulative_error = []

		for layer in self.layers:
			cumulative_error.append(np.zeros((layer.dimension, 1), dtype=np.float64))

		for input, target in zip(inputs, targets):

			input = np.array(input, ndmin=2).T
			target = np.array(target, ndmin=2).T
			
			self.layers[0].output = input

			for layer in range(1, self.num_layers):
				self.layers[layer].process(self.layers[layer - 1].output, self.weight_matrices[layer - 1].matrix)

			self.layers[-1].error = (target - self.layers[-1].output)

			for layer in range(self.num_layers - 2, -1, -1):
				self.layers[layer].error = np.dot(self.weight_matrices[layer].matrix.T, self.layers[layer + 1].error)
				cumulative_error[layer] += self.layers[layer].error

		for layer in range(self.num_layers - 1):
			self.weight_matrices[layer].matrix += self.learning_rate * np.dot(((cumulative_error[layer + 1] / batch_size) * self.layers[layer + 1].output * (1 - self.layers[layer + 1].output)), self.layers[layer].output.T)
			

	def query(self, input: np.array) -> np.array:
		"""
		Process an input vector without label and return the last layers output
		
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

	def query_probabilistic(self, input: np.array) -> np.array:
		"""
		Process an input vector without label and return an array of relative certainty (probability)
		
		Args:
		    input (np.array): Input Vector
		
		Returns:
		    np.array: Array of relative certainty
		"""

		output = self.query(input)
		total = sum(output)

		return output / total
