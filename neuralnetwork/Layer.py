import numpy as np

from .Utils import Utils


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

	def activation_function(self, x):

		return Utils.sigmoid(x)

	def process(self, input: np.array, weight_matrix: np.array):
		"""
		Process input data
		
		Args:
		    input (np.array): Input Vector
		    weight_matrix (np.array): Weight Matrix
		"""
		self.input = input
		self.output = self.activation_function(np.dot(weight_matrix, input))