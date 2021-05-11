import numpy as np

from .Utils import ActivationFunctions


activation_functions = {"sigmoid": [ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_derivative],
						"tanh": [ActivationFunctions.tanh, ActivationFunctions.tanh_derivative],
						"relu": [ActivationFunctions.relu, ActivationFunctions.relu_derivative]}


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
	
	def __init__(self, dimension: int, id: int, activation_function: str, *args, **kwargs):
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
		self._activation_function, self._activation_function_derivative = activation_functions[activation_function]

		self.input = []
		self.output = []
		self.error = []

	def __repr__(self) -> str:
		"""
		String Representation
		
		Returns:
		    str: Description
		"""
		return f"Layer object: {self.dimension} Perceptrons"

	def activation_function(self, x: np.array) -> np.array:
		"""
		Activation Function of the layers' perceptrons
		
		Args:
		    x (np.array): Processed data
		
		Returns:
		    np.array: Layers' output
		"""
		return self._activation_function(x)

	def process(self, input: np.array, weight_matrix: np.array):
		"""
		Process input data
		
		Args:
		    input (np.array): Input Vector
		    weight_matrix (np.array): Weight Matrix
		"""
		self.input = input
		self.output = self.activation_function(np.dot(weight_matrix, input))