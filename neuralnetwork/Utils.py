import numpy as np
from scipy import special


class ActivationFunctions:

	"""
	Collection of useful functions
	"""
	
	@staticmethod
	def sigmoid(x: float) -> float:
		"""
		Sigmoid Activation Function (Logistic Growth)

		sigma(x) = 1 / (1 + exp(-x))
		
		Args:
		    x (float): x
		
		Returns:
		    float: sigmoid(x)
		"""
		return special.expit(x)

	@staticmethod
	def sigmoid_derivative(x: float) -> float:
		"""
		Derivative of Sigmoid Function

		sigma'(x) = sigma(x) * (1 - sigma(x))
		
		Args:
		    x (float): x
		
		Returns:
		    float: d / dx sigmoid(x)
		"""
		return special.expit(x) * (1 - special.expit(x))

	@staticmethod
	def tanh(x: float) -> float:
		"""
		Tangens Hyperbolicus Activation Function

		tanh(x) = 1 - 2 / (exp(2x) + 1)
		
		Args:
		    x (float): x
		
		Returns:
		    float: tanh(x)
		"""
		return np.tanh(x)

	@staticmethod
	def tanh_derivative(x: float) -> float:
		"""
		Derivative of Tangens Hyperbolicus

		tanh'(x) = 1 / cosh(x)^2
		
		Args:
		    x (float): x
		
		Returns:
		    float: d / dx tanh(x)
		"""
		return 1 / (np.cosh(x))**2

	@staticmethod
	def relu(x: float) -> float:
		"""
		Rectified Linear Unit (ReLU) Activation Function
		
		Args:
		    x (float): x
		
		Returns:
		    float: ReLU(x)
		"""
		return x if (x > 0) else 0

