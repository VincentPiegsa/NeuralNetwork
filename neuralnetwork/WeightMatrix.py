import numpy as np 


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
