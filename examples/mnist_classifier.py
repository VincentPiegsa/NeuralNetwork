from neuralnetwork.NeuralNetwork import NeuralNetwork

from keras.datasets import mnist
import numpy as np


if __name__ == '__main__':
	
	nn = NeuralNetwork([784, 100, 100, 10], 5e-2)

	scorecard = []
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	for record, label in zip(x_train, y_train):

    # prescale the training data and labels
		record = np.array(record).flatten('C')
		inputs = np.asfarray(record / 255.0 * (1 - 1e-3)) + 1e-3

		targets = np.zeros(10) + 1e-3
		targets[label] = 1 - 1e-3

    # train the neural network with the data
		nn.train(inputs, targets)

	for record, label in zip(x_test, y_test):

    # prescale the testing data and labels
		record = np.array(record).flatten('C')
		inputs = np.asfarray(record / 255.0 * (1 - 1e-3)) + 1e-3

    # query the neural network
		output = nn.query(inputs)
		prediction = np.argmax(output)

    # check whether the network's prediction was correct
		if prediction == label:
			scorecard.append(1)
		else:
			scorecard.append(0)

  # calculate the final accuracy of the classifier
	accuracy = sum(scorecard) / len(scorecard) * 100

	print("-"*20 + "\n\n")
	print(f"Accuracy of {accuracy:.2f}%")
	print("\n\n" + "-"*20)
