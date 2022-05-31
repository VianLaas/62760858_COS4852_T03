import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class FeedForwardNetwork:
	def __init__(self, training_data, learning_rate = 0.05, max_epochs=5000, num_hidden = 2):
		self.training_data = training_data
		self.learning_rate = learning_rate
		self.max_epochs = max_epochs
		self.num_input = len(training_data[0][0])
		self.num_hidden = num_hidden
		self.num_output = len(training_data[0][1])

		# Error terms for statistical purposes
		self.errors = []

		# Inititialise weights between layers with random values in [-0.05, 0.05]
		self.input_to_hidden_weights = np.zeros((self.num_hidden, self.num_input))
		self.hidden_to_output_weights = np.zeros((self.num_output, self.num_hidden))

	def _sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def _feed(self, input):
		# The outputs for the input layer remain the same as the inputs themselves
		outputs = input

		# Start with an empty, intermediary list for hidden outputs
		hidden_outputs = []

		# The outputs for the hidden layer are the dot products of the input layer's outputs
		# and their weights
		for weights in self.input_to_hidden_weights:
			hidden_outputs.append(self._sigmoid(np.dot(input, weights)))

		# Add the inputs to the hidden outputs
		outputs = np.append(input, hidden_outputs)

		# The outputs for the output layer are the dot products of the hidden layer's outputs
		# and their weights
		for weights in self.hidden_to_output_weights:
			outputs = np.append(outputs, self._sigmoid(np.dot(hidden_outputs, weights)))

		return outputs

	def _output_delta(self, o, t):
		return o * (1 - o) * (self._sigmoid(t) - o)

	def _hidden_delta(self, o, h, deltas):
		output_term = 0
		row = h - self.num_input
		for k in range(self.num_input + self.num_hidden, self.num_input + self.num_hidden + self.num_output):
			column = k - self.num_input - self.num_hidden
			output_term += np.transpose(self.hidden_to_output_weights)[row,column] * deltas[k]

		return o * (1 - o) * output_term

	def _weight_increment(self, i, j, deltas, outputs):
		return self.learning_rate * deltas[j] * outputs[i]

	def _display_weights_and_error(self, iteration_number):
		# Header
		header = f"Iteration {iteration_number + 1}:"
		print("=" * len(header))
		print(header)
		print("=" * len(header))

		# Input to Hidden weights
		for row in range(0, self.num_hidden):
			for column in range(0, self.num_input):
				i = column
				j = row + self.num_input
				print(f"w_{i},{j} = {self.input_to_hidden_weights[row,column]}")
				
		# Hidden to Output weights
		for row in range(0, self.num_output):
			for column in range(0, self.num_hidden):
				i = column + self.num_input
				j = row + self.num_input + self.num_hidden
				print(f"w_{i},{j} = {self.hidden_to_output_weights[row,column]}")

		# Error Term
		error_term = f"Error Term = {self.errors[iteration_number]}"
		print("*" * len(error_term))
		print(error_term)
		print("*" * len(error_term))
		print()

	# Train the feedforward network using the stochastic Backpropagation algorithm
	def train(self, verbose = True):
		# Reset the errors to an empty list
		self.errors = []

		# Initialise all weights to small random values between 0.01 and 0.01 (to match the manual iterations)
		# These can be adjusted as necessary
		self.input_to_hidden_weights = np.random.uniform(low = 0.01, high = 0.01, size = (self.num_hidden, self.num_input))
		self.hidden_to_output_weights = np.random.uniform(low = 0.01, high = 0.01, size = (self.num_output, self.num_hidden))

		# Until the termination condition is met, Do:
		for epoch in range(self.max_epochs):
			# Start with an empty error list for this iteration
			iteration_errors = []

			# For every training instance in the training set
			for (x, t) in self.training_data:
				# Step (i)
				# Feed x into the network and compute o_u for every unit u
				outputs = self._feed(x)

				# Calculate the error term for this iteration, and add it to iteration_errors
				error_term = np.subtract(outputs[self.num_input + self.num_hidden : self.num_input + self.num_hidden + self.num_output], list(map(lambda x: self._sigmoid(x), t))) ** 2
				iteration_errors.append(error_term)

				# Step (ii)
				# Initialise the error terms to the empty dictionary (for easier indexing)
				deltas = {}

				# For each output unit k, calculate the error term delta[k]
				for k in range(self.num_input + self.num_hidden, len(outputs)):
					deltas[k] = self._output_delta(outputs[k], t[len(deltas)])

				# Step (iii)
				# For each hidden unit h, calculate the error term delta[h]
				for h in range(self.num_input, self.num_input + self.num_hidden):
					deltas[h] = self._hidden_delta(outputs[h], h, deltas)

				# Step (iv)
				# Update each network weight
				# First, the input_to_hidden weights
				for row in range(0, self.num_hidden):
					for column in range(0, self.num_input):
						i = column
						j = row + self.num_input
						self.input_to_hidden_weights[row,column] += self._weight_increment(i, j, deltas, outputs) * self.input_to_hidden_weights[row,column]
				
				# Then the hidden_to_output weights
				for row in range(0, self.num_output):
					for column in range(0, self.num_hidden):
						i = column + self.num_input
						j = row + self.num_input + self.num_hidden
						self.hidden_to_output_weights[row,column] += self._weight_increment(i, j, deltas, outputs) * self.hidden_to_output_weights[row,column]

			
			# Add the error term for this training instance to the list of overall errors
			self.errors.append(0.5 * np.sum(iteration_errors))

			if (verbose):
				# Display the weight progressions and network error value
				self._display_weights_and_error(epoch)

		# After training, plot the error terms over time
		df = pd.DataFrame(dict(Iteration = np.arange(self.max_epochs), Error = self.errors))
		g = sns.relplot(x = "Iteration", y = "Error", kind = "line", data = df, sort = False)
		g.figure.autofmt_xdate()
		plt.show(block = True)


def main():
	sns.set_theme(style = "darkgrid")
	training_data = [
		[[1, 0, 0, 1, 0, 0, 1, 0, 1, 0], [0.1, 0.9]],
		[[1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0.1, 0.9]],
		[[0, 1, 0, 1, 0, 0, 1, 0, 1, 0], [0.9, 0.1]],
		[[0, 0, 1, 0, 1, 0, 1, 0, 1, 0], [0.9, 0.1]],
		[[0, 0, 1, 0, 0, 1, 0, 1, 1, 0], [0.9, 0.1]],
		[[0, 0, 1, 0, 0, 1, 0, 1, 0, 1], [0.1, 0.9]],
		[[0, 1, 0, 0, 0, 1, 0, 1, 0, 1], [0.9, 0.1]],
		[[1, 0, 0, 0, 1, 0, 1, 0, 1, 0], [0.1, 0.9]],
		[[1, 0, 0, 0, 0, 1, 0, 1, 1, 0], [0.9, 0.1]],
		[[0, 0, 1, 0, 1, 0, 0, 1, 1, 0], [0.9, 0.1]],
		[[1, 0, 0, 0, 1, 0, 0, 1, 0, 1], [0.9, 0.1]],
		[[0, 1, 0, 0, 1, 0, 1, 0, 0, 1], [0.9, 0.1]],
		[[0, 1, 0, 1, 0, 0, 0, 1, 1, 0], [0.9, 0.1]],
		[[0, 0, 1, 0, 1, 0, 1, 0, 0, 1], [0.1, 0.9]]
	]

	network = FeedForwardNetwork(training_data, max_epochs = 2000, num_hidden = 2, learning_rate = 0.05)
	network.train()

if __name__ == "__main__":
	header = "62760858 - COS4852 - Task 3 - 2022"
	print("=" * len(header))
	print(header)
	print("=" * len(header))
	main()