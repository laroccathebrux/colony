import random
import math
from constants import *

class NeuralNetwork:
    """Class representing a simple neural network."""
    def __init__(self):
        self.inputs = [0.0] * INPUTS  # 14 inputs
        self.hidden_layer = [0.0] * NEURONS  # 4 hidden neurons
        self.outputs = [0.0] * OUTPUTS  # 2 outputs
        self.total_reward = 0.0  # Acumula as recompensas

        # Randomly initialize weights and biases
        self.weights_input_hidden = [[random.uniform(-1, 1) for _ in range(INPUTS)] for _ in range(NEURONS)]
        self.bias_hidden = [random.uniform(-1, 1) for _ in range(NEURONS)]

        self.weights_hidden_output = [[random.uniform(-1, 1) for _ in range(NEURONS)] for _ in range(OUTPUTS)]
        self.bias_output = [random.uniform(-1, 1) for _ in range(OUTPUTS)]

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + math.exp(-x))

    def mutate(self, mutation_rate):
        """Mutate the weights and biases with a given mutation rate."""
        for i in range(len(self.weights_input_hidden)):
            for j in range(len(self.weights_input_hidden[i])):
                if random.random() < mutation_rate:
                    print(f"Mutating Input weight {i} {j}")
                    self.weights_input_hidden[i][j] += random.uniform(-0.1, 0.1)

        for i in range(len(self.bias_hidden)):
            if random.random() < mutation_rate:
                print(f"Mutating hidden bias {i}")
                self.bias_hidden[i] += random.uniform(-0.1, 0.1)

        for i in range(len(self.weights_hidden_output)):
            for j in range(len(self.weights_hidden_output[i])):
                if random.random() < mutation_rate:
                    print(f"Mutating hidden weight {i} {j}")
                    self.weights_hidden_output[i][j] += random.uniform(-0.1, 0.1)

        for i in range(len(self.bias_output)):
            if random.random() < mutation_rate:
                print(f"Mutating output bias {i}")
                self.bias_output[i] += random.uniform(-0.1, 0.1)

    def forward(self, input_data):
        """Perform a forward pass through the network."""
        activation = 0.0
        if len(input_data) != INPUTS:
            raise ValueError(f"Input data must have {INPUTS} elements.")

        # Set inputs
        self.inputs = input_data

        # Calculate hidden layer activations
        for i in range(len(self.hidden_layer)):
            activation = sum(w * inp for w, inp in zip(self.weights_input_hidden[i], self.inputs))
            activation += self.bias_hidden[i]
            self.hidden_layer[i] = self.sigmoid(activation)

        # Calculate output layer activations
        for i in range(len(self.outputs)):
            activation = sum(w * hidden for w, hidden in zip(self.weights_hidden_output[i], self.hidden_layer))
            activation += self.bias_output[i]
            self.outputs[i] = self.sigmoid(activation)

        return self.outputs

    def calculate_reward(self, captured_prey=False, previous_distance=0.0, current_distance=0, energy_used=0, split_happened=False):
        """Calculate the reward for the current step."""
        reward = 0.0

        # Recompensa por capturar presa
        reward += 10 if captured_prey else -1

        # Aproximação do alvo
        reward += max(0, previous_distance - current_distance) * 2  # Peso maior para aproximação

        # Penalidade por energia gasta
        reward -= energy_used * 0.05

        # Recompensa por reprodução
        reward += 20 if split_happened else 0

        # Acumular a recompensa total
        self.total_reward += reward

        return reward
