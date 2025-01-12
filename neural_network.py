import random
import math
from constants import *
import pickle

class NeuralNetwork:
    """Class representing a simple neural network."""
    def __init__(self):

        self.inputs = [0.0] * INPUTS  # 14 inputs
        self.hidden_layer = [0.0] * NEURONS  # 4 hidden neurons
        self.outputs = [0.0] * OUTPUTS  # 2 outputs
        self.total_reward = 0.0  # Acumula as recompensas

        # Limitar pesos e biases a um intervalo menor
        self.weights_input_hidden = [[random.uniform(-0.5, 0.5) for _ in range(INPUTS)] for _ in range(NEURONS)]
        self.bias_hidden = [random.uniform(-0.5, 0.5) for _ in range(NEURONS)]
        self.weights_hidden_output = [[random.uniform(-0.5, 0.5) for _ in range(NEURONS)] for _ in range(OUTPUTS)]
        self.bias_output = [random.uniform(-0.5, 0.5) for _ in range(OUTPUTS)]
        
        self.generation = 1

    # Adicionando funções de ativação
    def relu(self, x):
        """ReLU activation function."""
        return max(0, x)
    
    def tanh(self, x):
        """tanh activation function."""
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + math.exp(-x))

    def mutate(self, mutation_rate):
        """Mutate the weights and biases with a given mutation rate."""
        for i in range(len(self.weights_input_hidden)):
            for j in range(len(self.weights_input_hidden[i])):
                if random.random() < mutation_rate:
                    #print(f"Mutating Input weight {i} {j}")
                    self.weights_input_hidden[i][j] += random.uniform(-0.1, 0.1)

        for i in range(len(self.bias_hidden)):
            if random.random() < mutation_rate:
                #print(f"Mutating hidden bias {i}")
                self.bias_hidden[i] += random.uniform(-0.1, 0.1)

        for i in range(len(self.weights_hidden_output)):
            for j in range(len(self.weights_hidden_output[i])):
                if random.random() < mutation_rate:
                    #print(f"Mutating hidden weight {i} {j}")
                    self.weights_hidden_output[i][j] += random.uniform(-0.1, 0.1)

        for i in range(len(self.bias_output)):
            if random.random() < mutation_rate:
                #print(f"Mutating output bias {i}")
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
            self.hidden_layer[i] = self.relu(activation)

        # Calculate output layer activations
        for i in range(len(self.outputs)):
            activation = sum(w * hidden for w, hidden in zip(self.weights_hidden_output[i], self.hidden_layer))
            activation += self.bias_output[i]
            self.outputs[i] = self.relu(activation)

        return self.outputs

    def calculate_reward(self, 
                     captured_prey=False, 
                     previous_distance=0.0, 
                     current_distance=0.0, 
                     energy_used=0, 
                     split_happened=False, 
                     speed=0.0, 
                     group_bonus=0):
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

        # Recompensa baseada na velocidade (quanto maior, melhor até um limite)
        reward += min(speed, 5) * 0.5  # Máximo de 5 pontos de bônus para velocidade

        # Estratégia de Grupo: Recompensa por cooperação (ou penalidade por competição)
        reward += group_bonus

        # Acumular a recompensa total
        self.total_reward += reward

        return reward

    
    def save(self, file_path, generation):
        """Save the neural network to a file."""
        data = {
            "weights_input_hidden": self.weights_input_hidden,
            "bias_hidden": self.bias_hidden,
            "weights_hidden_output": self.weights_hidden_output,
            "bias_output": self.bias_output,
            "total_reward": self.total_reward,
            "generation" : generation
        }
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Neural network saved to {file_path} - Generation {generation}")

    @staticmethod
    def load(file_path):
        """Load a neural network from a file."""
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            nn = NeuralNetwork()
            nn.weights_input_hidden = data["weights_input_hidden"]
            nn.bias_hidden = data["bias_hidden"]
            nn.weights_hidden_output = data["weights_hidden_output"]
            nn.bias_output = data["bias_output"]
            nn.total_reward = data["total_reward"]
            nn.generation = data["generation"]
            print(f"Neural network loaded from {file_path}")
            return nn
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            print(f"Failed to load neural network from {file_path}. Using default values.")
            return NeuralNetwork()
        
    def calculate_rank(self):
        """
        Calculate the rank of the neural network based on weights, biases, and total reward.
        Incorporates L2 regularization, reward-based evaluation, and generation normalization.
        """
        # Regularização L2: Soma dos quadrados dos pesos e bias
        l2_weights = sum(w**2 for row in self.weights_input_hidden for w in row)
        l2_weights += sum(w**2 for row in self.weights_hidden_output for w in row)
        l2_bias = sum(b**2 for b in self.bias_hidden)
        l2_bias += sum(b**2 for b in self.bias_output)

        # Recompensa acumulada com peso aumentado
        reward_factor = self.total_reward * 10

        # Penalidade L1 para valores extremos
        l1_penalty = sum(abs(w) for row in self.weights_input_hidden for w in row)
        l1_penalty += sum(abs(w) for row in self.weights_hidden_output for w in row)
        l1_penalty += sum(abs(b) for b in self.bias_hidden)
        l1_penalty += sum(abs(b) for b in self.bias_output)

        # Penalidade para redes sem histórico de desempenho
        no_reward_penalty = -10 if self.total_reward == 0 else 0

        # Combinar métricas em uma única pontuação de rank
        rank = reward_factor - 0.01 * l1_penalty + 0.001 * l2_weights - 0.1 * l2_bias + no_reward_penalty

        # Normalizar pelo número de gerações (opcional)
        rank = rank / self.generation

        return rank



    def crossover(self, other):
        """Realiza o cruzamento genético com outra rede neural."""
        child = NeuralNetwork()  # Cria uma nova rede neural vazia
        
        # Combina os pesos da camada de entrada para a oculta
        for i in range(len(self.weights_input_hidden)):
            for j in range(len(self.weights_input_hidden[i])):
                # Escolhe o peso de forma aleatória entre os dois pais
                if random.random() < 0.5:
                    child.weights_input_hidden[i][j] = self.weights_input_hidden[i][j]
                else:
                    child.weights_input_hidden[i][j] = other.weights_input_hidden[i][j]

        # Combina os bias da camada oculta
        for i in range(len(self.bias_hidden)):
            if random.random() < 0.5:
                child.bias_hidden[i] = self.bias_hidden[i]
            else:
                child.bias_hidden[i] = other.bias_hidden[i]

        # Combina os pesos da camada oculta para a saída
        for i in range(len(self.weights_hidden_output)):
            for j in range(len(self.weights_hidden_output[i])):
                if random.random() < 0.5:
                    child.weights_hidden_output[i][j] = self.weights_hidden_output[i][j]
                else:
                    child.weights_hidden_output[i][j] = other.weights_hidden_output[i][j]

        # Combina os bias da camada de saída
        for i in range(len(self.bias_output)):
            if random.random() < 0.5:
                child.bias_output[i] = self.bias_output[i]
            else:
                child.bias_output[i] = other.bias_output[i]
        
        # Perturbação adicional para diversidade genética
        mutation_rate = 0.1  # Taxa de mutação
        child.mutate(mutation_rate)
        
        return child
