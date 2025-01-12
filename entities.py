import pygame
import random
from math import cos, sin, radians, sqrt, degrees, atan2
from constants import *
from neural_network import NeuralNetwork
import numpy as np
import pickle

class SpatialGrid:
    """Class for optimizing entity lookups using a spatial grid."""
    def __init__(self, width, height, cell_size):
        self.cell_size = cell_size
        self.width = width
        self.height = height
        self.grid = {}

    def _get_cell(self, x, y):
        return int(x // self.cell_size), int(y // self.cell_size)

    def add_entity(self, entity):
        cell = self._get_cell(entity.x, entity.y)
        if cell not in self.grid:
            self.grid[cell] = []
        self.grid[cell].append(entity)

    def remove_entity(self, entity):
        cell = self._get_cell(entity.x, entity.y)
        if cell in self.grid and entity in self.grid[cell]:
            self.grid[cell].remove(entity)
            if not self.grid[cell]:
                del self.grid[cell]

    def update_entity(self, entity, old_x, old_y):
        old_cell = self._get_cell(old_x, old_y)
        new_cell = self._get_cell(entity.x, entity.y)
        if old_cell != new_cell:
            self.remove_entity(entity)
            self.add_entity(entity)

    def get_nearby_entities(self, x, y):
        cell = self._get_cell(x, y)
        nearby = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                neighbor_cell = (cell[0] + dx, cell[1] + dy)
                if neighbor_cell in self.grid:
                    nearby.extend(self.grid[neighbor_cell])
        return nearby

class Prey:
    """Class representing a prey entity."""
    def __init__(self, x, y, energy=None, neural_network=None):
        self.x = x
        self.y = y
        self.color = PREY_COLOR
        self.size = ENTITY_SIZE
        self.energy = energy if energy is not None else random.randint(40, 100)
        self.active = True  # Whether the prey is moving
        self.split_progress = 0  # Progress towards splitting
        self.sensors = [0, 0, 0, 0]  # Sensor distances for [front, back, left, right]
        self.neural_network = neural_network if neural_network else NeuralNetwork()  # Nova ou existente
        self.is_stationary = 0  # Whether the prey is stationary
        self.direction = random.uniform(0, 360)  # In degrees
        self.reward = 0.0 
        self.generation = 1
        self.last_direction = None 

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (self.x, self.y), self.size)

    def calculate_and_update_reward(self, captured_prey, previous_distance, current_distance, energy_used, split_happened):
        """Calcula a recompensa e atualiza a rede neural."""
        # Calcular a recompensa com base nos parâmetros do ambiente
        step_reward = self.neural_network.calculate_reward(
            captured_prey=captured_prey,
            previous_distance=previous_distance,
            current_distance=current_distance,
            energy_used=energy_used,
            split_happened=split_happened
        )

        # Atualizar a recompensa total do indivíduo
        self.reward += step_reward

        return step_reward

    def move(self):
        if self.active:
            inputs = self._prepare_inputs()
            speed, angular_velocity = self.neural_network.forward(inputs)

            # Normalize outputs
            speed = speed * 5  # Scale speed to a maximum of 5
            # Constrain angular velocity to reduce rotation
            max_angular_velocity = 0.1  # Adjust this value to allow slight rotations
            angular_velocity = max(-max_angular_velocity, min(max_angular_velocity, angular_velocity - 0.5)) * 2 * 3.14159
            self.direction += angular_velocity
            self.direction %= 360

            # Convert speed and angular velocity into movement
            dx = int(speed * cos(angular_velocity * 2 * 3.14159))
            dy = int(speed * sin(angular_velocity * 2 * 3.14159))

            # Check if the object is stationary
            self.is_stationary = (dx == 0 and dy == 0)

            old_x, old_y = self.x, self.y
            #self.x += dx
            #self.y += dy
            # Atualiza a posição
            self.x += speed * cos(radians(self.direction))
            self.y += speed * sin(radians(self.direction))

            # Wrap around the screen edges for infinite world effect
            if self.x + self.size < 0:  # Exiting left
                self.x = SCREEN_WIDTH - SIDEBAR_WIDTH + self.size
            elif self.x - self.size > SCREEN_WIDTH - SIDEBAR_WIDTH:  # Exiting right
                self.x = -self.size

            if self.y + self.size < 0:  # Exiting top
                self.y = SCREEN_HEIGHT - BOTTOMBAR_HEIGHT + self.size
            elif self.y - self.size > SCREEN_HEIGHT - BOTTOMBAR_HEIGHT:  # Exiting bottom
                self.y = -self.size
            
            return old_x, old_y
        return None, None

    def update_energy(self):
        if self.active:
            # Reduce energy only if not stationary
            if not self.is_stationary:
                self.energy -= PREY_ENERGY_DEPLETION_RATE
                if self.energy <= 0:
                    self.energy = 0
                    self.active = False
        else:
            self.energy += PREY_ENERGY_RECOVERY_RATE
            if self.energy >= 60:
                self.active = True

    def update_split(self, preys, grid):
        if self.split_progress < 100:
            self.split_progress += random.uniform(0, PREY_SPLIT_RATE)

        if self.split_progress >= 100: # and self.energy >= 30:
            self.split_progress = 0
            # Cruzamento genético entre dois indivíduos
            parent1 = random.choice(preys)
            parent2 = random.choice(preys)
            child = parent1.neural_network.crossover(parent2.neural_network)
            # Escolher posição inicial para o novo predador (pode ser aleatória ou baseada nos pais)
            child_x = (parent1.x + parent2.x) // 2  # Média das posições x dos pais
            child_y = (parent1.y + parent2.y) // 2  # Média das posições y dos pais
            # Criação de um novo predador com a rede neural do filho
            new_prey = Prey(child_x, child_y, self.energy, child)
            new_prey.generation = self.generation + 1
            new_prey.neural_network = NeuralNetwork()  # Create new neural network for offspring
            mutation_rate = max(0.01, MUTATION_RATE - new_prey.reward / 100)
            new_prey.neural_network.mutate(mutation_rate)
            preys.append(new_prey)
            grid.add_entity(new_prey)

            # Save the best neural network
            save_file = "models/prey_best_neural_network.pkl"
            best_rank = float("-inf")

            try:
                # Check if the file exists and read the rank from it
                with open(save_file, "rb") as f:
                    saved_data = pickle.load(f)
                    best_rank = (
                        # Regularização L2 para pesos e bias
                        0.001 * (
                            sum(w**2 for row in saved_data["weights_input_hidden"] for w in row) +
                            sum(w**2 for row in saved_data["weights_hidden_output"] for w in row) +
                            sum(b**2 for b in saved_data["bias_hidden"]) +
                            sum(b**2 for b in saved_data["bias_output"])
                        ) -
                        # Penalidade L1 para valores extremos
                        0.01 * (
                            sum(abs(w) for row in saved_data["weights_input_hidden"] for w in row) +
                            sum(abs(w) for row in saved_data["weights_hidden_output"] for w in row) +
                            sum(abs(b) for b in saved_data["bias_hidden"]) +
                            sum(abs(b) for b in saved_data["bias_output"])
                        ) +
                        # Recompensa acumulada
                        saved_data.get("total_reward", 0) * 10 -
                        # Penalidade adicional para bias (se aplicável)
                        0.1 * sum(b**2 for b in saved_data["bias_output"])
                    )


            except (FileNotFoundError, EOFError, pickle.UnpicklingError):
                pass  # Use default best_rank = -inf if file is missing or invalid

            # Compare ranks
            new_rank = new_prey.neural_network.calculate_rank()
            print(f"New rank: {new_rank} - Best rank: {best_rank}")
            if new_rank > best_rank:
                new_prey.neural_network.save(save_file, new_prey.generation)

            # Remove the mother if the number of preys exceeds the maximum allowed
            if len(preys) > PREY_MAX:
                grid.remove_entity(self)
                preys.remove(self)
                del self

    def get_sensors(self, preys, predators):
        """Retorna os sensores (raios) da presa, otimizados com NumPy."""
        sensors = []
        num_sensors = 4  # Número de raios
        vision_distance = 0.8 * min(SCREEN_WIDTH, SCREEN_HEIGHT)  # Distância máxima do sensor
        angle_step = 360 / num_sensors  # Ângulo entre sensores
        self_position = 0
        sensor_angle_degrees = 0

        # Pré-calcular os ângulos dos sensores
        sensor_angles = [radians(i * angle_step + self.direction) for i in range(num_sensors)]

        # Filtrar entidades próximas (reduz o número de iterações nos loops)
        filtered_preys = [prey for prey in preys if abs(prey.x - self.x) <= vision_distance and abs(prey.y - self.y) <= vision_distance and prey != self]
        filtered_predators = [predator for predator in predators if abs(predator.x - self.x) <= vision_distance and abs(predator.y - self.y) <= vision_distance]

        # Processar cada sensor individualmente
        for angle in sensor_angles:
            closest_distance = vision_distance  # Inicialmente o sensor tem alcance máximo
            target_type = None  # Tipo de alvo detectado

            # Direção do sensor
            sensor_dx = cos(angle)
            sensor_dy = sin(angle)

            # Verifica colisão com predadores (vetorizado)
            if filtered_predators:
                predator_positions = np.array([[predator.x, predator.y] for predator in filtered_predators])
                self_position = np.array([self.x, self.y])
                predator_vectors = predator_positions - self_position  # Vetores para cada predador
                predator_distances = np.linalg.norm(predator_vectors, axis=1)  # Distâncias dos predadores

                # Filtrar predadores dentro do campo de visão
                predator_angles = np.degrees(np.arctan2(predator_vectors[:, 1], predator_vectors[:, 0]))
                sensor_angle_degrees = np.degrees(np.arctan2(sensor_dy, sensor_dx))
                delta_angles = np.abs((predator_angles - sensor_angle_degrees + 180) % 360 - 180)

                # Predadores visíveis pelo sensor
                valid_predators = (predator_distances < vision_distance) & (delta_angles <= angle_step / 2)
                if valid_predators.any():
                    closest_predator_index = np.argmin(predator_distances[valid_predators])
                    closest_distance = predator_distances[valid_predators][closest_predator_index]
                    target_type = "predator"

            # Verifica colisão com outras presas (vetorizado)
            if filtered_preys:
                prey_positions = np.array([[prey.x, prey.y] for prey in filtered_preys])
                prey_vectors = prey_positions - self_position  # Vetores para cada presa
                prey_distances = np.linalg.norm(prey_vectors, axis=1)  # Distâncias das presas

                # Filtrar presas dentro do campo de visão
                prey_angles = np.degrees(np.arctan2(prey_vectors[:, 1], prey_vectors[:, 0]))
                delta_angles = np.abs((prey_angles - sensor_angle_degrees + 180) % 360 - 180)

                # Presas visíveis pelo sensor
                valid_preys = (prey_distances < vision_distance) & (delta_angles <= angle_step / 2)
                if valid_preys.any():
                    closest_prey_index = np.argmin(prey_distances[valid_preys])
                    closest_distance = prey_distances[valid_preys][closest_prey_index]
                    target_type = "prey"

            # Calcula o ponto final do sensor baseado na distância mais próxima
            end_x = self.x + sensor_dx * closest_distance
            end_y = self.y + sensor_dy * closest_distance

            # Adiciona o sensor ao resultado
            sensor = {
                "start": (self.x, self.y),
                "end": (end_x, end_y),
                "distance": closest_distance,
                "target_type": target_type
            }
            sensors.append(sensor)

        return sensors


    def _prepare_inputs(self):
        """Prepare inputs for the neural network."""
        inputs = []
        for sensor in self.sensors:
            #print(sensor["distance"])
            if isinstance(sensor, tuple) and len(sensor) == 4:
                distance, target = sensor["distance"], sensor["target_type"]
                inputs.append(distance / 50.0)  # Normalize distance to [0, 1]
                inputs.append(1 if target == 'PREY' else 0)  # Binary encoding for targets
            else:
                # Default values if sensor data is invalid
                inputs.append(1.0)  # Maximum normalized distance
                inputs.append(0)    # No target detected

        inputs.append(1)  # BIAS
        inputs.append(self.energy / 100.0)  # Normalize energy to [0, 1]
        inputs.append(0)  # DIGESTION (not applicable to prey)
        inputs.append(self.split_progress / 100.0)  # Normalize split progress to [0, 1]
        inputs.append(self.x / SCREEN_WIDTH)  # Normalize X position to [0, 1]
        inputs.append(self.y / SCREEN_HEIGHT)  # Normalize Y position to [0, 1]
        return inputs

class Predator:
    """Class representing a predator entity."""
    def __init__(self, x, y, energy=None, neural_network=None):
        self.x = x
        self.y = y
        self.color = PREDATOR_COLOR
        self.size = ENTITY_SIZE
        self.energy = energy if energy is not None else random.randint(50, 100)
        self.split_progress = 0  # Number of successful prey eaten for splitting
        self.digestion = 0  # Digestion timer
        self.sensors = [0, 0, 0, 0]  # Sensor distances for 4 angled directions
        self.neural_network = neural_network if neural_network else NeuralNetwork()  # Nova ou existente
        self.direction = random.uniform(0, 360)  # In degrees
        self.reward = 0.0
        self.generation = 1
        self.last_direction = None 

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (self.x, self.y), self.size)

    def move(self):
        inputs = self._prepare_inputs()
        speed, angular_velocity = self.neural_network.forward(inputs)

        # Normalize outputs
        speed = speed * 5  # Scale speed to a maximum of 5
        angular_velocity = (angular_velocity - 0.5) * 2 * 3.14159  # Scale to [-pi, pi]
        self.direction += angular_velocity
        self.direction %= 360

        # Convert speed and angular velocity into movement
        dx = int(speed * cos(angular_velocity * 2 * 3.14159))
        dy = int(speed * sin(angular_velocity * 2 * 3.14159))

        old_x, old_y = self.x, self.y
        #self.x += dx
        #self.y += dy
        # Atualiza a posição
        self.x += speed * cos(radians(self.direction))
        self.y += speed * sin(radians(self.direction))

        # Wrap around the screen edges for infinite world effect
        if self.x + self.size < 0:  # Exiting left
            self.x = SCREEN_WIDTH - SIDEBAR_WIDTH + self.size
        elif self.x - self.size > SCREEN_WIDTH - SIDEBAR_WIDTH:  # Exiting right
            self.x = -self.size

        if self.y + self.size < 0:  # Exiting top
            self.y = SCREEN_HEIGHT - BOTTOMBAR_HEIGHT + self.size
        elif self.y - self.size > SCREEN_HEIGHT - BOTTOMBAR_HEIGHT:  # Exiting bottom
            self.y = -self.size
        
        return old_x, old_y

    def update_energy(self):
        if self.digestion > 0:
            self.digestion -= DIGESTION_DEPLETION_RATE
            self.digestion = max(0, self.digestion)
        else:
            self.energy -= PREDATOR_ENERGY_DEPLETION_RATE
            if self.energy <= 0:
                self.energy = 0

    def eat_prey(self, preys, grid, predators):
        for prey in preys:
            if ((self.x - prey.x) ** 2 + (self.y - prey.y) ** 2) ** 0.5 <= self.size:
                if self.digestion == 0:
                    grid.remove_entity(prey)
                    preys.remove(prey)
                    del prey
                    if self.energy < 100:
                        self.energy += 10
                    self.digestion = 100  # Start digestion
                    self.split_progress += 1
                    if self.split_progress >= 2:
                        self.split_progress = 0
                        parent1 = random.choice(predators)
                        parent2 = random.choice(predators)

                        # Cruzamento genético para criar a rede neural do filho
                        child_network = parent1.neural_network.crossover(parent2.neural_network)

                        # Escolher posição inicial para o novo predador (pode ser aleatória ou baseada nos pais)
                        child_x = (parent1.x + parent2.x) // 2  # Média das posições x dos pais
                        child_y = (parent1.y + parent2.y) // 2  # Média das posições y dos pais
                        # Criar o novo predador com a rede neural resultante e posições iniciais
                        new_predator = Predator(child_x, child_y, self.energy, child_network)
                        #new_predator = Predator(self.x + random.randint(-10, 10), self.y + random.randint(-10, 10))
                        new_predator.generation = self.generation + 1
                        new_predator.neural_network = NeuralNetwork()  # Create new neural network for offspring
                        split_happened = True
                        new_predator.reward = new_predator.neural_network.calculate_reward(
                            split_happened=split_happened
                        )

                        mutation_rate = max(0.01, MUTATION_RATE - new_predator.reward / 100)
                        new_predator.neural_network.mutate(mutation_rate)

                        # Save the best neural network
                        save_file = "models/predator_best_neural_network.pkl"
                        best_rank = float("-inf")

                        try:
                            # Check if the file exists and read the rank from it
                            with open(save_file, "rb") as f:
                                saved_data = pickle.load(f)
                                best_rank = (
                                    # Regularização L2 para pesos e bias
                                    0.001 * (
                                        sum(w**2 for row in saved_data["weights_input_hidden"] for w in row) +
                                        sum(w**2 for row in saved_data["weights_hidden_output"] for w in row) +
                                        sum(b**2 for b in saved_data["bias_hidden"]) +
                                        sum(b**2 for b in saved_data["bias_output"])
                                    ) -
                                    # Penalidade L1 para valores extremos
                                    0.01 * (
                                        sum(abs(w) for row in saved_data["weights_input_hidden"] for w in row) +
                                        sum(abs(w) for row in saved_data["weights_hidden_output"] for w in row) +
                                        sum(abs(b) for b in saved_data["bias_hidden"]) +
                                        sum(abs(b) for b in saved_data["bias_output"])
                                    ) +
                                    # Recompensa acumulada
                                    saved_data.get("total_reward", 0) * 10 -
                                    # Penalidade adicional para bias (se aplicável)
                                    0.1 * sum(b**2 for b in saved_data["bias_output"])
                                )
                        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
                            pass  # Use default best_rank = -inf if file is missing or invalid

                        # Compare ranks
                        new_rank = new_predator.neural_network.calculate_rank()
                        if new_rank > best_rank:
                            new_predator.neural_network.save(save_file, new_predator.generation)

                        return new_predator
                return None

    def process_single_sensor(args):
        """Processa um único sensor com base nos argumentos fornecidos."""
        angle, self_position, vision_distance, preys, predators = args
        closest_distance = vision_distance
        target_type = None

        # Verifica colisão com presas
        for prey in preys:
            dx = prey.x - self_position[0]
            dy = prey.y - self_position[1]
            distance = (dx**2 + dy**2) ** 0.5  # Calcula a distância
            if distance < closest_distance:
                closest_distance = distance
                target_type = "prey"

        # Verifica colisão com predadores
        for predator in predators:
            dx = predator.x - self_position[0]
            dy = predator.y - self_position[1]
            distance = (dx**2 + dy**2) ** 0.5  # Calcula a distância
            if distance < closest_distance:
                closest_distance = distance
                target_type = "predator"

        # Calcula o ponto final do sensor
        end_x = self_position[0] + cos(angle) * closest_distance
        end_y = self_position[1] + sin(angle) * closest_distance

        return {
            "start": (self_position[0], self_position[1]),
            "end": (end_x, end_y),
            "distance": closest_distance,
            "target_type": target_type
        }

            
    def get_sensors(self, preys, predators):
        """Retorna os sensores (raios) do predador, otimizados com NumPy."""
        sensors = []
        num_sensors = 4  # Número de raios no campo de visão
        vision_distance = 0.8 * min(SCREEN_WIDTH, SCREEN_HEIGHT)  # Distância máxima do sensor
        angle_step = 45 / num_sensors  # Ângulo entre sensores dentro do campo de visão (45 graus)

        # Pré-calcular os ângulos dos sensores
        sensor_angles = [radians(self.direction - 22.5 + i * angle_step) for i in range(num_sensors)]

        # Posição inicial ajustada para os sensores
        start_x = self.x + cos(radians(self.direction)) * self.size
        start_y = self.y + sin(radians(self.direction)) * self.size
        self_position = np.array([start_x, start_y])  # Vetor da posição inicial do sensor

        # Processar cada sensor individualmente
        for angle in sensor_angles:
            closest_distance = vision_distance  # Inicialmente o sensor tem alcance máximo
            target_type = None  # Tipo de alvo detectado

            # Direção do sensor
            sensor_dx = cos(angle)
            sensor_dy = sin(angle)
            sensor_angle_degrees = np.degrees(np.arctan2(sensor_dy, sensor_dx))

            # Verifica colisão com presas (vetorizado)
            if preys:
                prey_positions = np.array([[prey.x, prey.y] for prey in preys])
                prey_vectors = prey_positions - self_position  # Vetores para cada presa
                prey_distances = np.linalg.norm(prey_vectors, axis=1)  # Distâncias das presas

                # Filtrar presas dentro do campo de visão
                prey_angles = np.degrees(np.arctan2(prey_vectors[:, 1], prey_vectors[:, 0]))
                delta_angles = np.abs((prey_angles - sensor_angle_degrees + 180) % 360 - 180)

                # Presas visíveis pelo sensor
                valid_preys = (prey_distances < vision_distance) & (delta_angles <= angle_step / 2)
                if valid_preys.any():
                    closest_prey_index = np.argmin(prey_distances[valid_preys])
                    closest_distance = prey_distances[valid_preys][closest_prey_index]
                    target_type = "prey"

            # Verifica colisão com predadores (vetorizado)
            if predators:
                predator_positions = np.array([[predator.x, predator.y] for predator in predators if predator != self])
                if len(predator_positions) > 0:  # Verifica se há predadores no alcance
                    predator_vectors = predator_positions - self_position  # Vetores para cada predador
                    predator_distances = np.linalg.norm(predator_vectors, axis=1)  # Distâncias dos predadores

                    # Filtrar predadores dentro do campo de visão
                    predator_angles = np.degrees(np.arctan2(predator_vectors[:, 1], predator_vectors[:, 0]))
                    delta_angles = np.abs((predator_angles - sensor_angle_degrees + 180) % 360 - 180)

                    # Predadores visíveis pelo sensor
                    valid_predators = (predator_distances < vision_distance) & (delta_angles <= angle_step / 2)
                    if valid_predators.any():
                        closest_predator_index = np.argmin(predator_distances[valid_predators])
                        closest_distance = predator_distances[valid_predators][closest_predator_index]
                        target_type = "predator"

            # Calcula o ponto final do sensor baseado na distância mais próxima
            end_x = start_x + sensor_dx * closest_distance
            end_y = start_y + sensor_dy * closest_distance

            # Adiciona o sensor ao resultado
            sensor = {
                "start": (start_x, start_y),
                "end": (end_x, end_y),
                "distance": closest_distance,
                "target_type": target_type
            }
            sensors.append(sensor)

        return sensors



    
    def _prepare_inputs(self):
        """Prepare inputs for the neural network."""
        inputs = []
        for sensor in self.sensors:
            if isinstance(sensor, tuple) and len(sensor) == 4:
                distance, target = sensor["distance"], sensor["target_type"]
                inputs.append(distance / 50.0)  # Normalize distance to [0, 1]
                inputs.append(1 if target == 'PREY' else 0)  # Binary encoding for targets
            else:
                # Default values if sensor data is invalid
                inputs.append(1.0)  # Maximum normalized distance
                inputs.append(0)    # No target detected

        inputs.append(1)  # BIAS
        inputs.append(self.energy / 100.0)  # Normalize energy to [0, 1]
        inputs.append(self.digestion / 100.0)  # Normalize digestion to [0, 1]
        inputs.append(self.split_progress / 100.0)  # Normalize split progress to [0, 1]
        inputs.append(self.x / SCREEN_WIDTH)  # Normalize X position to [0, 1]
        inputs.append(self.y / SCREEN_HEIGHT)  # Normalize Y position to [0, 1]
        return inputs

def create_entities(randomize_energy=False):
    """Create initial populations of prey and predators."""
    grid = SpatialGrid(SCREEN_WIDTH, SCREEN_HEIGHT, cell_size=50)

    # Paths to saved neural network models
    prey_model_path = "models/prey_best_neural_network.pkl"
    predator_model_path = "models/predator_best_neural_network.pkl"

    # Attempt to load the best neural network for preys
    try:
        prey_nn = NeuralNetwork.load(prey_model_path)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        prey_nn = None

    # Attempt to load the best neural network for predators
    try:
        predator_nn = NeuralNetwork.load(predator_model_path)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        predator_nn = None

    # Create initial preys
    preys = []
    for _ in range(PREY_INITIAL_POPULATION):
        prey = Prey(
            random.randint(0, SCREEN_WIDTH),
            random.randint(0, SCREEN_HEIGHT - BOTTOMBAR_HEIGHT),
            random.randint(40, 100) if randomize_energy else 100
        )
        if prey_nn:
            prey.neural_network = prey_nn
            prey.generation = prey_nn.generation
        preys.append(prey)

    # Create initial predators
    predators = []
    for _ in range(PREDATOR_INITIAL_POPULATION):
        predator = Predator(
            random.randint(0, SCREEN_WIDTH),
            random.randint(0, SCREEN_HEIGHT - BOTTOMBAR_HEIGHT),
            random.randint(50, 100) if randomize_energy else 100
        )
        if predator_nn:
            predator.neural_network = predator_nn
            predator.generation = predator_nn.generation
        predators.append(predator)

    # Add all entities to the spatial grid
    for entity in preys + predators:
        grid.add_entity(entity)

    return preys, predators, grid


def recreate_entities(grid, preys, predators, randomize_energy=False, entity_type=None):
    """Recreate populations of prey and predators while obeying PREY_MAX and PREDATOR_MAX."""
    new_preys = []
    new_predators = []

    # Load the best neural network based on entity type
    if entity_type == 'prey':
        save_file = "models/prey_best_neural_network.pkl"
    elif entity_type == 'predator':
        save_file = "models/predator_best_neural_network.pkl"
    else:
        save_file = None

    saved_neural_network = None
    saved_generation = 1

    if save_file:
        try:
            saved_neural_network = NeuralNetwork.load(save_file)
            saved_generation = saved_neural_network.generation  # Load generation from file
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            pass  # If the file doesn't exist, use default values

    # Clear the grid and remove existing entities
    if entity_type == 'prey':
        preys.clear()
        new_preys = [Prey(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT - BOTTOMBAR_HEIGHT),
                          random.randint(40, 100) if randomize_energy else 100)
                     for _ in range(PREY_MAX)]
        for prey in new_preys:
            prey.generation = saved_generation
            if saved_neural_network:
                prey.neural_network = saved_neural_network
        preys.extend(new_preys)
    elif entity_type == 'predator':
        predators.clear()
        new_predators = [Predator(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT - BOTTOMBAR_HEIGHT),
                                  random.randint(50, 100) if randomize_energy else 100)
                         for _ in range(PREDATOR_MAX)]
        for predator in new_predators:
            predator.generation = saved_generation
            if saved_neural_network:
                predator.neural_network = saved_neural_network
        predators.extend(new_predators)

    # Add new entities to the grid
    for entity in new_preys + new_predators:
        grid.add_entity(entity)




