import pygame
import random
from math import cos, sin, radians, sqrt, degrees, atan2
from constants import *
from neural_network import NeuralNetwork
import numpy as np
import pickle

best_rank = {
    "Prey": float("-inf"),
    "Predator": float("-inf")
}

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
        self.neural_network = neural_network if neural_network else NeuralNetwork()  # New or existing
        self.is_stationary = 0  # Whether the prey is stationary
        self.direction = random.uniform(0, 360)  # In degrees
        self.reward = 0.0 
        self.generation = 1
        self.last_direction = None 
        self.rank = 0.0

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (self.x, self.y), self.size)

    def calculate_and_update_reward(self, captured_prey, previous_distance, current_distance, energy_used, split_happened):
        """Calcula a recompensa e atualiza a rede neural."""
        # Calculate reward based on environment parameters
        step_reward = self.neural_network.calculate_reward(
            captured_prey=captured_prey,
            previous_distance=previous_distance,
            current_distance=current_distance,
            energy_used=energy_used,
            split_happened=split_happened
        )

        # Update the individual's total reward
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

    def update_split(self, preys, grid, previous_x, previous_y):
        if self.split_progress < 100:
            self.split_progress += random.uniform(0, PREY_SPLIT_RATE)

        if self.split_progress >= 100: # and self.energy >= 30:
            self.split_progress = 0

            # Save the best neural network
            save_file = "models/prey_best_neural_network.pkl"
            # Initialize parent2 as None
            parent3 = None

            try:
                # Check if the file exists and read the saved neural network
                with open(save_file, "rb") as f:
                    saved_data = pickle.load(f)

                    # Saved rank calculation
                    best_rank["Prey"] = saved_data.get("rank", float("-inf"))

                    # Extract the neural network data
                    saved_neural_network_data = saved_data.get("neural_network")

                    # If the saved data contains a neural network, create parent2
                    if saved_neural_network_data:
                        parent3_neural_network = NeuralNetwork(is_prey=True)
                        parent3_neural_network.load_from_data(saved_neural_network_data)

                        # Create parent2 using the saved neural network
                        parent3 = Prey(random.randint(0, grid.width), random.randint(0, grid.height), self.energy, parent3_neural_network)

            except (FileNotFoundError, EOFError, pickle.UnpicklingError, AttributeError):
                pass  # If the file is missing or invalid, parent2 will remain None

            parent2_filter = [prey for prey in preys if prey.generation == self.generation]
            parent2 = random.choice(parent2_filter) if parent2_filter != None else None

            # If parent2 was not successfully created, fallback to random choice
            if parent2 is None:
                parent2 = random.choice(preys)
            if parent3 is None:
                parent3 = random.choice(preys)
            # Genetic crossing between two individuals
            #parent1 = random.choice(preys)
            #parent2 = random.choice(preys)
            child = self.neural_network.crossover(parent2.neural_network)
            child = child.crossover(parent3.neural_network)
            # Choose starting position for new predator (can be random or based on parents)
            child_x = (self.x + parent2.x) // 2  # Average of parents' x positions
            child_y = (self.y + parent2.y) // 2  # Average of parents' y positions
            # Creation of a new predator with the son's neural network
            new_prey = Prey(child_x, child_y, self.energy, child)
            new_prey.generation = self.generation + 1
            new_prey.neural_network = NeuralNetwork(is_prey=True)  # Create new neural network for offspring
            #mutation_rate = max(0.01, MUTATION_RATE - new_prey.reward / 100)
            new_prey.neural_network.mutate(MUTATION_RATE)
            preys.append(new_prey)
            grid.add_entity(new_prey)

            # reward
            # Calculate speed based on motion
            speed = ((new_prey.x - previous_x)**2 + (new_prey.y - previous_y)**2) ** 0.5
            # Calculate the reward using the new factors
            reward = new_prey.neural_network.calculate_reward(
                captured_prey=False,  # Prey does not capture
                previous_distance=0.0,  # Distance does not apply here (or adjust as needed)
                current_distance=0.0,
                energy_used=new_prey.energy,  # Energy spent on movement
                split_happened=True,  # Breeding condition
                speed=speed,
                group_bonus=0  # Add group logic if needed
            )
            new_prey.neural_network.total_reward += reward
            new_prey.rank = new_prey.neural_network.calculate_rank()
            #best_rank["Prey"] = max(prey.rank for prey in preys)

            #print(f"Prey rank: {self.rank} - Best rank: {best_rank['Prey']}")

            if new_prey.rank >= best_rank["Prey"]:
                best_rank["Prey"] = new_prey.rank
                new_prey.neural_network.save(save_file, new_prey.generation, best_rank["Prey"])
            """
            new_rank = max(prey.neural_network.calculate_rank() for prey in preys)

            if new_rank > best_rank["Prey"]:
                best_rank["Prey"] = new_rank
                new_prey.neural_network.save(save_file, new_prey.generation, best_rank["Prey"])
            """
            # Remove the mother if the number of preys exceeds the maximum allowed
            if len(preys) > PREY_MAX:
                # Finding the smallest generation among the prey
                min_generation = min(prey.generation for prey in preys)
                
                # Filter prey that has the minimum generation
                min_generation_preys = [prey for prey in preys if prey.generation == min_generation]
                
                # Choose a random prey from the lowest generation
                prey_to_remove = random.choice(min_generation_preys)
                
                # Remove the chosen prey
                grid.remove_entity(prey_to_remove)
                preys.remove(prey_to_remove)
                del prey_to_remove

    def get_sensors(self, preys, predators):
        """Returns the prey's sensors (radii), optimized with NumPy."""
        sensors = []
        num_sensors = 4  # Number of sensors
        vision_distance = 0.8 * min(SCREEN_WIDTH, SCREEN_HEIGHT)  # Maximum sensor distance
        angle_step = 360 / num_sensors  # Angle between sensors
        self_position = 0
        sensor_angle_degrees = 0

        # Pre-calculate sensor angles
        sensor_angles = [radians(i * angle_step + self.direction) for i in range(num_sensors)]

        # Filter nearby entities (reduces the number of iterations in loops)
        filtered_preys = [prey for prey in preys if abs(prey.x - self.x) <= vision_distance and abs(prey.y - self.y) <= vision_distance and prey != self]
        filtered_predators = [predator for predator in predators if abs(predator.x - self.x) <= vision_distance and abs(predator.y - self.y) <= vision_distance]

        # Process each sensor individually
        for angle in sensor_angles:
            closest_distance = vision_distance  # Initially the sensor has maximum range
            target_type = None  # Target type detected

            # Sensor direction
            sensor_dx = cos(angle)
            sensor_dy = sin(angle)

            # Checks for collision with predators (vectorized)
            if filtered_predators:
                predator_positions = np.array([[predator.x, predator.y] for predator in filtered_predators])
                self_position = np.array([self.x, self.y])
                predator_vectors = predator_positions - self_position  # Vectors for each predator
                predator_distances = np.linalg.norm(predator_vectors, axis=1)  # Distances from predators

                # Filter predators within the field of view
                predator_angles = np.degrees(np.arctan2(predator_vectors[:, 1], predator_vectors[:, 0]))
                sensor_angle_degrees = np.degrees(np.arctan2(sensor_dy, sensor_dx))
                delta_angles = np.abs((predator_angles - sensor_angle_degrees + 180) % 360 - 180)

                # Predators visible by the sensor
                valid_predators = (predator_distances < vision_distance) & (delta_angles <= angle_step / 2)
                if valid_predators.any():
                    closest_predator_index = np.argmin(predator_distances[valid_predators])
                    closest_distance = predator_distances[valid_predators][closest_predator_index]
                    target_type = "predator"

            # Checks for collision with other prey (vectorized)
            if filtered_preys:
                prey_positions = np.array([[prey.x, prey.y] for prey in filtered_preys])
                prey_vectors = prey_positions - self_position  # Vectors for each prey
                prey_distances = np.linalg.norm(prey_vectors, axis=1)  # Prey distances

                # Filter prey within field of view
                prey_angles = np.degrees(np.arctan2(prey_vectors[:, 1], prey_vectors[:, 0]))
                delta_angles = np.abs((prey_angles - sensor_angle_degrees + 180) % 360 - 180)

                # Prey visible by sensor
                valid_preys = (prey_distances < vision_distance) & (delta_angles <= angle_step / 2)
                if valid_preys.any():
                    closest_prey_index = np.argmin(prey_distances[valid_preys])
                    closest_distance = prey_distances[valid_preys][closest_prey_index]
                    target_type = "prey"

            # Calculate the sensor endpoint based on the closest distance
            end_x = self.x + sensor_dx * closest_distance
            end_y = self.y + sensor_dy * closest_distance

            # Add the sensor to the result
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
        self.neural_network = neural_network if neural_network else NeuralNetwork()  # Nova or existing
        self.direction = random.uniform(0, 360)  # In degrees
        self.reward = 0.0
        self.generation = 1
        self.last_direction = None 
        self.rank = 0.0

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
        # Update position
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

    def eat_prey(self, preys, grid, predators, previous_x, previous_y):
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

                        # Initialize parent2 as None
                        parent2 = None

                        # Save the best neural network
                        save_file = "models/predator_best_neural_network.pkl"
                        #best_rank = float("-inf")

                        # Initialize parent2 as None
                        parent3 = None

                        try:
                            # Check if the file exists and read the saved neural network
                            with open(save_file, "rb") as f:
                                saved_data = pickle.load(f)

                                # Saved rank calculation
                                best_rank["Predator"] = saved_data.get("rank", float("-inf"))

                                # Extract the neural network data
                                saved_neural_network_data = saved_data.get("neural_network")

                                # If the saved data contains a neural network, create parent2
                                if saved_neural_network_data:
                                    parent3_neural_network = NeuralNetwork(is_prey=True)
                                    parent3_neural_network.load_from_data(saved_neural_network_data)

                                    # Create parent2 using the saved neural network
                                    parent3 = Predator(random.randint(0, grid.width), random.randint(0, grid.height), self.energy, parent3_neural_network)

                        except (FileNotFoundError, EOFError, pickle.UnpicklingError, AttributeError):
                            pass  # If the file is missing or invalid, parent2 will remain None

                        parent2_filter = [predator for predator in predators if predator.generation == self.generation]
                        parent2 = random.choice(parent2_filter if parent2_filter != None else None)

                        # If parent2 was not successfully created, fallback to random choice
                        if parent2 is None:
                            print("Parent2 is None")
                            parent2 = random.choice(preys)

                        if parent3 is None:
                            #print("Parent2 is None")
                            parent3 = random.choice(preys)

                        #parent1 = random.choice(predators)
                        #parent2 = random.choice(predators)

                        # Genetic crossover to create the offspring's neural network
                        child_network = self.neural_network.crossover(parent2.neural_network)
                        child_network = child_network.crossover(parent3.neural_network)

                        # Choose starting position for new predator (can be random or based on parents)
                        child_x = (self.x + parent2.x) // 2  # Average of parents' x positions
                        child_y = (self.y + parent2.y) // 2  # Average of parents' y positions
                        # Create the new predator with the resulting neural network and initial positions
                        new_predator = Predator(child_x, child_y, self.energy, child_network)
                        #new_predator = Predator(self.x + random.randint(-10, 10), self.y + random.randint(-10, 10))
                        new_predator.generation = self.generation + 1
                        new_predator.neural_network = NeuralNetwork()  # Create new neural network for offspring
                        split_happened = True
                        new_predator.reward = new_predator.neural_network.calculate_reward(
                            split_happened=split_happened
                        )

                        #mutation_rate = max(0.01, MUTATION_RATE - new_predator.reward / 100)
                        new_predator.neural_network.mutate(MUTATION_RATE)


                        # Calculate speed based on motion
                        speed = ((new_predator.x - previous_x)**2 + (new_predator.y - previous_y)**2) ** 0.5

                        # Group strategy: Reward for proximity to other predators
                        group_bonus = 0
                        for other in predators:
                            if other != new_predator:
                                distance = ((new_predator.x - other.x)**2 + (new_predator.y - other.y)**2) ** 0.5
                                if distance < 20:  # Reward if near other predators
                                    group_bonus += 1

                        # Calculate the reward
                        reward = new_predator.neural_network.calculate_reward(
                            captured_prey=new_predator is not None,
                            previous_distance=0.0,  # Adjust as needed
                            current_distance=0.0,
                            energy_used=new_predator.energy,
                            split_happened=True,
                            speed=speed,
                            group_bonus=group_bonus
                        )
                        new_predator.neural_network.total_reward += reward

                        # Compare ranks
                
                        new_predator.rank = new_predator.neural_network.calculate_rank()
                        #best_rank["Predator"] = max(predator.rank for predator in predators) 
                    

                        if new_predator.rank >= best_rank["Predator"]:
                            best_rank["Predator"] = new_predator.rank
                            new_predator.neural_network.save(save_file, new_predator.generation, best_rank["Predator"])

                        return new_predator
                return None

    def process_single_sensor(args):
        """Processa um único sensor com base nos argumentos fornecidos."""
        angle, self_position, vision_distance, preys, predators = args
        closest_distance = vision_distance
        target_type = None

        # Checks for collision with prey
        for prey in preys:
            dx = prey.x - self_position[0]
            dy = prey.y - self_position[1]
            distance = (dx**2 + dy**2) ** 0.5  # Calculate the distance
            if distance < closest_distance:
                closest_distance = distance
                target_type = "prey"

        # Checks for collision with predators
        for predator in predators:
            dx = predator.x - self_position[0]
            dy = predator.y - self_position[1]
            distance = (dx**2 + dy**2) ** 0.5  # Calculate the distance
            if distance < closest_distance:
                closest_distance = distance
                target_type = "predator"

        # Calculate the sensor endpoint
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
        num_sensors = 4  # Number of rays in the field of view
        vision_distance = 0.8 * min(SCREEN_WIDTH, SCREEN_HEIGHT)  # Maximum sensor distance
        angle_step = 45 / num_sensors  # Angle between sensors within field of view (45 degrees)

        # Pre-calculate sensor angles
        sensor_angles = [radians(self.direction - 22.5 + i * angle_step) for i in range(num_sensors)]

        # Adjusted home position for sensors
        start_x = self.x + cos(radians(self.direction)) * self.size
        start_y = self.y + sin(radians(self.direction)) * self.size
        self_position = np.array([start_x, start_y])  # Vector of the initial position of the sensor

        # Process each sensor individually
        for angle in sensor_angles:
            closest_distance = vision_distance  # Initially the sensor has maximum range
            target_type = None  # Target type detected

            # Sensor direction
            sensor_dx = cos(angle)
            sensor_dy = sin(angle)
            sensor_angle_degrees = np.degrees(np.arctan2(sensor_dy, sensor_dx))

            # Checks for collision with prey (vectorized)
            if preys:
                prey_positions = np.array([[prey.x, prey.y] for prey in preys])
                prey_vectors = prey_positions - self_position  # Vectors for each prey
                prey_distances = np.linalg.norm(prey_vectors, axis=1)  # Prey distances

                # Filter prey within field of view
                prey_angles = np.degrees(np.arctan2(prey_vectors[:, 1], prey_vectors[:, 0]))
                delta_angles = np.abs((prey_angles - sensor_angle_degrees + 180) % 360 - 180)

                # Prey visible by sensor
                valid_preys = (prey_distances < vision_distance) & (delta_angles <= angle_step / 2)
                if valid_preys.any():
                    closest_prey_index = np.argmin(prey_distances[valid_preys])
                    closest_distance = prey_distances[valid_preys][closest_prey_index]
                    target_type = "prey"

            # Checks for collision with predators (vectorized)
            if predators:
                predator_positions = np.array([[predator.x, predator.y] for predator in predators if predator != self])
                if len(predator_positions) > 0:  # Check for predators in range
                    predator_vectors = predator_positions - self_position  # Vectors for each predator
                    predator_distances = np.linalg.norm(predator_vectors, axis=1)  # Distances from predators

                    # Filter predators within the field of view
                    predator_angles = np.degrees(np.arctan2(predator_vectors[:, 1], predator_vectors[:, 0]))
                    delta_angles = np.abs((predator_angles - sensor_angle_degrees + 180) % 360 - 180)

                    # Predators visible by the sensor
                    valid_predators = (predator_distances < vision_distance) & (delta_angles <= angle_step / 2)
                    if valid_predators.any():
                        closest_predator_index = np.argmin(predator_distances[valid_predators])
                        closest_distance = predator_distances[valid_predators][closest_predator_index]
                        target_type = "predator"

            # Calculate the sensor endpoint based on the closest distance
            end_x = start_x + sensor_dx * closest_distance
            end_y = start_y + sensor_dy * closest_distance

            # Add the sensor to the result
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
            prey.rank = prey_nn.rank
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
            predator.rank = predator_nn.rank
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
    saved_rank = 0.0

    if save_file:
        try:
            saved_neural_network = NeuralNetwork.load(save_file)
            saved_generation = saved_neural_network.generation  # Load generation from file
            saved_rank = saved_neural_network.rank
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
            prey.rank = saved_rank
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
            predator.rank = saved_rank
            if saved_neural_network:
                predator.neural_network = saved_neural_network
        predators.extend(new_predators)

    # Add new entities to the grid
    for entity in new_preys + new_predators:
        grid.add_entity(entity)




