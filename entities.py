import pygame
import random
from math import cos, sin, radians
from constants import *
from neural_network import NeuralNetwork

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
    def __init__(self, x, y, energy=None):
        self.x = x
        self.y = y
        self.color = PREY_COLOR
        self.size = ENTITY_SIZE
        self.energy = energy if energy is not None else random.randint(40, 100)
        self.active = True  # Whether the prey is moving
        self.split_progress = 0  # Progress towards splitting
        self.sensors = [0, 0, 0, 0]  # Sensor distances for [front, back, left, right]
        self.neural_network = NeuralNetwork()  # Neural network instance
        self.is_stationary = 0  # Whether the prey is stationary

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (self.x, self.y), self.size)

    def move(self):
        if self.active:
            inputs = self._prepare_inputs()
            speed, angular_velocity = self.neural_network.forward(inputs)

            # Normalize outputs
            speed = speed * 5  # Scale speed to a maximum of 5
            # Constrain angular velocity to reduce rotation
            max_angular_velocity = 0.1  # Adjust this value to allow slight rotations
            angular_velocity = max(-max_angular_velocity, min(max_angular_velocity, angular_velocity - 0.5)) * 2 * 3.14159

            # Convert speed and angular velocity into movement
            dx = int(speed * cos(angular_velocity * 2 * 3.14159))
            dy = int(speed * sin(angular_velocity * 2 * 3.14159))

            # Check if the object is stationary
            self.is_stationary = (dx == 0 and dy == 0)

            old_x, old_y = self.x, self.y
            self.x += dx
            self.y += dy

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

    def update_split(self, preys):
        if self.split_progress < 100:
            self.split_progress += PREY_SPLIT_RATE
        if self.split_progress >= 100 and self.energy >= 30:
            self.split_progress = 0
            new_prey = Prey(self.x + random.randint(-10, 10), self.y + random.randint(-10, 10))
            new_prey.neural_network = NeuralNetwork()  # Create new neural network for offspring
            new_prey.neural_network.mutate(MUTATION_RATE)  # Apply mutation with a 10% rate
            preys.append(new_prey)

    def update_sensors(self, grid, frame_count):
        """Update sensor distances and targets."""
        if frame_count % 5 != 0:  # Update sensors every 5 frames
            return
        
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  if isinstance(self, Prey) else [
            (cos(radians(angle)), -sin(radians(angle))) for angle in [0, -45, 45, -90]
        ] # Front, back, left, right

        self.sensors = []

        for dx, dy in directions:
            try:
                distance, target = self._cast_ray(dx, dy, grid)
                self.sensors.append((distance, target))
            except Exception as e:
                # Handle unexpected issues gracefully
                self.sensors.append((50, None))  # Default to max distance, no target

    def _cast_ray(self, dx, dy, grid):
        """Optimized ray casting to reduce processing time."""
        max_distance = 50 if isinstance(self, Prey) else 150  # Short for prey, long for predators
        x, y = self.x, self.y

        for d in range(1, max_distance):
            x += dx
            y += dy

            # Check for walls
            if x < 0 or x >= SCREEN_WIDTH or y < 0 or y >= SCREEN_HEIGHT - BOTTOMBAR_HEIGHT:
                return d, 'WALL'

            # Check for nearby entities in relevant grid cells
            for entity in grid.get_nearby_entities(x, y):
                if entity is not self and abs(entity.x - x) <= self.size and abs(entity.y - y) <= self.size:
                    return d, 'PREY' if isinstance(entity, Prey) else 'PREDATOR'      
                                                  
        return max_distance, None

    def _prepare_inputs(self):
        """Prepare inputs for the neural network."""
        inputs = []
        for sensor in self.sensors:
            if isinstance(sensor, tuple) and len(sensor) == 2:
                distance, target = sensor
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
    def __init__(self, x, y, energy=None):
        self.x = x
        self.y = y
        self.color = PREDATOR_COLOR
        self.size = ENTITY_SIZE
        self.energy = energy if energy is not None else random.randint(50, 100)
        self.split_progress = 0  # Number of successful prey eaten for splitting
        self.digestion = 0  # Digestion timer
        self.sensors = [0, 0, 0, 0]  # Sensor distances for 4 angled directions
        self.neural_network = NeuralNetwork()  # Neural network instance

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (self.x, self.y), self.size)

    def move(self):
        inputs = self._prepare_inputs()
        speed, angular_velocity = self.neural_network.forward(inputs)

        # Normalize outputs
        speed = speed * 5  # Scale speed to a maximum of 5
        angular_velocity = (angular_velocity - 0.5) * 2 * 3.14159  # Scale to [-pi, pi]

        # Convert speed and angular velocity into movement
        dx = int(speed * cos(angular_velocity * 2 * 3.14159))
        dy = int(speed * sin(angular_velocity * 2 * 3.14159))

        old_x, old_y = self.x, self.y
        self.x += dx
        self.y += dy

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

    def eat_prey(self, preys, grid):
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
                    if self.split_progress >= 3:
                        self.split_progress = 0
                        new_predator = Predator(self.x + random.randint(-10, 10), self.y + random.randint(-10, 10))
                        new_predator.neural_network = NeuralNetwork()  # Create new neural network for offspring
                        new_predator.neural_network.mutate(MUTATION_RATE)  # Apply mutation with a 10% rate
                        return new_predator
                return None

    def update_sensors(self, grid, frame_count):
        """Update sensor distances and targets."""
        if frame_count % 5 != 0:  # Update sensors every 5 frames
            return

        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)] if isinstance(self, Prey) else [
            (cos(radians(angle)), -sin(radians(angle))) for angle in [0, -45, 45, -90]
        ]
        self.sensors = []

        for dx, dy in directions:
            try:
                distance, target = self._cast_ray(dx, dy, grid)
                self.sensors.append((distance, target))
            except Exception as e:
                # Handle unexpected issues gracefully
                self.sensors.append((150, None))  # Default to max distance, no target

    def _cast_ray(self, dx, dy, grid):
        """Cast a ray in the given direction and return distance and target type."""
        max_distance = 150 if isinstance(self, Prey) else 50 # Long range

        x, y = self.x, self.y

        for d in range(1, max_distance):
            x += dx
            y += dy

            # Check for walls
            if x < 0 or x >= SCREEN_WIDTH or y < 0 or y >= SCREEN_HEIGHT - BOTTOMBAR_HEIGHT:
                return d, 'WALL'
            
            # Check for nearby entities in relevant grid cells
            for entity in grid.get_nearby_entities(x, y):
                if entity is not self and abs(entity.x - x) <= self.size and abs(entity.y - y) <= self.size:
                    return d, 'PREY' if isinstance(entity, Prey) else 'PREDATOR'
                
        return max_distance, None

    def _prepare_inputs(self):
        """Prepare inputs for the neural network."""
        inputs = []
        for sensor in self.sensors:
            if isinstance(sensor, tuple) and len(sensor) == 2:
                distance, target = sensor
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
    preys = [Prey(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT - BOTTOMBAR_HEIGHT),
                  random.randint(40, 100) if randomize_energy else 100)
             for _ in range(PREY_INITIAL_POPULATION)]
    predators = [Predator(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT - BOTTOMBAR_HEIGHT),
                          random.randint(50, 100) if randomize_energy else 100)
                 for _ in range(PREDATOR_INITIAL_POPULATION)]
    for entity in preys + predators:
        grid.add_entity(entity)
    return preys, predators, grid
