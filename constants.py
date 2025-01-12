# SIMULATOR CONSTANTS
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
BACKGROUND_COLOR = (0, 0, 0)
SIDEBAR_WIDTH = 200
SIDEBAR_HEIGHT = SCREEN_HEIGHT
SIDEBAR_COLOR = (192, 192, 192)
BOTTOMBAR_WIDTH = SCREEN_WIDTH
BOTTOMBAR_HEIGHT = 200
BOTTOMBAR_COLOR = (128, 128, 128)
MAX_DATA_POINTS = 100
SELECT_ENTITY_COLOR = (255, 255, 255)
PADDING_SPACE = 30

# ENTITIES CONSTANTS
ENTITY_SIZE = 5
# PREY
PREY_COLOR = (0, 255, 0)
PREY_ENERGY_RECOVERY_RATE = 0.5
PREY_ENERGY_DEPLETION_RATE = 0.4
PREY_SPLIT_RATE = 0.8
PREY_MAX = 100
PREY_INITIAL_POPULATION = 100
PREY_SPEED = 1
# PREDATOR
PREDATOR_COLOR = (255, 0, 0)
PREDATOR_ENERGY_DEPLETION_RATE = 0.2
DIGESTION_DEPLETION_RATE = 3
PREDATOR_MAX = 30
PREDATOR_INITIAL_POPULATION = 30
PREDATOR_SPEED = 2

# NEURAL NETWORK CONSTANTS
INPUTS = 14
INPUTS_DATA = ['SENSOR_1', 'SENSOR_1_ENTITY', 'SENSOR_2', 'SENSOR_2_ENTITY', 'SENSOR_3', 'SENSOR_3_ENTITY', 'SENSOR_4', 'SENSOR_4_ENTITY', 'BIAS', 'ENERGY', 'DIGESTION', 'SPLIT', 'X', 'Y']
NEURONS = 16
OUTPUTS = 2
OUTPUTS_DATA = ['SPEED', 'ANGULAR_VELOCITY']
MUTATION_RATE = 0.005

# NEURAL NETWORK LAYERS GRAPH CONSTANTS
INPUT_ACTIVE_COLOR = (0, 255, 0)
INPUT_DEACTIVE_COLOR = (0, 128, 0)
NEURON_ACTIVE_COLOR = (255, 0, 0)
NEURON_DEACTIVE_COLOR = (128, 0, 0)
OUTPUT_ACTIVE_COLOR = (0, 0, 255)
OUTPUT_DEACTIVE_COLOR = (0, 0, 128)
CONNECTION_ACTIVE_COLOR = (255, 255, 255)
CONNECTION_DEACTIVE_COLOR = (128, 128, 128)