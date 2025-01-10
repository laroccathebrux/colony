# Predator and Prey Simulation with Neural Networks

This project is a simulation of an ecosystem with predators and prey. The entities use artificial neural networks to define their behaviors, including movement, sensor-based decisions, and interactions such as hunting and evasion. The project is developed in Python, using the `pygame` library for visualization.

## Features

### Entities
1. **Predators (red):**
   - Move using a neural network.
   - Consume energy while moving.
   - Hunt prey to regain energy.
   - Die if their energy reaches zero.
   - Can split (reproduce) after eating 3 prey, inheriting and mutating the brain of the original entity.
   - Have a digestion cooldown after consuming prey, during which no other prey can be eaten.
   - Have front-facing sensors to detect prey, walls, or other predators.

2. **Prey (green):**
   - Move using a neural network.
   - Consume energy while moving.
   - Completely stop moving when energy reaches zero and resume movement after recovering 60% energy.
   - Can split after reaching 100% division progress, inheriting and mutating the brain of the original entity.
   - Have 360° sensors to detect predators, walls, and other prey.

### Spatial Partitioning
- A **Spatial Partitioning** system (grid) is implemented to reduce the number of checks between entities. This optimizes collision and sensor calculations by limiting interactions to nearby entities.

### Neural Networks
- Each entity has a neural network with:
  - **Inputs:** Sensor distances, detected entity, BIAS, energy, digestion, split progress, X and Y position.
  - **Outputs:** Linear and angular velocity.
  - **Mutation:** Applied during splitting, including changes to weights, new connections, or new neurons.

### Visualization
- The simulation displays:
  - Active sensors (colored lines) for the selected entity.
  - Bars showing energy, digestion (for predators), and division progress.
  - Population graph.
  - Neural network of the selected individual.

---

## How to Run the Project

### Prerequisites
- Python 3.10 or later.
- `pygame` library.

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/repository-name.git
    cd repository-name
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the project:
    ```bash
    python main.py
    ```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.