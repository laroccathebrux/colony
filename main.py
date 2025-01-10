import pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import os
from collections import deque
from constants import *
from entities import create_entities, Prey, Predator
from math import cos, sin, radians

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Prey and Predator Simulator")

# Create a figure for the matplotlib plot
fig, ax = plt.subplots(figsize=(5, 2))
canvas = FigureCanvas(fig)

# Real-time data storage
prey_data = deque(maxlen=MAX_DATA_POINTS)
predator_data = deque(maxlen=MAX_DATA_POINTS)
time_data = deque(maxlen=MAX_DATA_POINTS)

for i in range(MAX_DATA_POINTS):
    prey_data.append(0)
    predator_data.append(0)
    time_data.append(i)

def ensure_directory_exists(directory):
    """Ensure the specified directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_screenshot(surface):
    """Save a screenshot of the initial state to the img/evolve_xxx directory."""
    directory = "img"
    ensure_directory_exists(directory)

    # Find the next available increment
    increment = 1
    while os.path.exists(f"{directory}/evolve_{increment:03d}.png"):
        increment += 1

    filename = f"{directory}/evolve_{increment:03d}.png"
    pygame.image.save(surface, filename)
    print(f"Screenshot saved to {filename}")

def draw_sensors(surface, entity):
    """Draw the sensors of the selected entity."""
    if entity and hasattr(entity, 'sensors'):
        # Define directions for Prey and angles for Predator
        if isinstance(entity, Prey):
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Front, back, left, right
        else:  # Predator
            angles = [0, -45, 45, -90]  # Angles for sensors
            directions = [(cos(radians(angle)), -sin(radians(angle))) for angle in angles]

        for (dx, dy), (distance, target) in zip(directions, entity.sensors):
            end_x = int(entity.x + distance * dx)
            end_y = int(entity.y + distance * dy)

            # Determine sensor color based on the target
            if target == 'PREY':
                color = (0, 255, 0)
            elif target == 'PREDATOR':
                color = (255, 0, 0)
            elif target == 'WALL':
                color = (0, 0, 255)
            else:
                color = (255, 255, 255)

            # Draw the sensor ray
            pygame.draw.line(surface, color, (entity.x, entity.y), (end_x, end_y), 2)

def draw_sidebar(surface, selected_entity):
    if selected_entity and selected_entity.energy <= 0:
        selected_entity = None
    """Draw the sidebar on the right side of the screen."""
    pygame.draw.rect(surface, SIDEBAR_COLOR, (SCREEN_WIDTH - SIDEBAR_WIDTH, 0, SIDEBAR_WIDTH, SIDEBAR_HEIGHT))
    font = pygame.font.Font(None, 24)
    text = font.render("Simulation Info", True, (0, 0, 0))
    surface.blit(text, (SCREEN_WIDTH - SIDEBAR_WIDTH + 20, 20))

    if selected_entity:
        energy_label = font.render("ENERGY", True, (0, 0, 0))
        surface.blit(energy_label, (SCREEN_WIDTH - SIDEBAR_WIDTH + 20, 60))

        # Draw energy bar
        bar_width = SIDEBAR_WIDTH - 40
        bar_height = 20
        energy_percentage = selected_entity.energy / 100
        pygame.draw.rect(surface, (200, 0, 0), (SCREEN_WIDTH - SIDEBAR_WIDTH + 20, 90, bar_width, bar_height))
        pygame.draw.rect(surface, (0, 200, 0), (SCREEN_WIDTH - SIDEBAR_WIDTH + 20, 90, int(bar_width * energy_percentage), bar_height))

        # Draw SPLIT info
        split_label = font.render("SPLIT", True, (0, 0, 0))
        surface.blit(split_label, (SCREEN_WIDTH - SIDEBAR_WIDTH + 20, 130))

        # Calcular a porcentagem de SPLIT com base no tipo de entidade
        if hasattr(selected_entity, 'split_progress'):
            if isinstance(selected_entity, Predator):
                # Predador: Normalizar de 0 a 3
                split_percentage = selected_entity.split_progress / 3  # Normaliza para 0 a 1
            elif isinstance(selected_entity, Prey):
                # Prey: Normalizar de 0 a 100%
                split_percentage = selected_entity.split_progress / 100  # Normaliza para 0 a 1
        else:
            split_percentage = 0

        # Desenhar a barra de progresso
        pygame.draw.rect(
            surface, 
            (200, 200, 200), 
            (SCREEN_WIDTH - SIDEBAR_WIDTH + 20, 160, bar_width, bar_height)
        )
        pygame.draw.rect(
            surface, 
            (0, 0, 200), 
            (SCREEN_WIDTH - SIDEBAR_WIDTH + 20, 160, int(bar_width * split_percentage), bar_height)
        )

        # Draw DIGESTION info (only for predators)
        if hasattr(selected_entity, 'digestion'):
            digestion_label = font.render("DIGESTION", True, (0, 0, 0))
            surface.blit(digestion_label, (SCREEN_WIDTH - SIDEBAR_WIDTH + 20, 200))
            digestion_percentage = selected_entity.digestion / 100
            pygame.draw.rect(surface, (200, 200, 200), (SCREEN_WIDTH - SIDEBAR_WIDTH + 20, 230, bar_width, bar_height))
            pygame.draw.rect(surface, (200, 100, 0), (SCREEN_WIDTH - SIDEBAR_WIDTH + 20, 230, int(bar_width * digestion_percentage), bar_height))

        # Sensors Information
        '''
        sensor_label = font.render("SENSORS", True, (0, 0, 0))
        surface.blit(sensor_label, (SCREEN_WIDTH - SIDEBAR_WIDTH + 20, 270))
        for i, (distance, target) in enumerate(selected_entity.sensors):
            sensor_info = font.render(f"{i + 1}: {distance} ({target})", True, (0, 0, 0))
            surface.blit(sensor_info, (SCREEN_WIDTH - SIDEBAR_WIDTH + 20, 300 + i * 20))
        '''

def draw_bottom_bar(surface):
    """Draw the bottom bar containing the matplotlib graph."""
    pygame.draw.rect(surface, BOTTOMBAR_COLOR, (0, SCREEN_HEIGHT - BOTTOMBAR_HEIGHT, SCREEN_WIDTH, BOTTOMBAR_HEIGHT))

    ax.clear()
    ax.plot(range(len(time_data)), list(prey_data), label="Prey")
    ax.plot(range(len(time_data)), list(predator_data), label="Predators")
    ax.set_title("Population Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Count")
    ax.set_xlim(0, MAX_DATA_POINTS)
    ax.set_ylim(0, max(max(prey_data, default=0), max(predator_data, default=0)) + 10)
    ax.legend()

    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()

    # Create a pygame surface from the matplotlib figure
    graph_surface = pygame.image.fromstring(raw_data, size, "RGB")
    surface.blit(graph_surface, (0, SCREEN_HEIGHT - BOTTOMBAR_HEIGHT))

def draw_neural_network(surface, neural_network, x_offset, y_offset):
    """Draw the neural network of a selected entity."""
    # Dimensions
    layer_width = 50  # Width per neuron layer
    neuron_radius = 5
    spacing_y = 12  # Vertical spacing between neurons
    spacing_x = 100  # Horizontal spacing between layers

    # Colors
    input_color = lambda active: INPUT_ACTIVE_COLOR if active else INPUT_DEACTIVE_COLOR
    neuron_color = lambda active: NEURON_ACTIVE_COLOR if active else NEURON_DEACTIVE_COLOR
    output_color = lambda active: OUTPUT_ACTIVE_COLOR if active else OUTPUT_DEACTIVE_COLOR
    connection_color = lambda active: CONNECTION_ACTIVE_COLOR if active else CONNECTION_DEACTIVE_COLOR

    # Inputs
    for i, input_value in enumerate(neural_network.inputs):
        color = input_color(input_value > 0.5)
        x = x_offset
        y = y_offset + i * spacing_y
        pygame.draw.circle(surface, color, (x, y), neuron_radius)

    # Hidden Layer
    hidden_layer_x = x_offset + spacing_x
    for i, hidden_value in enumerate(neural_network.hidden_layer):
        color = neuron_color(hidden_value > 0.5)
        x = hidden_layer_x
        y = y_offset + i * spacing_y
        pygame.draw.circle(surface, color, (x, y), neuron_radius)

        # Connections from inputs to hidden layer
        for j, input_value in enumerate(neural_network.inputs):
            connection_color_value = connection_color(input_value > 0.5 and hidden_value > 0.5)
            pygame.draw.line(
                surface,
                connection_color_value,
                (x_offset, y_offset + j * spacing_y),
                (x, y),
                1,
            )

    # Outputs
    output_layer_x = hidden_layer_x + spacing_x
    for i, output_value in enumerate(neural_network.outputs):
        color = output_color(output_value > 0.5)
        x = output_layer_x
        y = y_offset + i * spacing_y
        pygame.draw.circle(surface, color, (x, y), neuron_radius)

        # Connections from hidden to output layer
        for j, hidden_value in enumerate(neural_network.hidden_layer):
            connection_color_value = connection_color(hidden_value > 0.5 and output_value > 0.5)
            pygame.draw.line(
                surface,
                connection_color_value,
                (hidden_layer_x, y_offset + j * spacing_y),
                (x, y),
                1,
            )

def main():
    """Main simulation loop."""
    running = True
    clock = pygame.time.Clock()
    preys, predators, grid = create_entities(randomize_energy=True)
    selected_entity = None

    frame_count = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left mouse button
                mouse_x, mouse_y = event.pos
                selected_entity = None
                for entity in preys + predators:
                    if ((mouse_x - entity.x) ** 2 + (mouse_y - entity.y) ** 2) ** 0.5 <= entity.size:
                        selected_entity = entity
                        break

        # Update simulation
        for prey in preys[:]:
            #prey.update_sensors(grid, frame_count)
            old_x, old_y = prey.move()
            if old_x is not None:
                grid.update_entity(prey, old_x, old_y)
            prey.update_energy()
            prey.update_split(preys)

        for predator in predators[:]:
            #predator.update_sensors(grid, frame_count)
            old_x, old_y = predator.move()
            if old_x is not None:
                grid.update_entity(predator, old_x, old_y)
            predator.update_energy()
            new_predator = predator.eat_prey(preys, grid)
            if new_predator:
                predators.append(new_predator)
                grid.add_entity(new_predator)
            
        # Remove dead predators
        for predator in predators[:]:
            if predator.energy <= 0:
                grid.remove_entity(predator)
                predators.remove(predator)
                del predator

        # Update data for the graph
        prey_data.append(len(preys))
        predator_data.append(len(predators))
        time_data.append(frame_count)

        # Draw the main screen
        screen.fill(BACKGROUND_COLOR)

        # Draw entities and sensors
        for prey in preys:
            #prey.update_sensors(grid, frame_count)  # Update sensors for prey
            color = SELECT_ENTITY_COLOR if prey == selected_entity else prey.color
            pygame.draw.circle(screen, color, (prey.x, prey.y), prey.size)
            if prey == selected_entity:
                #draw_sensors(screen, prey)
                continue

        for predator in predators:
            #predator.update_sensors(grid, frame_count)  # Update sensors for predator
            color = SELECT_ENTITY_COLOR if predator == selected_entity else predator.color
            pygame.draw.circle(screen, color, (predator.x, predator.y), predator.size)
            if predator == selected_entity:
                #draw_sensors(screen, predator)
                continue

        # Draw sidebar
        draw_sidebar(screen, selected_entity)

        # Draw bottom bar
        draw_bottom_bar(screen)

        # Desenhar rede neural da entidade selecionada
        if selected_entity:
            draw_neural_network(
                screen,
                selected_entity.neural_network,
                SCREEN_WIDTH - 400,  # X offset (ao lado do gráfico)
                SCREEN_HEIGHT - BOTTOMBAR_HEIGHT + 20,  # Y offset
            )

        # Update the display
        pygame.display.flip()

        # Increment frame count
        frame_count += 1

        # Cap the frame rate
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()