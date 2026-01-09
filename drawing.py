
import pygame
import numpy as np

    # Initialize Pygame
pygame.init()

# Grid dimensions
GRID_SIZE = 28
PIXEL_SIZE = 10  # Size of each "pixel" on screen for easier drawing
WINDOW_SIZE = GRID_SIZE * PIXEL_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (50, 50, 50)

# Set up the display
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("MNIST Digit Drawer")

# Initialize the pixel grid (representing the 28x28 image)
pixel_data = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

running = True
drawing = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: # Left click
                drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                x, y = event.pos
                grid_x = x // PIXEL_SIZE
                grid_y = y // PIXEL_SIZE
                if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
                    pixel_data[grid_y, grid_x] = 255 # Draw white

    # Clear the screen
    screen.fill(BLACK)

    # Draw the grid lines
    for i in range(GRID_SIZE + 1):
        pygame.draw.line(screen, GRAY, (i * PIXEL_SIZE, 0), (i * PIXEL_SIZE, WINDOW_SIZE))
        pygame.draw.line(screen, GRAY, (0, i * PIXEL_SIZE), (WINDOW_SIZE, i * PIXEL_SIZE))

    # Draw the pixels based on pixel_data
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if pixel_data[y, x] == 255:
                pygame.draw.rect(screen, WHITE, (x * PIXEL_SIZE, y * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))

    pygame.display.flip()

pygame.quit()

# You can now save or process 'pixel_data' as your MNIST image
print(pixel_data)