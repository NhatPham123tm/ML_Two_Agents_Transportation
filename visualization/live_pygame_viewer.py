
# real-time Pygame viewer for step logs
from typing import List, Tuple
import pygame

def run(step_log: List[dict], grid_shape: Tuple[int, int], obstacles: List[Tuple[int, int]], cell_px: int = 48) -> None:
    H, W = grid_shape
    pygame.init()
    screen = pygame.display.set_mode((W * cell_px, H * cell_px))
    clock = pygame.time.Clock()
    i, paused, speed = 0, False, 10
    running = True

    def draw_cell(r, c):
        rect = pygame.Rect(c * cell_px, r * cell_px, cell_px, cell_px)
        pygame.draw.rect(screen, (230, 230, 230), rect, 0)
        pygame.draw.rect(screen, (200, 200, 200), rect, 1)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    i = min(i + 1, len(step_log) - 1)
                elif event.key == pygame.K_LEFT:
                    i = max(i - 1, 0)
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed = min(60, speed + 1)
                elif event.key == pygame.K_MINUS:
                    speed = max(1, speed - 1)
                elif event.key == pygame.K_r:
                    i = 0
        if not paused:
            i = min(i + 1, len(step_log) - 1)

        screen.fill((255, 255, 255))
        # grid
        for r in range(H):
            for c in range(W):
                draw_cell(r, c)
        # obstacles
        for (r, c) in obstacles:
            rect = pygame.Rect(c * cell_px, r * cell_px, cell_px, cell_px)
            pygame.draw.rect(screen, (120, 120, 120), rect)

        if step_log:
            row = step_log[i]
            rF, cF = row['pos_F']
            rM, cM = row['pos_M']
            pygame.draw.circle(screen, (50, 100, 220), (cF * cell_px + cell_px // 2, rF * cell_px + cell_px // 2), cell_px // 3)
            pygame.draw.circle(screen, (220, 60, 60), (cM * cell_px + cell_px // 2, rM * cell_px + cell_px // 2), cell_px // 3)

        pygame.display.flip()
        clock.tick(speed)

    pygame.quit()
