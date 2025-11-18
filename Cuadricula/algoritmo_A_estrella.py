import pygame
from heapq import heappush, heappop
import math

pygame.init()

# Configuración
ANCHO = 800
ALTO = 600
FILAS = 11  # Cambia el número de filas
COLUMNAS = 11  # Cambia el número de columnas
VENTANA = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("A* Pathfinding con Diagonal")

# Colores
BLANCO, NEGRO, GRIS = (255, 255, 255), (0, 0, 0), (128, 128, 128)
VERDE, ROJO, NARANJA, PURPURA, CIAN = (0, 255, 0), (255, 0, 0), (255, 165, 0), (128, 0, 128), (64, 224, 208)

class Nodo:
    def __init__(self, fila, col, ancho, alto):
        self.fila, self.col = fila, col
        self.x, self.y = col * ancho, fila * alto
        self.color = BLANCO
        self.ancho = ancho
        self.alto = alto
        self.vecinos = []

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.alto))

    def actualizar_vecinos(self, grid, filas, columnas):
        self.vecinos = []
        # Movimientos: arriba, abajo, izquierda, derecha y las 4 diagonales
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),  # Ortogonales
                (1, 1), (1, -1), (-1, 1), (-1, -1)]  # Diagonales
        
        for df, dc in dirs:
            f, c = self.fila + df, self.col + dc
            if 0 <= f < filas and 0 <= c < columnas and grid[f][c].color != NEGRO:
                # Para movimiento diagonal, verificar que no haya obstáculos adyacentes
                if df != 0 and dc != 0:  # Es diagonal
                    if grid[self.fila + df][self.col].color == NEGRO or grid[self.fila][self.col + dc].color == NEGRO:
                        continue  # No permitir diagonal si hay obstáculo en el camino
                self.vecinos.append(grid[f][c])

def crear_grid(filas, columnas, ancho_ventana, alto_ventana):
    ancho_nodo = ancho_ventana // columnas
    alto_nodo = alto_ventana // filas
    return [[Nodo(i, j, ancho_nodo, alto_nodo) for j in range(columnas)] for i in range(filas)]

def dibujar(ventana, grid, filas, columnas, ancho_ventana, alto_ventana):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)
    
    ancho_nodo = ancho_ventana // columnas
    alto_nodo = alto_ventana // filas
    
    for i in range(filas + 1):
        pygame.draw.line(ventana, GRIS, (0, i * alto_nodo), (ancho_ventana, i * alto_nodo))
    for j in range(columnas + 1):
        pygame.draw.line(ventana, GRIS, (j * ancho_nodo, 0), (j * ancho_nodo, alto_ventana))
    
    pygame.display.update()

def heuristica(a, b):
    # Distancia euclidiana para mejor precisión con diagonales
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def a_estrella(ventana, grid, inicio, fin, filas, columnas, ancho_ventana, alto_ventana):
    contador = 0
    open_heap = [(0, contador, inicio)]
    came_from = {}
    g_score = {nodo: float("inf") for fila in grid for nodo in fila}
    g_score[inicio] = 0
    open_set = {inicio}

    while open_heap:
        actual = heappop(open_heap)[2]
        open_set.discard(actual)

        if actual == fin:
            # Reconstruir camino
            while actual in came_from:
                actual = came_from[actual]
                if actual.color not in [NARANJA, PURPURA]:
                    actual.color = CIAN
            return True

        for vecino in actual.vecinos:
            # Costo diagonal es √2 ≈ 1.414, costo ortogonal es 1
            es_diagonal = abs(vecino.fila - actual.fila) == 1 and abs(vecino.col - actual.col) == 1
            costo = math.sqrt(2) if es_diagonal else 1
            temp_g = g_score[actual] + costo
            
            if temp_g < g_score[vecino]:
                came_from[vecino] = actual
                g_score[vecino] = temp_g
                if vecino not in open_set:
                    contador += 1
                    f_score = temp_g + heuristica((vecino.fila, vecino.col), (fin.fila, fin.col))
                    heappush(open_heap, (f_score, contador, vecino))
                    open_set.add(vecino)
                    if vecino.color not in [NARANJA, PURPURA]:
                        vecino.color = VERDE

        if actual.color not in [NARANJA, PURPURA]:
            actual.color = ROJO
        dibujar(ventana, grid, filas, columnas, ancho_ventana, alto_ventana)

    return False

def main():
    grid = crear_grid(FILAS, COLUMNAS, ANCHO, ALTO)
    inicio, fin = None, None
    corriendo = True

    while corriendo:
        dibujar(VENTANA, grid, FILAS, COLUMNAS, ANCHO, ALTO)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if pygame.mouse.get_pressed()[0]:  # Click izquierdo
                x, y = pygame.mouse.get_pos()
                fila = y // (ALTO // FILAS)
                col = x // (ANCHO // COLUMNAS)
                
                if 0 <= fila < FILAS and 0 <= col < COLUMNAS:
                    nodo = grid[fila][col]
                    
                    if not inicio and nodo.color == BLANCO:
                        inicio = nodo
                        nodo.color = NARANJA
                    elif not fin and nodo != inicio and nodo.color == BLANCO:
                        fin = nodo
                        nodo.color = PURPURA
                    elif nodo != inicio and nodo != fin:
                        nodo.color = NEGRO

            elif pygame.mouse.get_pressed()[2]:  # Click derecho
                x, y = pygame.mouse.get_pos()
                fila = y // (ALTO // FILAS)
                col = x // (ANCHO // COLUMNAS)
                
                if 0 <= fila < FILAS and 0 <= col < COLUMNAS:
                    nodo = grid[fila][col]
                    nodo.color = BLANCO
                    if nodo == inicio:
                        inicio = None
                    elif nodo == fin:
                        fin = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Limpiar cuadrícula (mantiene inicio, fin y paredes)
                    for fila in grid:
                        for nodo in fila:
                            if nodo.color not in [NARANJA, PURPURA, NEGRO]:
                                nodo.color = BLANCO
                    
                    # Ejecutar A* si hay inicio y fin
                    if inicio and fin:
                        for fila in grid:
                            for nodo in fila:
                                nodo.actualizar_vecinos(grid, FILAS, COLUMNAS)
                        a_estrella(VENTANA, grid, inicio, fin, FILAS, COLUMNAS, ANCHO, ALTO)

                if event.key == pygame.K_c:
                    inicio, fin = None, None
                    grid = crear_grid(FILAS, COLUMNAS, ANCHO, ALTO)

    pygame.quit()

main()