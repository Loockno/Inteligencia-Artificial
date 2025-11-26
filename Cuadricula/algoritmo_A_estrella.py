import pygame
from heapq import heappush, heappop
import math

# --- Configuración Inicial ---
pygame.init()

ANCHO_VENTANA = 800
ALTO_VENTANA = 600
FILAS = 11        # Aumenté un poco la resolución para ver mejor el camino
COLUMNAS = 11
ANCHO_NODO = ANCHO_VENTANA // COLUMNAS
ALTO_NODO = ALTO_VENTANA // FILAS

VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ALTO_VENTANA))
pygame.display.set_caption("A* Pathfinding Optimizado (Euclidiano)")

# Colores (RGB)
BLANCO = (255, 255, 255)
NEGRO = (20, 20, 20)      # Pared
GRIS = (200, 200, 200)    # Líneas grid
VERDE = (0, 255, 128)     # Open Set (Nodos considerados)
ROJO = (255, 100, 100)    # Closed Set (Nodos visitados)
NARANJA = (255, 165, 0)   # Inicio
TURQUESA = (64, 224, 208) # Fin
AZUL = (50, 150, 255)     # Camino final

class Nodo:
    def __init__(self, fila, col):
        self.fila = fila
        self.col = col
        self.x = col * ANCHO_NODO
        self.y = fila * ALTO_NODO
        self.color = BLANCO
        self.vecinos = []
        self.padre = None # Para reconstruir el camino

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == TURQUESA

    def reset(self):
        self.color = BLANCO
        self.padre = None

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_fin(self):
        self.color = TURQUESA

    def hacer_visitado(self):
        self.color = ROJO

    def hacer_open(self):
        self.color = VERDE

    def hacer_camino(self):
        self.color = AZUL

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, ANCHO_NODO, ALTO_NODO))

    def actualizar_vecinos(self, grid):
        self.vecinos = []
        # (f, c) -> Abajo, Arriba, Derecha, Izquierda, y Diagonales
        direcciones = [
            (1, 0), (-1, 0), (0, 1), (0, -1), # Ortogonales
            (1, 1), (1, -1), (-1, 1), (-1, -1) # Diagonales
        ]

        for df, dc in direcciones:
            f, c = self.fila + df, self.col + dc

            if 0 <= f < FILAS and 0 <= c < COLUMNAS:
                vecino = grid[f][c]
                if not vecino.es_pared():
                    # Lógica para evitar atravesar esquinas de paredes (Wall Clipping)
                    if abs(df) == 1 and abs(dc) == 1: # Es diagonal
                        # Verifica bloqueos adyacentes ortogonales
                        if grid[self.fila + df][self.col].es_pared() or grid[self.fila][self.col + dc].es_pared():
                            continue 
                    
                    self.vecinos.append(vecino)

def crear_grid():
    grid = []
    for i in range(FILAS):
        grid.append([])
        for j in range(COLUMNAS):
            nodo = Nodo(i, j)
            grid[i].append(nodo)
    return grid

def dibujar_grid(ventana):
    for i in range(FILAS):
        pygame.draw.line(ventana, GRIS, (0, i * ALTO_NODO), (ANCHO_VENTANA, i * ALTO_NODO))
    for j in range(COLUMNAS):
        pygame.draw.line(ventana, GRIS, (j * ANCHO_NODO, 0), (j * ANCHO_NODO, ALTO_VENTANA))

def dibujar(ventana, grid):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)
    dibujar_grid(ventana)
    pygame.display.update()

def heuristica(p1, p2):
    # Distancia Euclidiana (Pitágoras) es la correcta para movimientos diagonales
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruir_camino(nodo_actual, draw_func):
    while nodo_actual.padre:
        nodo_actual = nodo_actual.padre
        if not nodo_actual.es_inicio(): 
            nodo_actual.hacer_camino()
        draw_func()

def a_estrella(draw_func, grid, inicio, fin):
    count = 0
    open_set = [] # Priority Queue
    heappush(open_set, (0, count, inicio))
    
    # g_score: costo exacto desde el inicio hasta este nodo
    g_score = {nodo: float("inf") for fila in grid for nodo in fila}
    g_score[inicio] = 0
    
    # f_score: g_score + heurística
    f_score = {nodo: float("inf") for fila in grid for nodo in fila}
    f_score[inicio] = heuristica(inicio.get_pos(), fin.get_pos())

    open_set_hash = {inicio} # Para búsqueda rápida O(1)

    while open_set:
        # Manejo de eventos básico para poder cerrar la ventana mientras calcula
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        # Obtener el nodo con el f_score más bajo
        actual = heappop(open_set)[2]
        open_set_hash.remove(actual)

        if actual == fin:
            reconstruir_camino(fin, draw_func)
            fin.hacer_fin() # Restaurar color fin
            return True

        for vecino in actual.vecinos:
            # Costo: 1 para ortogonal, 1.414 (raiz de 2) para diagonal
            es_diagonal = abs(vecino.fila - actual.fila) == 1 and abs(vecino.col - actual.col) == 1
            peso = 1.414 if es_diagonal else 1.0
            
            temp_g_score = g_score[actual] + peso

            if temp_g_score < g_score[vecino]:
                vecino.padre = actual
                g_score[vecino] = temp_g_score
                f_score[vecino] = temp_g_score + heuristica(vecino.get_pos(), fin.get_pos())
                
                if vecino not in open_set_hash:
                    count += 1
                    heappush(open_set, (f_score[vecino], count, vecino))
                    open_set_hash.add(vecino)
                    if not vecino.es_fin():
                        vecino.hacer_open()

        draw_func()

        if actual != inicio:
            actual.hacer_visitado()

    return False

def main():
    grid = crear_grid()
    inicio = None
    fin = None
    corriendo = True

    while corriendo:
        dibujar(VENTANA, grid)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            # Click Izquierdo: Poner Inicio, Fin o Pared
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                fila, col = pos[1] // ALTO_NODO, pos[0] // ANCHO_NODO
                if 0 <= fila < FILAS and 0 <= col < COLUMNAS:
                    nodo = grid[fila][col]
                    if not inicio and nodo != fin:
                        inicio = nodo
                        inicio.hacer_inicio()
                    elif not fin and nodo != inicio:
                        fin = nodo
                        fin.hacer_fin()
                    elif nodo != inicio and nodo != fin:
                        nodo.hacer_pared()

            # Click Derecho: Borrar
            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                fila, col = pos[1] // ALTO_NODO, pos[0] // ANCHO_NODO
                if 0 <= fila < FILAS and 0 <= col < COLUMNAS:
                    nodo = grid[fila][col]
                    nodo.reset()
                    if nodo == inicio:
                        inicio = None
                    elif nodo == fin:
                        fin = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and inicio and fin:
                    # Actualizar vecinos solo antes de correr
                    for fila in grid:
                        for nodo in fila:
                            nodo.actualizar_vecinos(grid)
                    
                    a_estrella(lambda: dibujar(VENTANA, grid), grid, inicio, fin)

                if event.key == pygame.K_c:
                    inicio = None
                    fin = None
                    grid = crear_grid()

    pygame.quit()

if __name__ == "__main__":
    main()