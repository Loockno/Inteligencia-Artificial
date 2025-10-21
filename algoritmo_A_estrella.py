import pygame
from heapq import heappush, heappop

pygame.init()

# Configuraciones iniciales
ANCHO_VENTANA = 800
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Visualización de Nodos (A*)")

# Colores (RGB)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
VERDE = (0, 255, 0)      # abierto (open set)
ROJO = (255, 0, 0)       # cerrado (closed set)
NARANJA = (255, 165, 0)  # inicio
PURPURA = (128, 0, 128)  # fin
CIAN = (64, 224, 208)    # camino reconstruido

class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = fila * ancho
        self.y = col * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas
        self.vecinos = []

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA

    def hacer_abierto(self):
        if not self.es_inicio() and not self.es_fin():
            self.color = VERDE

    def hacer_cerrado(self):
        if not self.es_inicio() and not self.es_fin():
            self.color = ROJO

    def hacer_camino(self):
        if not self.es_inicio() and not self.es_fin():
            self.color = CIAN

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.y, self.x, self.ancho, self.ancho))

    def actualizar_vecinos(self, grid):
        self.vecinos = []
        filas = self.total_filas
        # Abajo
        if self.fila + 1 < filas and not grid[self.fila + 1][self.col].es_pared():
            self.vecinos.append(grid[self.fila + 1][self.col])
        # Arriba
        if self.fila - 1 >= 0 and not grid[self.fila - 1][self.col].es_pared():
            self.vecinos.append(grid[self.fila - 1][self.col])
        # Derecha
        if self.col + 1 < filas and not grid[self.fila][self.col + 1].es_pared():
            self.vecinos.append(grid[self.fila][self.col + 1])
        # Izquierda
        if self.col - 1 >= 0 and not grid[self.fila][self.col - 1].es_pared():
            self.vecinos.append(grid[self.fila][self.col - 1])

def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid

def dibujar_grid(ventana, filas, ancho):
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, GRIS, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, GRIS, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))

def dibujar(ventana, grid, filas, ancho):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)
    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()

def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    x, y = pos  # x horizontal, y vertical (pantalla)
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col

def heuristica(a, b):
    # Distancia Manhattan
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruir_camino(came_from, actual, dibujar_func):
    while actual in came_from:
        actual = came_from[actual]
        if not actual.es_inicio():
            actual.hacer_camino()
        dibujar_func()

def a_estrella(dibujar_func, grid, inicio, fin):
    contador = 0
    open_heap = []
    heappush(open_heap, (0, contador, inicio))
    came_from = {}

    g_score = {nodo: float("inf") for fila in grid for nodo in fila}
    f_score = {nodo: float("inf") for fila in grid for nodo in fila}
    g_score[inicio] = 0
    f_score[inicio] = heuristica(inicio.get_pos(), fin.get_pos())

    open_set = {inicio}

    while open_heap:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        actual = heappop(open_heap)[2]
        open_set.discard(actual)

        if actual == fin:
            reconstruir_camino(came_from, fin, dibujar_func)
            fin.hacer_fin()
            inicio.hacer_inicio()
            return True

        for vecino in actual.vecinos:
            tentativo_g = g_score[actual] + 1
            if tentativo_g < g_score[vecino]:
                came_from[vecino] = actual
                g_score[vecino] = tentativo_g
                f_score[vecino] = tentativo_g + heuristica(vecino.get_pos(), fin.get_pos())
                if vecino not in open_set:
                    contador += 1
                    heappush(open_heap, (f_score[vecino], contador, vecino))
                    open_set.add(vecino)
                    vecino.hacer_abierto()

        if not actual.es_inicio():
            actual.hacer_cerrado()

        dibujar_func()

    return False

def limpiar_grid(grid):
    for fila in grid:
        for nodo in fila:
            if not nodo.es_pared():
                nodo.restablecer()

def main(ventana, ancho):
    FILAS = 40
    grid = crear_grid(FILAS, ancho)

    inicio = None
    fin = None
    corriendo = True

    while corriendo:
        for fila in grid:
            for nodo in fila:
                nodo.actualizar_vecinos(grid)

        dibujar(ventana, grid, FILAS, ancho)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if pygame.mouse.get_pressed()[0]:  # Click izquierdo
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                if 0 <= fila < FILAS and 0 <= col < FILAS:
                    nodo = grid[fila][col]
                    if not inicio and nodo != fin and not nodo.es_pared():
                        inicio = nodo
                        inicio.hacer_inicio()
                    elif not fin and nodo != inicio and not nodo.es_pared():
                        fin = nodo
                        fin.hacer_fin()
                    elif nodo != fin and nodo != inicio:
                        nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]:  # Click derecho
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                if 0 <= fila < FILAS and 0 <= col < FILAS:
                    nodo = grid[fila][col]
                    nodo.restablecer()
                    if nodo == inicio:
                        inicio = None
                    elif nodo == fin:
                        fin = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and inicio and fin:
                    for fila in grid:
                        for nodo in fila:
                            nodo.actualizar_vecinos(grid)
                    a_estrella(lambda: dibujar(ventana, grid, FILAS, ancho), grid, inicio, fin)

                if event.key == pygame.K_c:
                    inicio = None
                    fin = None
                    grid = crear_grid(FILAS, ancho)

    pygame.quit()

main(VENTANA, ANCHO_VENTANA)
