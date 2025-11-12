import cv2
import os
from pathlib import Path

def redimensionar_imagenes_carpetas(carpeta_entrada, carpeta_salida, tamaño=(28, 28)):
    """
    Redimensiona imágenes manteniendo la estructura de subcarpetas.
    
    Args:
        carpeta_entrada: Carpeta raíz con subcarpetas de imágenes
        carpeta_salida: Carpeta donde se guardará la misma estructura
        tamaño: Tupla (ancho, alto). Default: (28, 28)
    """
    
    # Extensiones soportadas
    extensiones = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    total_procesadas = 0
    total_errores = 0
    
    print(f"🔄 Redimensionando imágenes a {tamaño[0]}x{tamaño[1]}...")
    print("=" * 60)
    
    # Recorrer todas las subcarpetas
    for root, dirs, files in os.walk(carpeta_entrada):
        # Calcular ruta relativa
        ruta_relativa = os.path.relpath(root, carpeta_entrada)
        ruta_destino = os.path.join(carpeta_salida, ruta_relativa)
        
        # Crear carpeta de destino si no existe
        Path(ruta_destino).mkdir(parents=True, exist_ok=True)
        
        # Filtrar solo imágenes
        imagenes = [f for f in files if f.lower().endswith(extensiones)]
        
        if imagenes:
            carpeta_actual = os.path.basename(root)
            print(f"\n📁 Procesando carpeta: {carpeta_actual}")
            print(f"   Imágenes encontradas: {len(imagenes)}")
            
            for idx, archivo in enumerate(imagenes, 1):
                ruta_entrada = os.path.join(root, archivo)
                ruta_salida_img = os.path.join(ruta_destino, archivo)
                
                try:
                    # Leer imagen
                    img = cv2.imread(ruta_entrada)
                    
                    if img is None:
                        print(f"   ❌ Error al leer: {archivo}")
                        total_errores += 1
                        continue
                    
                    # Redimensionar
                    img_redimensionada = cv2.resize(img, tamaño, 
                                                   interpolation=cv2.INTER_AREA)
                    
                    # Guardar
                    cv2.imwrite(ruta_salida_img, img_redimensionada)
                    total_procesadas += 1
                    
                    # Mostrar progreso
                    if idx % 10 == 0 or idx == len(imagenes):
                        print(f"   ✓ {idx}/{len(imagenes)} completadas")
                        
                except Exception as e:
                    print(f"   ❌ Error en {archivo}: {str(e)}")
                    total_errores += 1
    
    print("\n" + "=" * 60)
    print(f"✅ Proceso finalizado!")
    print(f"   Total procesadas: {total_procesadas}")
    print(f"   Errores: {total_errores}")
    print(f"   Carpeta destino: {carpeta_salida}")


if __name__ == "__main__":
    # Configuración para tu estructura
    CARPETA_ENTRADA = "./Emociones/Emociones"
    CARPETA_SALIDA = "Emociones2"
    TAMAÑO = (28, 28)
    
    # Ejecutar
    redimensionar_imagenes_carpetas(CARPETA_ENTRADA, CARPETA_SALIDA, TAMAÑO)