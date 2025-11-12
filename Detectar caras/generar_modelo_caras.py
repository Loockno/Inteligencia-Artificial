import cv2 as cv
import numpy as np
import os, random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count

# ==================== CONFIGURACIÓN DE RENDIMIENTO ====================
# Configura OpenCV para usar todos los cores
cv.setNumThreads(cpu_count())  # Usa todos los cores disponibles
print(f"🚀 OpenCV usando {cv.getNumThreads()} threads (de {cpu_count()} cores)")

# Configuración del dataset
dataSet = './Detectar caras/fotos_28x28'
target_size = (100, 100)
max_per_class = 3500
min_per_class = 20
seed = 42
random.seed(seed)

def is_valid(img):
    return img is not None and img.size > 0

def process_image(filepath):
    """Procesa una imagen individual (para paralelización)"""
    img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
    if not is_valid(img):
        return None
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.shape != target_size:
        img = cv.resize(img, target_size, interpolation=cv.INTER_AREA)
    return img

def load_person_images_parallel(person_path, files, max_workers=None):
    """Carga imágenes de una persona usando paralelización"""
    if max_workers is None:
        max_workers = min(cpu_count(), len(files))
    
    valid_imgs = []
    invalid_count = 0
    
    # Procesa imágenes en paralelo
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_image, files))
    
    for img in results:
        if img is not None:
            valid_imgs.append(img)
        else:
            invalid_count += 1
    
    return valid_imgs, invalid_count

# ==================== CARGA DE DATOS ====================
stats_shapes = defaultdict(Counter)
stats_types  = defaultdict(Counter)
invalid_by_person = defaultdict(int)

facesData, labels = [], []
label_map, next_label = {}, 0

print("📂 Cargando dataset...")
for person in sorted(os.listdir(dataSet)):
    p = os.path.join(dataSet, person)
    if not os.path.isdir(p):
        continue
    if person not in label_map:
        label_map[person] = next_label
        next_label += 1

    files = [os.path.join(p, f) for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
    random.shuffle(files)

    # ⚡ CARGA PARALELA DE IMÁGENES
    valid_imgs, invalid_count = load_person_images_parallel(p, files)
    invalid_by_person[person] = invalid_count

    # Estadísticas
    for img in valid_imgs:
        stats_shapes[person][img.shape] += 1
        stats_types[person][str(img.dtype)] += 1

    # Balanceo
    if len(valid_imgs) < min_per_class:
        print(f"[OMITIDA] {person}: {len(valid_imgs)} válidas (< {min_per_class}).")
        continue

    valid_imgs = valid_imgs[:max_per_class]

    for img in valid_imgs:
        facesData.append(img)
        labels.append(label_map[person])

    print(f"✓ {person}: usadas {len(valid_imgs)} | inválidas {invalid_count}")

# Reporte de homogeneidad
print("\n=== Homogeneidad ===")
for person in sorted(stats_shapes.keys()):
    print(f"{person}: shapes {dict(stats_shapes[person])} | dtypes {dict(stats_types[person])}")

labels = np.asarray(labels, dtype=np.int32)
print(f"\n📊 Total imágenes: {len(facesData)} | Clases: {len(set(labels.tolist()))}")

# ==================== ENTRENAMIENTO OPTIMIZADO ====================
print("\n🔥 Entrenando modelo FisherFaces...")
print("⏳ Este proceso puede tardar varios minutos...")

# FisherFaces usa internamente los threads de OpenCV configurados arriba
rec = cv.face.FisherFaceRecognizer_create()
rec.train(facesData, labels)
rec.write('FisherFace.xml')

print("✅ Modelo guardado: FisherFace.xml")
print(f"💾 Tamaño del modelo: {os.path.getsize('FisherFace.xml') / 1024:.2f} KB")