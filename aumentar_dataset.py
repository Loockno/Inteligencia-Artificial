import pandas as pd
import ollama
import random
import os
import time
from datetime import datetime, timedelta

# ================= CONFIGURACIÓN =================
# Rotar entre modelos para mayor diversidad
MODELOS_DISPONIBLES = [
    "llama3.2:latest",  # Mejor calidad
    "mistral:7b",       # Excelente para creatividad
]

ARCHIVO_ENTRADA = 'dataset_proyecto3.csv'
ARCHIVO_SALIDA = 'dataset_aumentado_ollama.csv'

# Ajusta según tu capacidad de procesamiento
CANTIDAD_A_GENERAR = 4500
MAX_REINTENTOS = 3  # Intentos por tweet si falla validación

# Rangos realistas de engagement según sentimiento
ENGAGEMENT_RANGES = {
    'positivo': {'likes': (1500, 8000), 'reposts': (200, 1500)},
    'neutral': {'likes': (1000, 5000), 'reposts': (150, 800)},
    'negativo': {'likes': (800, 4000), 'reposts': (100, 600)}
}
# =================================================

print("🦙 Generador de Dataset Mejorado para Análisis Gen Z")
print(f"📊 Modelos disponibles: {', '.join(MODELOS_DISPONIBLES)}\n")

# ========== FUNCIONES AUXILIARES ==========

def cargar_dataset_origen(archivo):
    """Carga el dataset con múltiples encodings"""
    for encoding in ['latin1', 'utf-8', 'cp1252', 'iso-8859-1']:
        try:
            df = pd.read_csv(archivo, encoding=encoding)
            print(f"✅ Dataset cargado ({len(df)} registros) - Encoding: {encoding}")
            return df
        except:
            continue
    raise Exception("❌ No se pudo cargar el archivo con ningún encoding")

def validar_tweet(texto):
    """Valida que el tweet cumpla criterios de calidad"""
    if not texto or len(texto.strip()) < 20:
        return False, "Muy corto"
    
    if len(texto) > 280:
        return False, "Excede 280 caracteres"
    
    # Detectar respuestas del modelo que no son tweets
    frases_prohibidas = [
        "claro", "aquí tienes", "por supuesto", "tweet:", 
        "aquí está", "este es", "ejemplo:", "respuesta:"
    ]
    texto_lower = texto.lower()[:50]  # Solo revisar inicio
    for frase in frases_prohibidas:
        if frase in texto_lower:
            return False, f"Contiene '{frase}'"
    
    # Evitar tweets que son solo puntuación o emojis
    if len(texto.replace(" ", "").replace(".", "").replace(",", "")) < 15:
        return False, "Muy poca sustancia"
    
    return True, "OK"

def limpiar_tweet(texto):
    """Limpia el texto generado por el modelo"""
    texto = texto.strip()
    
    # Remover comillas al inicio/final
    if texto.startswith('"') and texto.endswith('"'):
        texto = texto[1:-1]
    if texto.startswith("'") and texto.endswith("'"):
        texto = texto[1:-1]
    
    # Remover prefijos comunes
    prefijos = ["Tweet: ", "tweet: ", "TWEET: ", "Respuesta: "]
    for prefijo in prefijos:
        if texto.startswith(prefijo):
            texto = texto[len(prefijo):]
    
    return texto.strip()

def generar_fecha_realista():
    """Genera fechas distribuidas a lo largo de 2024"""
    fecha_inicio = datetime(2024, 1, 1)
    dias_aleatorios = random.randint(0, 365)
    fecha = fecha_inicio + timedelta(days=dias_aleatorios)
    return fecha.strftime('%d/%m/%Y')

def generar_engagement(sentimiento, base_value, tipo='likes'):
    """Genera métricas de engagement realistas"""
    rango = ENGAGEMENT_RANGES.get(sentimiento, ENGAGEMENT_RANGES['neutral'])[tipo]
    
    # Usar base_value como referencia pero con variación
    if base_value > 0:
        factor = random.uniform(0.7, 1.3)
        valor = int(base_value * factor)
    else:
        valor = random.randint(rango[0], rango[1])
    
    # Asegurar que esté en rango realista
    return max(rango[0], min(rango[1], valor))

def generar_tweet_ollama(tema, sentimiento, ejemplo, modelo):
    """Genera un tweet usando Ollama con prompt mejorado"""
    
    # Mapeo de sentimientos a instrucciones específicas
    tono_map = {
        'positivo': 'optimista pero realista, con algo de ironía generacional',
        'negativo': 'crítico, desencantado, tal vez sarcástico o resignado',
        'neutral': 'reflexivo y observador, sin tomar partido claro'
    }
    
    # Ejemplos de estilo Gen Z
    estilos_genz = [
        "usa lenguaje casual, puedes tener algún typo ocasional",
        "escribe como si estuvieras pensando en voz alta",
        "puedes usar emojis si suman al mensaje (sin exagerar)",
        "sé auténtico, no corporativo ni forzado"
    ]
    
    prompt = f"""Eres un usuario anónimo de Twitter de la Generación Z (18-25 años).

Escribe UN SOLO tweet auténtico sobre: "{tema}"

Tono: {tono_map.get(sentimiento, 'neutral')}
Estilo: {random.choice(estilos_genz)}

Reglas:
- Entre 50-280 caracteres
- Lenguaje casual, directo
- Puede tener errores de tipeo ocasionales (pero legible)
- NO uses más de 2 emojis
- NO uses hashtags (o máximo 1)
- Suena humano y real

Inspiración (NO COPIES): "{ejemplo}"

IMPORTANTE: Responde SOLO con el texto del tweet. Sin introducciones ni explicaciones."""

    try:
        response = ollama.chat(
            model=modelo,
            messages=[{'role': 'user', 'content': prompt}],
            options={
                'temperature': 0.85,  # Balance creatividad/coherencia
                'top_p': 0.9,
                'top_k': 40,
                'num_predict': 100   # Limitar longitud de respuesta
            }
        )
        
        texto = response['message']['content']
        return limpiar_tweet(texto)
        
    except Exception as e:
        print(f"      ❌ Error con modelo {modelo}: {str(e)[:60]}")
        return None

def calcular_estadisticas(df):
    """Calcula estadísticas del dataset generado"""
    stats = {
        'total': len(df),
        'promedio_longitud': df['texto'].str.len().mean(),
        'con_emoji': df['texto'].str.contains('[😀-🙏]', regex=True).sum(),
        'por_sentimiento': df['sentimiento'].value_counts().to_dict(),
        'engagement_promedio': {
            'likes': df['likes'].mean(),
            'reposts': df['reposts'].mean()
        }
    }
    return stats

# ========== INICIALIZACIÓN ==========

df_origen = cargar_dataset_origen(ARCHIVO_ENTRADA)

# Gestionar IDs para continuar donde nos quedamos
if os.path.exists(ARCHIVO_SALIDA):
    try:
        df_existente = pd.read_csv(ARCHIVO_SALIDA, usecols=['id'])
        max_orig = df_origen['id'].max()
        max_exist = df_existente['id'].max()
        ultimo_id = max(max_orig, int(max_exist)) if not pd.isna(max_exist) else max_orig
        print(f"➡️  Continuando desde ID {ultimo_id + 1}\n")
    except:
        ultimo_id = df_origen['id'].max()
else:
    ultimo_id = df_origen['id'].max()
    print("🆕 Creando nuevo archivo de salida\n")

# ========== GENERACIÓN PRINCIPAL ==========

nuevos_registros = []
estadisticas = {
    'generados': 0,
    'fallidos': 0,
    'reintentos_totales': 0
}

print(f"🚀 Iniciando generación de {CANTIDAD_A_GENERAR} tweets...\n")
start_time = time.time()

for i in range(CANTIDAD_A_GENERAR):
    # Seleccionar ejemplo padre aleatorio
    fila_padre = df_origen.sample(1).iloc[0]
    tema = fila_padre['tema']
    sentimiento = fila_padre['sentimiento']
    ejemplo = fila_padre['texto']
    
    id_actual = ultimo_id + 1 + i
    
    # Rotar entre modelos para diversidad
    modelo_actual = random.choice(MODELOS_DISPONIBLES)
    
    print(f"[{i+1}/{CANTIDAD_A_GENERAR}] ID:{id_actual} | Modelo: {modelo_actual}")
    print(f"   📌 Tema: {tema[:50]}...")
    print(f"   💭 Sentimiento: {sentimiento}")
    
    # Intentar generar con validación
    tweet_valido = None
    for intento in range(MAX_REINTENTOS):
        tweet_generado = generar_tweet_ollama(tema, sentimiento, ejemplo, modelo_actual)
        
        if tweet_generado:
            es_valido, razon = validar_tweet(tweet_generado)
            
            if es_valido:
                tweet_valido = tweet_generado
                print(f"   ✅ Generado ({len(tweet_generado)} chars)")
                break
            else:
                print(f"   ⚠️  Intento {intento+1} rechazado: {razon}")
                estadisticas['reintentos_totales'] += 1
        else:
            print(f"   ⚠️  Intento {intento+1} falló en generación")
            estadisticas['reintentos_totales'] += 1
    
    if tweet_valido:
        # Crear registro completo
        nuevo_reg = {
            'id': id_actual,
            'fecha': generar_fecha_realista(),
            'texto': tweet_valido,
            'tema': tema,
            'sentimiento': sentimiento,
            'likes': generar_engagement(sentimiento, fila_padre['likes'], 'likes'),
            'reposts': generar_engagement(sentimiento, fila_padre['reposts'], 'reposts')
        }
        nuevos_registros.append(nuevo_reg)
        estadisticas['generados'] += 1
    else:
        print(f"   ❌ No se pudo generar después de {MAX_REINTENTOS} intentos")
        estadisticas['fallidos'] += 1
    
    print()  # Línea en blanco
    
    # Guardado parcial cada 5 registros exitosos
    if len(nuevos_registros) >= 5:
        df_temp = pd.DataFrame(nuevos_registros)
        escribir_header = not os.path.exists(ARCHIVO_SALIDA)
        df_temp.to_csv(ARCHIVO_SALIDA, mode='a', header=escribir_header, 
                       index=False, encoding='utf-8-sig')
        print(f"💾 Guardado parcial: {len(nuevos_registros)} registros\n")
        nuevos_registros = []

# Guardado final
if nuevos_registros:
    df_temp = pd.DataFrame(nuevos_registros)
    escribir_header = not os.path.exists(ARCHIVO_SALIDA)
    df_temp.to_csv(ARCHIVO_SALIDA, mode='a', header=escribir_header, 
                   index=False, encoding='utf-8-sig')
    print(f"💾 Guardado final: {len(nuevos_registros)} registros")

# ========== REPORTE FINAL ==========

tiempo_total = time.time() - start_time
tiempo_promedio = tiempo_total / CANTIDAD_A_GENERAR if CANTIDAD_A_GENERAR > 0 else 0

print("\n" + "="*60)
print("✨ GENERACIÓN COMPLETADA")
print("="*60)
print(f"⏱️  Tiempo total: {tiempo_total:.2f}s ({tiempo_promedio:.2f}s por tweet)")
print(f"✅ Generados exitosos: {estadisticas['generados']}")
print(f"❌ Fallidos: {estadisticas['fallidos']}")
print(f"🔄 Reintentos necesarios: {estadisticas['reintentos_totales']}")
print(f"📈 Tasa de éxito: {(estadisticas['generados']/CANTIDAD_A_GENERAR*100):.1f}%")

# Estadísticas del dataset completo
if os.path.exists(ARCHIVO_SALIDA):
    df_final = pd.read_csv(ARCHIVO_SALIDA)
    stats = calcular_estadisticas(df_final)
    
    print(f"\n📊 ESTADÍSTICAS DEL DATASET COMPLETO:")
    print(f"   Total de registros: {stats['total']}")
    print(f"   Longitud promedio: {stats['promedio_longitud']:.0f} caracteres")
    print(f"   Tweets con emoji: {stats['con_emoji']} ({stats['con_emoji']/stats['total']*100:.1f}%)")
    print(f"   \n   Distribución por sentimiento:")
    for sent, count in stats['por_sentimiento'].items():
        print(f"      • {sent}: {count} ({count/stats['total']*100:.1f}%)")
    print(f"   \n   Engagement promedio:")
    print(f"      • Likes: {stats['engagement_promedio']['likes']:.0f}")
    print(f"      • Reposts: {stats['engagement_promedio']['reposts']:.0f}")

print(f"\n📂 Archivo generado: {ARCHIVO_SALIDA}")
print("="*60)