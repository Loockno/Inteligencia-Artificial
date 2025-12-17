import pandas as pd
import os

# --- CONFIGURACIÓN ---
# Lista de archivos a procesar en orden
ARCHIVOS_INPUT = ["dataset_proyecto3.csv", "dataset_aumentado_ollama.csv"]
OUTPUT_CORPUS = "corpus_combinado_completo.txt" 

def leer_csv_robusto(filepath):
    """Intenta leer el CSV con diferentes codificaciones hasta que funcione."""
    codificaciones = ['latin-1', 'utf-8', 'cp1252']
    
    for cod in codificaciones:
        try:
            # engine='python' y on_bad_lines='skip' son vitales para tus archivos
            df = pd.read_csv(filepath, encoding=cod, on_bad_lines='skip', engine='python')
            print(f"   ✅ Leído exitosamente usando codificación: {cod}")
            return df
        except Exception:
            continue # Si falla, intenta la siguiente codificación
            
    print(f"   ❌ Error fatal: No se pudo leer {filepath} con ninguna codificación conocida.")
    return None

def generar_corpus():
    total_registros_global = 0
    
    print(f"🚀 Iniciando fusión de corpus...")
    
    # Abrimos el archivo de salida en modo escritura ('w') una sola vez
    with open(OUTPUT_CORPUS, "w", encoding="utf-8") as f:
        f.write("DOCUMENTO MAESTRO COMBINADO: PROYECTO 3 - GEN Z\n")
        f.write(f"Fuentes de datos: {', '.join(ARCHIVOS_INPUT)}\n")
        f.write("Objetivo: Análisis filosófico de crisis de sentido y tecnología.\n")
        f.write("="*60 + "\n\n")

        # Iteramos sobre cada archivo de la lista
        for archivo in ARCHIVOS_INPUT:
            print(f"\n📂 Procesando archivo: {archivo}...")
            
            if not os.path.exists(archivo):
                print(f"   ⚠️ El archivo {archivo} no existe. Saltando...")
                continue

            # Usamos la función inteligente para leer
            df = leer_csv_robusto(archivo)
            
            if df is None:
                continue

            print(f"   📝 Convirtiendo {len(df)} registros a narrativa...")

            for index, row in df.iterrows():
                # Limpieza y obtención segura de datos
                fecha = row.get('fecha', 'Fecha desconocida')
                tema = row.get('tema', 'General')
                sentimiento = row.get('sentimiento', 'Neutro')
                texto_raw = str(row.get('texto', ''))
                texto = texto_raw.strip()
                
                try:
                    likes = int(row.get('likes', 0))
                except:
                    likes = 0
                try:
                    reposts = int(row.get('reposts', 0))
                except:
                    reposts = 0
                
                # Agregamos una etiqueta de origen para que la IA sepa de qué archivo vino
                narrativa = (
                    f"--- REGISTRO (Fuente: {archivo}) ---\n"
                    f"CONTEXTO: En la fecha {fecha}, se registró una interacción sobre el tema '{tema}'.\n"
                    f"ESTADO EMOCIONAL: El usuario refleja un sentimiento calificado como '{sentimiento}'.\n"
                    f"TESTIMONIO: \"{texto}\"\n"
                    f"IMPACTO SOCIAL: {likes} likes, {reposts} compartidos.\n"
                    f"\n" 
                )
                
                f.write(narrativa)
            
            total_registros_global += len(df)

    print(f"\n✅ ¡FUSIÓN COMPLETADA! Archivo generado: {OUTPUT_CORPUS}")
    print(f"📊 Total de registros procesados: {total_registros_global}")
    print("👉 Sube este único archivo .txt a AnythingLLM.")

if __name__ == "__main__":
    generar_corpus()