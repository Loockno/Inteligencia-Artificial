import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import os

# --- CONFIGURACIÓN ---
NOMBRE_ARCHIVO = "corpus_combinado_completo.txt"
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300 

def limpiar_texto_sucio(texto):
    """
    Limpia la basura de codificación (mojibake) y símbolos raros.
    """
    if not isinstance(texto, str):
        return ""
        
    # 1. Eliminar patrones específicos de error de codificación comunes
    texto = texto.replace('ï¿½', 'ó') # Intento de arreglo común
    texto = texto.replace('Ã³', 'ó')
    texto = texto.replace('Ã', 'í')
    
    # 2. Eliminar cualquier caracter que NO sea letra, espacio o puntuación básica
    # Esto elimina los 1/2, los simbolos raros, etc.
    texto = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s\.,]', '', texto)
    
    # 3. Eliminar números sueltos (ensucian la nube de palabras)
    texto = re.sub(r'\b\d+\b', '', texto)
    
    # 4. Quitar espacios extra
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto

def parsear_corpus(filepath):
    data = {
        'tema': [],
        'sentimiento': [],
        'likes': [],
        'texto': []
    }
    
    if not os.path.exists(filepath):
        print(f"❌ Error: No encuentro '{filepath}'")
        return pd.DataFrame()

    print(f"📂 Leyendo '{filepath}' y limpiando errores de codificación...")
    
    # Intentamos leer con 'utf-8' pero ignorando errores para que no falle
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        contenido = f.read()

    # Extracción con Regex
    temas = re.findall(r"sobre el tema '(.*?)'", contenido)
    sentimientos = re.findall(r"calificado como '(.*?)'", contenido)
    likes_raw = re.findall(r"IMPACTO SOCIAL: (\d+) likes", contenido)
    likes = [int(x) for x in likes_raw]
    textos = re.findall(r'TESTIMONIO: "(.*?)"', contenido)

    min_len = min(len(temas), len(sentimientos), len(likes), len(textos))
    
    # Aplicamos la limpieza INMEDIATAMENTE al crear el DataFrame
    data['tema'] = [limpiar_texto_sucio(t) for t in temas[:min_len]]
    data['sentimiento'] = sentimientos[:min_len] # Los sentimientos suelen estar bien
    data['likes'] = likes[:min_len]
    data['texto'] = [limpiar_texto_sucio(t) for t in textos[:min_len]]
    
    return pd.DataFrame(data)

# --- GRÁFICA 1: SENTIMIENTOS ---
def plot_sentimientos(df):
    plt.figure(figsize=(8, 6))
    df['sentimiento'] = df['sentimiento'].str.capitalize()
    conteo = df['sentimiento'].value_counts()
    
    colores_map = {"Positivo": "#66b3ff", "Neutral": "#999999", "Negativo": "#ff6666"}
    colores = [colores_map.get(x, '#cccccc') for x in conteo.index]
    
    plt.pie(conteo, labels=conteo.index, autopct='%1.1f%%', startangle=140, colors=colores)
    plt.title('Distribución Emocional (Proyecto 3)', fontsize=14)
    plt.tight_layout()
    plt.savefig("Grafica_1_Sentimientos_CLEAN.png")
    print("✅ Gráfica 1 guardada.")

# --- GRÁFICA 2: VIRALIDAD (Corregida) ---
def plot_viralidad(df):
    plt.figure(figsize=(12, 6))
    
    # Acortar temas largos para que no ocupen todo el gráfico
    df['tema_short'] = df['tema'].apply(lambda x: x[:40] + '...' if len(x) > 40 else x)
    
    # Agrupar y ordenar
    df_agrupado = df.groupby('tema_short')['likes'].mean().sort_values(ascending=False).head(8)
    
    sns.barplot(x=df_agrupado.values, y=df_agrupado.index, palette="viridis", hue=df_agrupado.index, legend=False)
    
    plt.title('Temas con Mayor Impacto Social (Promedio de Likes)', fontsize=14)
    plt.xlabel('Promedio de Likes')
    
    plt.tight_layout()
    plt.savefig("Grafica_2_Viralidad_CLEAN.png")
    print("✅ Gráfica 2 guardada (Textos limpios).")

# --- GRÁFICA 3: NUBE DE PALABRAS (Corregida) ---
def plot_nube_palabras(df):
    plt.figure(figsize=(10, 6))
    
    texto_completo = " ".join(df['texto'].astype(str).tolist()).lower()
    
    # Stopwords extendidas para quitar basura residual
    stopwords_es = set(['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí', 'porque', 'esta', 'son', 'mí', 'yo', 'me', 'testimonio', 'contexto', 'nan', 'null'])
    
    # Generar nube con filtro de regex interno para solo aceptar letras
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white', 
                          stopwords=stopwords_es,
                          colormap='magma',
                          regexp=r"[a-záéíóúñ]+", # SOLO letras, ignora símbolos raros
                          min_font_size=10).generate(texto_completo)
    
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Mapa Semántico: Términos Recurrentes", fontsize=16)
    
    plt.tight_layout()
    plt.savefig("Grafica_3_NubePalabras_CLEAN.png")
    print("✅ Gráfica 3 guardada (Sin símbolos raros).")

if __name__ == "__main__":
    df = parsear_corpus(NOMBRE_ARCHIVO)
    if not df.empty:
        plot_sentimientos(df)
        plot_viralidad(df)
        plot_nube_palabras(df)
        print("\n✨ ¡Listo! Revisa las nuevas imágenes terminadas en '_CLEAN.png'")