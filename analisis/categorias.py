import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
from itertools import chain

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  

def cargar_dataset(ruta=None):
    """
    Carga el CSV y estandariza un poco las columnas.
    Intenta varios encodings comunes hasta que alguno funcione.
    """
    if ruta is None:
        ruta = BASE_DIR / "datasetTexto.csv"

    posibles_encodings = ["utf-8", "latin-1", "cp1252"]

    ultimo_error = None
    for enc in posibles_encodings:
        try:
            print(f"Intentando leer con encoding = {enc} ...")
            df = pd.read_csv(
                ruta,
                encoding=enc,
                parse_dates=["Fecha"],  
                dayfirst=True
            )
            print(f"✅ Leído correctamente con encoding = {enc}")
            break
        except UnicodeDecodeError as e:
            print(f"❌ Falló con encoding = {enc}")
            ultimo_error = e
            df = None

    if df is None:
        raise ultimo_error

    df.columns = [c.strip() for c in df.columns]

    print("Columnas:", df.columns.tolist())
    print("\nPrimeras filas:")
    print(df.head())

    return df

def analisis_descriptivo(df):
    print("\n=== INFO GENERAL ===")
    print(df.info())

    print("\n=== VALORES NULOS ===")
    print(df.isna().sum())

    print("\n=== DUPLICADOS ===")
    print("Filas duplicadas:", df.duplicated().sum())

    if "Categoria" in df.columns:
        print("\n=== Notas por CATEGORÍA ===")
        print(df["Categoria"].value_counts())

    if "Medio" in df.columns:
        print("\n=== Notas por MEDIO ===")
        print(df["Medio"].value_counts())

    if "Fecha" in df.columns:
        print("\n=== Publicaciones por FECHA ===")
        print(df.groupby("Fecha")["ID"].count().sort_index())

    for col in ["Titulo", "Resumen", "Comentario_Reaccion"]:
        if col in df.columns:
            df[f"{col}_len_char"] = df[col].astype(str).str.len()
            df[f"{col}_len_words"] = df[col].astype(str).str.split().str.len()
            print(f"\n=== Estadísticos para {col} ===")
            print(df[[f"{col}_len_char", f"{col}_len_words"]].describe())

    return df

def graficas_basicas(df):
    plt.rcParams["figure.figsize"] = (10, 4)

    if "Fecha" in df.columns:
        pub_por_dia = df.groupby("Fecha")["ID"].count().sort_index()
        pub_por_dia.plot(kind="bar")
        plt.title("Número de notas por día")
        plt.xlabel("Fecha")
        plt.ylabel("Cantidad de notas")
        plt.tight_layout()
        plt.show()

    if "Categoria" in df.columns:
        df["Categoria"].value_counts().plot(kind="bar")
        plt.title("Notas por categoría")
        plt.xlabel("Categoría")
        plt.ylabel("Cantidad")
        plt.tight_layout()
        plt.show()

    if "Medio" in df.columns:
        df["Medio"].value_counts().head(10).plot(kind="bar")
        plt.title("Top 10 medios por número de notas")
        plt.xlabel("Medio")
        plt.ylabel("Cantidad")
        plt.tight_layout()
        plt.show()

import nltk
from nltk.corpus import stopwords

def preparar_texto(df):
    nltk.download("stopwords", quiet=True)
    stopwords_es = set(stopwords.words("spanish"))

    def limpiar_texto(texto):
        texto = str(texto).lower()
        texto = re.sub(r"http\S+|www\.\S+", "", texto)             
        texto = re.sub(r"[^a-záéíóúñü0-9# ]", " ", texto)          
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto

    def tokenizar(texto):
        palabras = texto.split()
        palabras = [p for p in palabras if p not in stopwords_es and len(p) > 2]
        return palabras

    for col in ["Titulo", "Resumen", "Comentario_Reaccion"]:
        if col in df.columns:
            clean_col = f"{col}_clean"
            tok_col = f"{col}_tokens"
            df[clean_col] = df[col].apply(limpiar_texto)
            df[tok_col] = df[clean_col].apply(tokenizar)

    return df

from sklearn.feature_extraction.text import CountVectorizer

def top_palabras(serie_tokens, n=30):
    todas = list(chain.from_iterable(serie_tokens))
    conteo = Counter(todas)
    return conteo.most_common(n)

def mostrar_top_palabras(df):
    for col in ["Titulo", "Resumen", "Comentario_Reaccion"]:
        tok_col = f"{col}_tokens"
        if tok_col in df.columns:
            print(f"\n=== Top palabras en {col.upper()} ===")
            print(top_palabras(df[tok_col], n=30))

def top_ngrams(textos, ngram_range=(2,2), top=20):
    vectorizer = CountVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        min_df=1
    )
    X = vectorizer.fit_transform(textos)
    suma = X.sum(axis=0)
    freqs = [
        (word, int(suma[0, idx]))
        for word, idx in vectorizer.vocabulary_.items()
    ]
    freqs = sorted(freqs, key=lambda x: x[1], reverse=True)
    return freqs[:top]

def mostrar_ngrams(df):
    if "Comentario_Reaccion_clean" not in df.columns:
        print("No existe Comentario_Reaccion_clean; ejecuta preparar_texto(df) primero.")
        return

    textos = df["Comentario_Reaccion_clean"].tolist()
    print("\n=== Top BIGRAMAS en comentarios ===")
    print(top_ngrams(textos, (2,2), top=20))

    print("\n=== Top TRIGRAMAS en comentarios ===")
    print(top_ngrams(textos, (3,3), top=20))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def topic_modeling(df, n_temas=4):
    """
    Crea temas usando Resumen + Comentario_Reaccion.
    """
    if "Resumen_clean" not in df.columns or "Comentario_Reaccion_clean" not in df.columns:
        print("Falta limpiar texto. Ejecuta preparar_texto(df) primero.")
        return df

    df["texto_combinado"] = df["Resumen_clean"] + " " + df["Comentario_Reaccion_clean"]

    vectorizer = TfidfVectorizer(
        max_df=0.95,
        min_df=1,
        max_features=1000
    )
    X = vectorizer.fit_transform(df["texto_combinado"])

    lda = LatentDirichletAllocation(
        n_components=n_temas,
        random_state=42,
        learning_method="batch"
    )
    lda.fit(X)

    feature_names = vectorizer.get_feature_names_out()

    def mostrar_temas(model, feature_names, n_top_words=10):
        for idx, topic in enumerate(model.components_):
            print(f"\n=== TEMA {idx} ===")
            top_indices = topic.argsort()[:-n_top_words - 1:-1]
            palabras = [feature_names[i] for i in top_indices]
            print(", ".join(palabras))

    mostrar_temas(lda, feature_names)

    topic_distribution = lda.transform(X)
    df["tema_dominante"] = topic_distribution.argmax(axis=1)

    print("\n=== Cantidad de notas por TEMA ===")
    print(df["tema_dominante"].value_counts())

    if "Categoria" in df.columns:
        print("\n=== TEMA x CATEGORÍA ===")
        print(pd.crosstab(df["tema_dominante"], df["Categoria"]))

    if "Medio" in df.columns:
        print("\n=== TEMA x MEDIO ===")
        print(pd.crosstab(df["tema_dominante"], df["Medio"]))

    return df

def guardar_resultados(df, ruta_salida="datasetTexto_enriquecido.csv"):
    columnas_salida = [c for c in df.columns]  # si quieres todas
    df.to_csv(ruta_salida, index=False, encoding="utf-8")
    print(f"\nArchivo guardado en: {ruta_salida}")

if __name__ == "__main__":
    df = cargar_dataset()
    df = analisis_descriptivo(df)
    graficas_basicas(df)
    df = preparar_texto(df)
    mostrar_top_palabras(df)
    mostrar_ngrams(df)
    df = topic_modeling(df, n_temas=4)
    guardar_resultados(df)
