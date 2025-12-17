import os
import glob
import feedparser
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# --- CONFIGURACIÓN ---
CSV_PATH = "./dataset_proyecto3.csv" 
RSS_FOLDER = "./rss_datos" 
MODEL_NAME = "mistral:7b"
OUTPUT_FILE = "reporte_final_generado.md" # Aquí se guardarán las respuestas

print(f"--- Sistema RAG Automatizado (Proyecto 3) con {MODEL_NAME} ---")

# --- LISTA COMPLETA DE PREGUNTAS DEL PROYECTO ---
PREGUNTAS_PROYECTO = [
    "¿Qué expresiones o términos utiliza la Gen Z para describir el vacío existencial en redes sociales?",
    "¿Cómo influyen los algoritmos de recomendación en la construcción de su identidad?",
    "¿Qué emociones aparecen con mayor frecuencia cuando se habla de burnout o presión digital?",
    "¿La Gen Z percibe la autonomía como algo propio o como algo condicionado por la tecnología?",
    "¿Qué diferencias hay entre discursos auténticos vs discursos performativos en plataformas como TikTok?",
    "¿Existen patrones de lenguaje que indiquen crisis de sentido o desorientación vital?",
    "¿Cómo se refleja la idea de 'identidad líquida' en los datos recuperados?",
    "¿Qué menciones aparecen sobre libertad, control o manipulación algorítmica?",
    "¿Se observan señales de que los algoritmos crean deseos o hábitos?",
    "¿Qué temas o preocupaciones predominan en la conversación digital sobre propósito de vida?",
    "¿Hay evidencia de rechazo a los metarrelatos o valores tradicionales?",
    "¿Cómo aparece la figura del 'yo digital' en los textos analizados?",
    "¿Qué ejemplos concretos muestran pérdida del pensamiento crítico por efecto de la burbuja de filtros?",
    "¿Existen contrastes entre la visión que la Gen Z tiene de sí misma y lo que los datos sugieren?",
    "¿Qué rol juega la hiperconectividad en la ansiedad o depresión mencionada?",
    "¿Se observan patrones que apoyen las ideas de Byung-Chul Han sobre rendimiento y autoexplotación?",
    "¿Cómo interpretaría Foucault el régimen de vigilancia algorítmica detectado?",
    "¿Qué evidencias hay de que la tecnología 'desoculta' y transforma la vida según Heidegger?",
    "¿El espacio público digital está debilitado como afirma Habermas? ¿Qué muestran los datos?",
    "¿Cuáles son los principales miedos, frustraciones y esperanzas de la Gen Z frente al futuro?"
]

# --- 1. CARGA DE DATOS (HÍBRIDA) ---
def load_data():
    docs = []
    
    # A) CSV
    print("1️⃣  Cargando Dataset CSV...")
    if os.path.exists(CSV_PATH):
        loader = CSVLoader(file_path=CSV_PATH, encoding="utf-8")
        csv_docs = loader.load()
        for doc in csv_docs:
            doc.page_content = f"[TESTIMONIO ESTUDIANTE] {doc.page_content}"
        docs.extend(csv_docs)
        print(f"   ✅ {len(csv_docs)} registros sintéticos.")
    else:
        print(f"   ❌ No encontré {CSV_PATH}")

    # B) RSS
    print("2️⃣  Cargando Noticias RSS...")
    xml_files = glob.glob(os.path.join(RSS_FOLDER, "*.xml"))
    if xml_files:
        for file in xml_files:
            try:
                feed = feedparser.parse(file)
                for entry in feed.entries:
                    text = f"[NOTICIA REAL - Fuente: El País] {entry.title}. {entry.description}"
                    docs.append(Document(page_content=text, metadata={"source": "RSS"}))
                print(f"   ✅ Leído: {os.path.basename(file)}")
            except:
                pass
    else:
        print("   ⚠️ No hay archivos XML en rss_datos.")
    
    return docs

docs_totales = load_data()

# --- 2. VECTOR STORE ---
print("\n🧠 Generando Embeddings (esto solo tarda la primera vez)...")
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs_totales, embedding_function, persist_directory="./chroma_db_final")

# --- 3. CONFIGURACIÓN RAG ---
llm = ChatOllama(model=MODEL_NAME, temperature=0.3)

template = """
Eres un filósofo experto analizando datos para el Proyecto: 'La Generación Z y la Crisis de Sentido'.
Usa los siguientes datos recuperados para responder.

CONTEXTO:
{context}

PREGUNTA: 
{question}

INSTRUCCIONES:
1. Responde basándote estrictamente en el contexto (Testimonios y Noticias).
2. Cita autores cuando corresponda:
   - Cansancio/Rendimiento -> Byung-Chul Han.
   - Liquidez/Cambio -> Bauman.
   - Vigilancia/Poder -> Foucault.
   - Tecnología/Ser -> Heidegger.
   - Espacio Público -> Habermas.
3. Sé directo y académico.

RESPUESTA:
"""
prompt = ChatPromptTemplate.from_template(template)
retriever = vectorstore.as_retriever(search_kwargs={"k": 7}) # Top 7 fragmentos más relevantes

rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 4. EJECUCIÓN Y GUARDADO ---
print(f"\n🚀 Iniciando análisis de las {len(PREGUNTAS_PROYECTO)} preguntas...")
print(f"📝 Los resultados se guardarán en: {OUTPUT_FILE}\n")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("# REPORTE DE ANÁLISIS AUTOMATIZADO CON RAG\n")
    f.write("## Proyecto 3: La Generación Z y la Crisis de Sentido\n\n")

    for i, pregunta in enumerate(PREGUNTAS_PROYECTO, 1):
        print(f"⏳ ({i}/{len(PREGUNTAS_PROYECTO)}) Analizando: {pregunta[:40]}...")
        
        try:
            respuesta = rag_chain.invoke(pregunta)
            
            # Escribir en el archivo
            f.write(f"### Pregunta {i}: {pregunta}\n\n")
            f.write(f"**Análisis del Modelo:**\n\n{respuesta}\n\n")
            f.write("---\n\n")
            
            # Forzar guardado en disco por si se cancela el script
            f.flush() 
            
        except Exception as e:
            print(f"❌ Error en pregunta {i}: {e}")
            f.write(f"### Pregunta {i}: {pregunta}\n\nERROR: {e}\n\n---\n")

print(f"\n✅ ¡LISTO! Abre el archivo '{OUTPUT_FILE}' para ver tu reporte completo.")