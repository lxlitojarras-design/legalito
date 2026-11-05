"""
Chatbot RAG local con Ollama (100% gratis y sin límites para ferias).
Interfaz : Streamlit
Lógica   : LangChain + Ollama
"""

import io
import os

import streamlit as st
from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# ----------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------------------------------------------------------
TAMANO_FRAGMENTO = 1000
SOLAPE_FRAGMENTO = 200
FRAGMENTOS_A_RECUPERAR = 2
MENSAJES_DE_MEMORIA = 4

MENSAJE_SIN_RESPUESTA = "No encontré esa información en los documentos cargados."

# Modelos locales en Ollama (asegúrate de tenerlos descargados en tu PC)
MODELO_CHAT_DEFAULT = "llama3"
MODELO_EMB_DEFAULT = "llama3"

PROMPT_RESPUESTA = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente virtual amable, educado y servicial para una feria.\n"
            "1. Si el usuario te saluda ('hola', 'buenos días') o hace preguntas casuales, responde con naturalidad.\n"
            "2. Si hay un CONTEXTO de documentos disponible, úsalo obligatoriamente para responder con precisión.\n"
            "3. Si te preguntan sobre los documentos y la respuesta NO está en el contexto, di exactamente: "
            f"'{MENSAJE_SIN_RESPUESTA}'\n"
            "4. Responde siempre en español.",
        ),
        (
            "human",
            "CONTEXTO DE DOCUMENTOS:\n{contexto}\n\nHISTORIAL:\n{historial}\n\nPREGUNTA: {pregunta}",
        ),
    ]
)


# ----------------------------------------------------------------------------
# LECTURA DE ARCHIVOS
# ----------------------------------------------------------------------------
def leer_pdf(nombre: str, datos: bytes) -> list[Document]:
    lector = PdfReader(io.BytesIO(datos))
    documentos = []
    for numero, pagina in enumerate(lector.pages, start=1):
        texto = (pagina.extract_text() or "").strip()
        if texto:
            documentos.append(
                Document(page_content=texto, metadata={"source": nombre, "page": numero})
            )
    return documentos


def leer_txt(nombre: str, datos: bytes) -> list[Document]:
    for codificacion in ("utf-8", "latin-1"):
        try:
            texto = datos.decode(codificacion).strip()
            break
        except UnicodeDecodeError:
            continue
    return [Document(page_content=texto, metadata={"source": nombre})] if texto else []


def leer_docx(nombre: str, datos: bytes) -> list[Document]:
    doc = DocxDocument(io.BytesIO(datos))
    partes = [p.text for p in doc.paragraphs if p.text.strip()]
    for tabla in doc.tables:
        for fila in tabla.rows:
            celdas = [c.text.strip() for c in fila.cells if c.text.strip()]
            if celdas:
                partes.append(" | ".join(celdas))
    texto = "\n".join(partes).strip()
    return [Document(page_content=texto, metadata={"source": nombre})] if texto else []


LECTORES = {".pdf": leer_pdf, ".txt": leer_txt, ".docx": leer_docx}


def leer_archivo(archivo) -> list[Document]:
    extension = os.path.splitext(archivo.name)[1].lower()
    return LECTORES[extension](archivo.name, archivo.getvalue())


# ----------------------------------------------------------------------------
# INTERFAZ
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Chatbot para Feria (Ollama)", page_icon="🤖", layout="wide")
st.title("🤖 Chatbot Local para Feria")
st.caption("Funciona sin internet y sin límites de peticiones usando Ollama.")

for clave, valor in {
    "mensajes": [],
    "indice": None,
    "llm": None,
    "resumen": None,
}.items():
    st.session_state.setdefault(clave, valor)

with st.sidebar:
    st.header("⚙️ Configuración Local")
    modelo_chat = st.text_input("Modelo Ollama (Chat)", MODELO_CHAT_DEFAULT)
    modelo_emb = st.text_input("Modelo Ollama (Embeddings)", MODELO_EMB_DEFAULT)

    st.divider()
    st.header("📁 Documentos")
    archivos = st.file_uploader(
        "Sube PDF, TXT o DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if st.button("🗑️ Limpiar chat", use_container_width=True):
        st.session_state.mensajes = []
        st.rerun()

    zona_estado = st.container()

# Inicializar modelos de Ollama de forma automática
if st.session_state.llm is None:
    try:
        st.session_state.llm = ChatOllama(model=modelo_chat, temperature=0.3)
    except Exception as e:
        st.error(f"Error conectando con Ollama: Asegúrate de que Ollama está abierto en tu PC. Detalles: {e}")

# Procesar archivos si se suben
if archivos and st.session_state.llm:
    with zona_estado:
        with st.status("Procesando documentos localmente…", expanded=True) as estado:
            try:
                embeddings = OllamaEmbeddings(model=modelo_emb)
                divisor = RecursiveCharacterTextSplitter(
                    chunk_size=TAMANO_FRAGMENTO, chunk_overlap=SOLAPE_FRAGMENTO
                )
                fragmentos = []
                for archivo in archivos:
                    docs = leer_archivo(archivo)
                    fragmentos.extend(divisor.split_documents(docs))

                if fragmentos:
                    indice = InMemoryVectorStore(embeddings)
                    indice.add_documents(fragmentos)
                    st.session_state.indice = indice
                    st.session_state.resumen = (len(archivos), len(fragmentos))
                    estado.update(label="¡Listo para la feria!", state="complete")
                else:
                    estado.update(label="No se extrajo texto", state="error")
            except Exception as error:
                estado.update(label="Error al procesar", state="error")
                st.error(str(error))

with zona_estado:
    if st.session_state.indice is not None:
        n_archivos, n_frag = st.session_state.resumen
        st.success(f"✅ {n_archivos} documentos listos · {n_frag} fragmentos.")
    else:
        st.info("💡 Ollama activo. Puedes saludar o subir documentos para empezar.")

# Chat
for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])

pregunta = st.chat_input("Escribe un saludo o pregunta sobre tus documentos…")

if pregunta:
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        try:
            historial_texto = "\n".join(
                f"{'Usuario' if m['rol']=='user' else 'Asistente'}: {m['contenido']}"
                for m in st.session_state.mensajes[-MENSAJES_DE_MEMORIA:]
            )

            fragmentos = []
            if st.session_state.indice is not None:
                try:
                    fragmentos = st.session_state.indice.similarity_search(pregunta, k=FRAGMENTOS_A_RECUPERAR)
                except Exception:
                    fragmentos = []

            contexto_texto = "\n\n---\n\n".join([f"[{f.metadata['source']}]\n{f.page_content}" for f in fragmentos]) if fragmentos else "(Sin documentos consultados)"

            cadena = PROMPT_RESPUESTA | st.session_state.llm | StrOutputParser()
            flujo = cadena.stream({
                "contexto": contexto_texto,
                "historial": historial_texto,
                "pregunta": pregunta
            })

            respuesta = st.write_stream(flujo)
            st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta})
        except Exception as error:
            st.error(f"Error generando respuesta: {error}")
            st.session_state.mensajes.pop()
