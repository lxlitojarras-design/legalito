"""
Chatbot RAG local: conversa con tus propios documentos (PDF, TXT, DOCX).

Interfaz : Streamlit
Lógica   : LangChain (langchain-core + text-splitters)
Modelos  : OpenAI o Google Gemini (se elige en la barra lateral)

Cómo funciona (RAG = Retrieval-Augmented Generation):
  1. Subes archivos en la barra lateral.
  2. El texto se divide en fragmentos y se convierte en vectores (embeddings).
  3. Al preguntar, se buscan los fragmentos más parecidos a la pregunta.
  4. El modelo responde SOLO con esos fragmentos (nada de conocimiento externo).

Ejecutar:  streamlit run app.py
"""

import hashlib
import io
import os

import streamlit as st
from docx import Document as DocxDocument
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# Carga las variables del archivo .env (si existe) en el entorno.
load_dotenv()

# ----------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------------------------------------------------------
TAMANO_FRAGMENTO = 1000      # caracteres por fragmento
SOLAPE_FRAGMENTO = 200       # caracteres compartidos entre fragmentos vecinos
FRAGMENTOS_A_RECUPERAR = 4   # cuántos fragmentos se le dan al modelo por pregunta
LOTE_EMBEDDINGS = 50         # fragmentos por llamada (evita límites de la API)
MENSAJES_DE_MEMORIA = 6      # últimos mensajes que recuerda el chatbot

MENSAJE_SIN_RESPUESTA = "No encontré esa información en los documentos cargados."

# Modelos por defecto. Si alguno deja de existir, cámbialo desde la barra lateral.
PROVEEDORES = {
    "OpenAI": {
        "variable_env": "OPENAI_API_KEY",
        "chat": "gpt-4o-mini",
        "embeddings": "text-embedding-3-small",
    },
    "Google Gemini": {
        "variable_env": "GOOGLE_API_KEY",
        "chat": "gemini-2.5-flash",
        "embeddings": "gemini-embedding-001",
    },
}

PROMPT_CONDENSAR = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Reformula la última pregunta del usuario como una pregunta completa e "
            "independiente, usando el historial para resolver referencias como "
            "'eso', 'él' o 'el segundo punto'. Si ya es independiente, devuélvela "
            "igual. Devuelve únicamente la pregunta, sin explicaciones.",
        ),
        ("human", "Historial:\n{historial}\n\nÚltima pregunta: {pregunta}"),
    ]
)

PROMPT_RESPUESTA = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente que responde ÚNICAMENTE con la información del "
            "CONTEXTO proporcionado. Reglas:\n"
            "- No uses conocimiento externo ni inventes datos.\n"
            f"- Si la respuesta no está en el contexto, responde exactamente: "
            f"'{MENSAJE_SIN_RESPUESTA}'\n"
            "- Responde en el mismo idioma de la pregunta.\n"
            "- Cuando ayude, menciona entre corchetes el archivo de donde sale "
            "la información, por ejemplo [informe.pdf].",
        ),
        (
            "human",
            "CONTEXTO:\n{contexto}\n\nHISTORIAL DE LA CONVERSACIÓN:\n{historial}\n\n"
            "PREGUNTA: {pregunta}",
        ),
    ]
)


# ----------------------------------------------------------------------------
# LECTURA DE ARCHIVOS (todo en memoria, sin archivos temporales)
# ----------------------------------------------------------------------------
def leer_pdf(nombre: str, datos: bytes) -> list[Document]:
    """Devuelve un Document por página con texto."""
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
    """Lee texto plano probando varias codificaciones habituales."""
    for codificacion in ("utf-8", "latin-1"):
        try:
            texto = datos.decode(codificacion).strip()
            break
        except UnicodeDecodeError:
            continue
    return [Document(page_content=texto, metadata={"source": nombre})] if texto else []


def leer_docx(nombre: str, datos: bytes) -> list[Document]:
    """Extrae párrafos y tablas de un archivo Word (.docx)."""
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
    """Elige el lector adecuado según la extensión del archivo subido."""
    extension = os.path.splitext(archivo.name)[1].lower()
    return LECTORES[extension](archivo.name, archivo.getvalue())


# ----------------------------------------------------------------------------
# MODELOS E ÍNDICE VECTORIAL
# ----------------------------------------------------------------------------
def crear_modelos(proveedor: str, api_key: str, modelo_chat: str, modelo_emb: str):
    """Crea el modelo de chat y el de embeddings del proveedor elegido."""
    if proveedor == "OpenAI":
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        llm = ChatOpenAI(model=modelo_chat, temperature=0, api_key=api_key)
        embeddings = OpenAIEmbeddings(model=modelo_emb, api_key=api_key)
    else:
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
            GoogleGenerativeAIEmbeddings,
        )

        llm = ChatGoogleGenerativeAI(
            model=modelo_chat, temperature=0, google_api_key=api_key
        )
        embeddings = GoogleGenerativeAIEmbeddings(
            model=modelo_emb, google_api_key=api_key
        )
    return llm, embeddings


def construir_indice(archivos, embeddings, estado):
    """Lee, divide e indexa los archivos. Devuelve (índice, n_fragmentos, avisos)."""
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=TAMANO_FRAGMENTO, chunk_overlap=SOLAPE_FRAGMENTO
    )
    fragmentos: list[Document] = []
    avisos: list[str] = []

    for archivo in archivos:
        estado.update(label=f"Leyendo «{archivo.name}»…")
        try:
            documentos = leer_archivo(archivo)
        except Exception as error:  # PDF cifrado, archivo dañado, etc.
            avisos.append(f"No se pudo leer «{archivo.name}»: {error}")
            continue
        if not documentos:
            avisos.append(
                f"«{archivo.name}» no tiene texto extraíble "
                "(¿es un PDF escaneado como imagen?)."
            )
            continue
        fragmentos.extend(divisor.split_documents(documentos))

    if not fragmentos:
        return None, 0, avisos

    indice = InMemoryVectorStore(embeddings)
    barra = st.progress(0.0)
    for inicio in range(0, len(fragmentos), LOTE_EMBEDDINGS):
        lote = fragmentos[inicio : inicio + LOTE_EMBEDDINGS]
        hasta = min(inicio + LOTE_EMBEDDINGS, len(fragmentos))
        estado.update(label=f"Indexando fragmentos {hasta}/{len(fragmentos)}…")
        indice.add_documents(lote)
        barra.progress(hasta / len(fragmentos))
    barra.empty()
    return indice, len(fragmentos), avisos


# ----------------------------------------------------------------------------
# LÓGICA DE CONVERSACIÓN
# ----------------------------------------------------------------------------
def formatear_historial(mensajes: list[dict]) -> str:
    recientes = mensajes[-MENSAJES_DE_MEMORIA:]
    if not recientes:
        return "(sin historial)"
    return "\n".join(
        f"{'Usuario' if m['rol'] == 'user' else 'Asistente'}: {m['contenido']}"
        for m in recientes
    )


def formatear_contexto(fragmentos: list[Document]) -> str:
    bloques = []
    for f in fragmentos:
        pagina = f.metadata.get("page")
        origen = f.metadata["source"] + (f", pág. {pagina}" if pagina else "")
        bloques.append(f"[{origen}]\n{f.page_content}")
    return "\n\n---\n\n".join(bloques)


def preparar_respuesta(pregunta: str, historial: list[dict]):
    """Busca fragmentos relevantes y devuelve (flujo_de_texto, fragmentos)."""
    llm = st.session_state.llm
    texto_historial = formatear_historial(historial)

    # Memoria: si hay historial, la pregunta se reformula para que se entienda
    # sola (ej. "¿y el segundo?" -> "¿Cuál es el segundo requisito del contrato?").
    pregunta_busqueda = pregunta
    if historial:
        pregunta_busqueda = (
            PROMPT_CONDENSAR | llm | StrOutputParser()
        ).invoke({"historial": texto_historial, "pregunta": pregunta}).strip()

    fragmentos = st.session_state.indice.similarity_search(
        pregunta_busqueda, k=FRAGMENTOS_A_RECUPERAR
    )
    cadena = PROMPT_RESPUESTA | llm | StrOutputParser()
    flujo = cadena.stream(
        {
            "contexto": formatear_contexto(fragmentos),
            "historial": texto_historial,
            "pregunta": pregunta,
        }
    )
    return flujo, fragmentos


def mostrar_fuentes(fragmentos: list[Document]):
    with st.expander("📚 Fuentes consultadas"):
        for f in fragmentos:
            pagina = f.metadata.get("page")
            etiqueta = f.metadata["source"] + (f" — pág. {pagina}" if pagina else "")
            st.markdown(f"**{etiqueta}**")
            vista = f.page_content[:350]
            st.caption(vista + ("…" if len(f.page_content) > 350 else ""))


# ----------------------------------------------------------------------------
# INTERFAZ
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Chatbot con tus documentos", page_icon="📄", layout="wide")
st.title("📄 Chatbot con tus documentos")
st.caption("Responde únicamente con la información de los archivos que subas.")

# Estado persistente entre interacciones de Streamlit.
for clave, valor in {
    "mensajes": [],
    "indice": None,
    "llm": None,
    "config_indice": None,
    "firma_intentada": None,
    "resumen": None,
}.items():
    st.session_state.setdefault(clave, valor)

# ---- Barra lateral ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    proveedor = st.selectbox("Proveedor de IA", list(PROVEEDORES))
    datos_proveedor = PROVEEDORES[proveedor]

    # >>> AQUÍ VA TU API KEY <<<
    # Opción A (recomendada): ponla en el archivo .env (ver .env.example).
    # Opción B: pégala en este campo; solo vive en esta sesión del navegador.
    api_key = st.text_input(
        f"API Key de {proveedor}",
        type="password",
        placeholder=f"o define {datos_proveedor['variable_env']} en .env",
    ) or os.getenv(datos_proveedor["variable_env"], "")

    with st.expander("Modelos (avanzado)"):
        modelo_chat = st.text_input("Modelo de chat", datos_proveedor["chat"])
        modelo_emb = st.text_input("Modelo de embeddings", datos_proveedor["embeddings"])

    st.divider()
    st.header("📁 Documentos")
    archivos = st.file_uploader(
        "Sube PDF, TXT o DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)
    if col1.button("🔄 Reprocesar", use_container_width=True):
        st.session_state.firma_intentada = None
    if col2.button("🗑️ Limpiar chat", use_container_width=True):
        st.session_state.mensajes = []
        st.rerun()

    zona_estado = st.container()

# ---- Procesamiento automático al subir/cambiar archivos --------------------
if archivos and api_key:
    # La "firma" cambia si cambian los archivos, el proveedor o los modelos.
    firma = hashlib.md5(
        "|".join(
            [proveedor, modelo_chat, modelo_emb]
            + [f"{a.name}:{a.size}" for a in archivos]
        ).encode()
    ).hexdigest()

    if firma != st.session_state.firma_intentada:
        st.session_state.firma_intentada = firma
        st.session_state.indice = None
        st.session_state.resumen = None
        with zona_estado:
            with st.status("Procesando documentos…", expanded=True) as estado:
                try:
                    llm, embeddings = crear_modelos(
                        proveedor, api_key, modelo_chat, modelo_emb
                    )
                    indice, total, avisos = construir_indice(archivos, embeddings, estado)
                    for aviso in avisos:
                        st.warning(aviso)
                    if indice is None:
                        estado.update(label="No se pudo extraer texto", state="error")
                    else:
                        st.session_state.llm = llm
                        st.session_state.indice = indice
                        st.session_state.resumen = (len(archivos), total)
                        estado.update(label="Procesamiento terminado", state="complete")
                except Exception as error:
                    estado.update(label="Error al procesar", state="error")
                    st.error(f"{type(error).__name__}: {error}")

# ---- Indicador de estado (siempre visible en la barra lateral) -------------
with zona_estado:
    if st.session_state.indice is not None:
        n_archivos, n_fragmentos = st.session_state.resumen
        st.success(
            f"✅ {n_archivos} documento(s) procesado(s) · {n_fragmentos} fragmentos. "
            "¡Listo para consultar!"
        )
    elif not api_key:
        st.info("🔑 Falta la API Key (campo de arriba o archivo .env).")
    elif not archivos:
        st.info("📥 Sube al menos un documento para empezar.")

# ---- Chat ------------------------------------------------------------------
for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])
        if mensaje.get("fuentes"):
            mostrar_fuentes(mensaje["fuentes"])

listo = st.session_state.indice is not None
pregunta = st.chat_input(
    "Haz una pregunta sobre tus documentos…"
    if listo
    else "Sube documentos y configura tu API Key para chatear",
    disabled=not listo,
)

if pregunta:
    historial_previo = list(st.session_state.mensajes)  # antes de añadir la pregunta
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        try:
            flujo, fragmentos = preparar_respuesta(pregunta, historial_previo)
            respuesta = st.write_stream(flujo)
            mostrar_fuentes(fragmentos)
            st.session_state.mensajes.append(
                {"rol": "assistant", "contenido": respuesta, "fuentes": fragmentos}
            )
        except Exception as error:
            st.error(f"No se pudo generar la respuesta ({type(error).__name__}): {error}")
            st.session_state.mensajes.pop()  # quita la pregunta que falló
