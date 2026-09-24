import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# Forzar el layout centrado y configurar el título de la pestaña
st.set_page_config(layout="centered", page_title="RAG App")

# Banner personalizado con HTML y CSS inyectado
st.markdown("""
    <style>
    .main-banner {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        padding: 40px 20px;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-banner h1 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-banner p {
        margin-top: 10px;
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
    }
    </style>
    <div class="main-banner">
        <h1>Generación Aumentada por Recuperación (RAG) 💬</h1>
        <p>Este agente te ayudará a realizar análisis sobre el PDF cargado</p>
    </div>
""", unsafe_allow_html=True)

st.caption(f"Versión de Python: {platform.python_version()}")

# Entrada de API Key centrada en el flujo principal
ke = st.text_input('Ingresa tu Clave de OpenAI', type="password", placeholder="sk-...")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# Uploader de PDF en la columna principal
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

# Lógica de procesamiento
if pdf is not None and ke:
    try:
        # Extraer texto del PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.info(f"Texto extraído: {len(text)} caracteres")
        
        # Dividir texto en fragmentos
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"Documento dividido en {len(chunks)} fragmentos")
        
        # Crear embeddings y base de conocimiento
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # Interfaz de pregunta centrada
        st.markdown("### ¿Qué necesitas saber sobre el documento?")
        user_question = st.text_area(" ", placeholder="Escribe tu pregunta aquí...", label_visibility="collapsed")
        
        # Procesar respuesta
        if user_question:
            with st.spinner("Analizando el documento..."):
                docs = knowledge_base.similarity_search(user_question)
                llm = OpenAI(temperature=0, model_name="gpt-4o-mini-2024-07-18")
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=user_question)
                
                # Mostrar respuesta en una tarjeta limpia
                st.markdown("---")
                st.markdown("#### Respuesta:")
                st.markdown(response)
                
    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("Por favor carga un archivo PDF para comenzar")