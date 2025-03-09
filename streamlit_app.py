import streamlit as st
from supabase import create_client
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
import os
import traceback

# Konfigurasi
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# Inisialisasi koneksi
@st.cache_resource
def init_supabase():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

@st.cache_resource
def init_gemini():
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    return genai.GenerativeModel('gemini-pro')

supabase = init_supabase()
gemini_model = init_gemini()

# Fungsi pemrosesan PDF
def process_pdf(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_text(text)

# Penyimpanan ke Supabase
def store_in_supabase(chunks, filename):
    try:
        SupabaseVectorStore.from_texts(
            texts=chunks,
            embedding=EMBEDDINGS,
            client=supabase,
            table_name="documents",
            metadatas=[{"filename": filename} for _ in chunks]  # Parameter diperbaiki
        )
        return True
    except Exception as e:
        st.error(f"Gagal menyimpan ke Supabase: {str(e)}")
        st.error(traceback.format_exc())  # Tampilkan detail error
        return False

# Pencarian di Supabase
def search_supabase(query, threshold=0.7):
    try:
        vector_store = SupabaseVectorStore(
            client=supabase,
            embedding=EMBEDDINGS,
            table_name="documents"
        )
        results = vector_store.similarity_search_with_score(query, k=1)
        if results and results[0][1] >= threshold:
            return results[0][0].page_content
        return None
    except Exception as e:
        st.error(f"Error pencarian: {str(e)}")
        return None

# Generate jawaban Gemini
def generate_gemini_response(query):
    try:
        response = gemini_model.generate_content(query)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Tampilkan daftar PDF
def show_pdf_list():
    try:
        query = supabase.table("documents").select("metadata->>filename").execute()
        files = list({item['metadata']['filename'] for item in query.data})
        return files
    except Exception as e:
        st.error(f"Error mengambil daftar PDF: {str(e)}")
        return []

# Antarmuka Streamlit
st.set_page_config(page_title="PDF Chatbot", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("Pengaturan")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file:
        with st.spinner("Memproses PDF..."):
            chunks = process_pdf(uploaded_file)
            if store_in_supabase(chunks, uploaded_file.name):
                st.success(f"File {uploaded_file.name} berhasil diproses!")
            else:
                st.error("Gagal memproses file")

    if st.button("Tampilkan Daftar PDF"):
        pdf_list = show_pdf_list()
        if pdf_list:
            st.subheader("Dokumen Tersedia:")
            for pdf in pdf_list:
                st.write(pdf)
        else:
            st.write("Belum ada PDF yang diupload")

# Area Chat Utama
st.title("PDF Chatbot 🤖")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Apa pertanyaan Anda?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.spinner("Mencari jawaban..."):
        pdf_answer = search_supabase(prompt)
        
        if pdf_answer:
            response = f"📄 Dari dokumen:\n\n{pdf_answer}"
        else:
            response = f"🌐 Dari Gemini:\n\n{generate_gemini_response(prompt)}"
        
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
