# app.py
import streamlit as st
from supabase import create_client
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.supabase import SupabaseVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
import os

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
        text += page.extract_text()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Penyimpanan ke Supabase
def store_in_supabase(chunks, filename):
    vector_store = SupabaseVectorStore.from_texts(
        chunks,
        EMBEDDINGS,
        client=supabase,
        table_name="documents",
        chunk_size=500,
        metadata=[{"filename": filename}]*len(chunks)
    )

# Pencarian di Supabase
def search_supabase(query, threshold=0.7):
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=EMBEDDINGS,
        table_name="documents"
    )
    
    results = vector_store.similarity_search_with_score(query, k=1)
    if results and results[0][1] > threshold:
        return results[0][0].page_content
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
    query = supabase.table("documents").select("metadata->filename").execute()
    files = {str(item['metadata']['filename']) for item in query.data}
    return list(files)

# Antarmuka Streamlit
st.set_page_config(page_title="PDF Chatbot", layout="wide")

# Inisialisasi session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("Pengaturan")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file:
        with st.spinner("Memproses PDF..."):
            chunks = process_pdf(uploaded_file)
            store_in_supabase(chunks, uploaded_file.name)
            st.success(f"File {uploaded_file.name} berhasil diproses!")

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

# Tampilkan riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input pengguna
if prompt := st.chat_input("Apa pertanyaan Anda?"):
    # Tambahkan ke riwayat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Tampilkan pesan user
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Cari jawaban
    with st.spinner("Mencari jawaban..."):
        # Cari di PDF
        pdf_answer = search_supabase(prompt)
        
        if pdf_answer:
            response = f"📄 Dari dokumen:\n\n{pdf_answer}"
        else:
            # Fallback ke Gemini
            response = f"🌐 Dari Gemini:\n\n{generate_gemini_response(prompt)}"
        
        # Tampilkan response
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Simpan ke riwayat
        st.session_state.messages.append({"role": "assistant", "content": response})
