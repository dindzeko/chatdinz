import streamlit as st
from supabase import create_client
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
import os
import traceback

# -------------------- Konfigurasi --------------------
# Pastikan ini dijalankan pertama kali
st.set_page_config(page_title="PDF Chatbot", layout="wide")

# Inisialisasi koneksi
@st.cache_resource
def init_services():
    # Supabase
    supabase = create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )
    
    # Google Gemini
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Embedding
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="retrieval_document"
    )
    
    return supabase, gemini_model, embeddings

supabase_client, gemini_model, embeddings = init_services()

# -------------------- Fungsi Utama --------------------
def process_and_store_pdf(pdf_file):
    """Ekstrak teks dari PDF, lakukan chunking, dan simpan ke Supabase"""
    try:
        # Ekstrak teks dari PDF
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        
        # Chunking teks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        
        # Simpan ke Supabase
        SupabaseVectorStore.from_texts(
            texts=chunks,
            embedding=embeddings,
            client=supabase_client,
            table_name="pdf_embeddings",
            query_name="match_documents",
            metadatas=[{"source": pdf_file.name} for _ in chunks]
        )
        
        return True, f"Berhasil memproses {pdf_file.name}"
    
    except Exception as e:
        return False, f"Error: {str(e)}\n{traceback.format_exc()}"

def search_documents(query):
    """Cari dokumen relevan dari Supabase"""
    try:
        vector_store = SupabaseVectorStore(
            client=supabase_client,
            embedding=embeddings,
            table_name="pdf_embeddings",
            query_name="match_documents"
        )
        
        # Cari 3 dokumen teratas dengan threshold 0.7
        results = vector_store.similarity_search_with_score(
            query,
            k=3,
            filter={"source": {"$eq": st.session_state.current_pdf}}
        )
        
        # Filter berdasarkan threshold
        filtered = [doc for doc, score in results if score >= 0.7]
        return filtered
    
    except Exception as e:
        st.error(f"Error pencarian: {str(e)}")
        return []

def generate_response(query, context=None):
    """Buat respons menggunakan Gemini"""
    try:
        if context:
            prompt = f"""Context: {context}
            Pertanyaan: {query}
            
            Berikan jawaban berdasarkan konteks di atas. Jika tidak ada informasi yang relevan, 
            beri tahu pengguna bahwa Anda tidak tahu."""
        else:
            prompt = query
        
        response = gemini_model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        return f"Error: {str(e)}"

# -------------------- Aplikasi Streamlit --------------------
# Inisialisasi session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None

# Sidebar untuk upload PDF
with st.sidebar:
    st.header("Pengaturan")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file:
        success, message = process_and_store_pdf(uploaded_file)
        if success:
            st.session_state.current_pdf = uploaded_file.name
            st.success(message)
        else:
            st.error(message)

# Tampilan utama
st.title("PDF Chatbot 📄🤖")

# Tampilkan riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input pengguna
if prompt := st.chat_input("Apa pertanyaan Anda?"):
    # Tambahkan pesan pengguna ke riwayat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Proses pertanyaan
    with st.spinner("Mencari jawaban..."):
        # Cari konteks dari Supabase
        context_docs = search_documents(prompt)
        context = "\n\n".join([doc.page_content for doc in context_docs]) if context_docs else None
        
        # Generate respons
        response = generate_response(prompt, context)
        
        # Tambahkan indikasi sumber
        if context_docs:
            source_note = f" (Sumber: {st.session_state.current_pdf})"
        else:
            source_note = " (Sumber: Pengetahuan Gemini)"
        
        # Tampilkan respons
        with st.chat_message("assistant"):
            st.markdown(response)
            st.caption(f"Jawaban{source_note}")
        
        # Simpan ke riwayat
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
