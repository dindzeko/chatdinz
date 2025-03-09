import streamlit as st
from supabase import create_client, Client
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
import traceback

# -------------------- Konfigurasi --------------------
st.set_page_config(page_title="PDF Chatbot", layout="wide")

# Cek secrets dasar
required_secrets = ["SUPABASE_URL", "SUPABASE_KEY", "GEMINI_API_KEY"]
missing_secrets = [s for s in required_secrets if s not in st.secrets]

if missing_secrets:
    st.error(f"Konfigurasi tidak lengkap: {', '.join(missing_secrets)} belum diset")
    st.stop()

# -------------------- Inisialisasi Layanan --------------------
def check_supabase_connection(supabase: Client) -> bool:
    """Cek koneksi ke Supabase"""
    try:
        # Coba akses tabel sistem untuk verifikasi
        supabase.table("any_table").select("*").execute()
        return True
    except Exception as e:
        if "404" in str(e):
            # Tabel tidak ada, tapi koneksi berhasil
            return True
        st.error(f"Supabase Error: {str(e)}")
        return False

def check_gemini_connection() -> bool:
    """Cek koneksi ke Google Gemini"""
    try:
        genai.list_models()
        return True
    except Exception as e:
        st.error(f"Gemini Error: {str(e)}")
        return False

# Inisialisasi dengan status visual
with st.spinner("Menghubungkan ke layanan..."):
    # Inisialisasi Supabase
    try:
        supabase_client = create_client(
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_KEY"]
        )
    except Exception as e:
        st.error(f"Gagal inisialisasi Supabase: {str(e)}")
        st.stop()

    # Inisialisasi Gemini
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        st.error(f"Gagal konfigurasi Gemini: {str(e)}")
        st.stop()

    # Cek koneksi
    supabase_connected = check_supabase_connection(supabase_client)
    gemini_connected = check_gemini_connection()

# Tampilkan status koneksi
if supabase_connected and gemini_connected:
    st.success("✅ Koneksi berhasil!")
    st.info("Supabase: Terhubung | Gemini: Tersedia model {}".format(
        ', '.join([m.name for m in genai.list_models()])
    ))
elif supabase_connected:
    st.warning("⚠️ Supabase terhubung, Gemini gagal")
    st.stop()
elif gemini_connected:
    st.warning("⚠️ Gemini terhubung, Supabase gagal")
    st.stop()
else:
    st.error("❌ Gagal terhubung ke semua layanan")
    st.stop()

# -------------------- Fungsi Utama --------------------
def process_and_store_pdf(pdf_file):
    """Ekstrak teks dari PDF, lakukan chunking, dan simpan ke Supabase"""
    try:
        # Ekstrak teks dari PDF
        pdf_reader = PdfReader(pdf_file)
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        
        # Chunking teks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        
        # Simpan ke Supabase
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_document"
        )
        
        with st.spinner("Menyimpan ke database..."):
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
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_document"
        )
        
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
            filter={"source": {"$eq": st.session_state.get("current_pdf")}}
        )
        
        return [doc for doc, score in results if score >= 0.7]
    
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
        with st.spinner("Memproses PDF..."):
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
        source_note = (
            f" (Sumber: {st.session_state.current_pdf})"
            if context_docs
            else " (Sumber: Pengetahuan Gemini)"
        )
        
        # Tampilkan respons
        with st.chat_message("assistant"):
            st.markdown(response)
            st.caption(f"Jawaban{source_note}")
        
        # Simpan ke riwayat
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
