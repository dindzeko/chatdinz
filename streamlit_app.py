import streamlit as st
from supabase import create_client, Client
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
import traceback
import httpx

# -------------------- KONFIGURASI AWAL --------------------
st.set_page_config(page_title="PDF Chatbot", layout="wide")

# Cek kelengkapan konfigurasi
REQUIRED_SECRETS = ["SUPABASE_URL", "SUPABASE_KEY", "GEMINI_API_KEY"]
missing_secrets = [s for s in REQUIRED_SECRETS if s not in st.secrets]

if missing_secrets:
    st.error(f"⚠️ Konfigurasi tidak lengkap: {', '.join(missing_secrets)} belum diset")
    st.stop()

# -------------------- INISIALISASI LAYANAN --------------------
def check_supabase_connection(url: str, key: str) -> bool:
    """Memverifikasi koneksi ke Supabase"""
    try:
        headers = {"apikey": key, "Authorization": f"Bearer {key}"}
        response = httpx.get(f"{url}/rest/v1/", headers=headers, timeout=10)
        return response.status_code in (200, 401)  # 401 = kunci salah tapi server aktif
    except Exception as e:
        st.error(f"❌ Gagal terhubung ke Supabase: {str(e)}")
        return False

def initialize_services():
    """Inisialisasi dan validasi semua dependensi"""
    with st.spinner("🔌 Menghubungkan ke layanan..."):
        try:
            # Inisialisasi Supabase
            sb_client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
            if not check_supabase_connection(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"]):
                return None, None
            
            # Inisialisasi Gemini
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            genai.get_model('gemini-pro')  # Validasi model
            gemini_model = genai.GenerativeModel('gemini-pro')
            
            return sb_client, gemini_model
            
        except Exception as e:
            st.error(f"🚨 Gagal inisialisasi: {str(e)}")
            return None, None

# -------------------- FUNGSI UTAMA --------------------
def proses_pdf(pdf_file):
    """Ekstrak dan simpan data PDF ke database"""
    try:
        # Ekstrak teks
        pdf_reader = PdfReader(pdf_file)
        text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
        
        # Split dokumen
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "]
        )
        chunks = splitter.split_text(text)
        
        # Simpan embedding
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_document"
        )
        
        with st.spinner("💾 Menyimpan ke database..."):
            SupabaseVectorStore.from_texts(
                texts=chunks,
                embedding=embeddings,
                client=st.session_state.supabase,
                table_name="documents",
                query_name="match_documents",
                metadatas=[{"source": pdf_file.name}]*len(chunks)
            )
            
        return True, f"✅ Berhasil memproses {pdf_file.name}"
        
    except Exception as e:
        return False, f"❌ Error: {str(e)}\n{traceback.format_exc()}"

def cari_dokumen(query):
    """Cari dokumen relevan dari database"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_query"
        )
        
        vector_store = SupabaseVectorStore(
            client=st.session_state.supabase,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents"
        )
        
        results = vector_store.similarity_search(
            query, 
            k=3,
            filter={"source": st.session_state.current_pdf}
        )
        
        return results if results else None
        
    except Exception as e:
        st.error(f"🔍 Gagal pencarian: {str(e)}")
        return None

def generate_jawaban(query, context=None):
    """Generate jawaban menggunakan AI"""
    try:
        prompt_template = """
        {context}
        
        Pertanyaan: {query}
        
        Jawablah dengan:
        1. Menggunakan bahasa Indonesia yang baik
        2. Hanya berdasarkan konteks di atas
        3. Jika tidak tahu, jawab 'Maaf, informasi tidak ditemukan dalam dokumen'
        """
        
        prompt = prompt_template.format(
            context=f"Konteks:\n{context}" if context else "",
            query=query
        )
        
        response = st.session_state.gemini.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# -------------------- ANTARMUKA PENGGUNA --------------------
# Inisialisasi state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None
if "supabase" not in st.session_state:
    st.session_state.supabase = None
if "gemini" not in st.session_state:
    st.session_state.gemini = None

# Sidebar untuk upload PDF
with st.sidebar:
    st.subheader("📁 Kelola Dokumen")
    uploaded_file = st.file_uploader("Unggah PDF", type=["pdf"])
    
    if uploaded_file:
        if st.session_state.supabase:
            success, msg = proses_pdf(uploaded_file)
            if success:
                st.session_state.current_pdf = uploaded_file.name
                st.success(msg)
            else:
                st.error(msg)
        else:
            st.error("❌ Silakan tunggu inisialisasi selesai")

# Main UI
st.title("💬 PDF Chatbot Cerdas")

# Inisialisasi layanan
if not st.session_state.supabase or not st.session_state.gemini:
    st.session_state.supabase, st.session_state.gemini = initialize_services()
    
    if st.session_state.supabase and st.session_state.gemini:
        st.success("✅ Layanan terhubung dengan sukses!")
    else:
        st.error("❌ Gagal menghubungkan ke layanan. Periksa konfigurasi.")
        st.stop()

# Tampilkan riwayat chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input pengguna
if prompt := st.chat_input("Apa pertanyaan Anda?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.spinner("🔎 Mencari jawaban..."):
        try:
            # Langkah 1: Cari konteks
            konteks = cari_dokumen(prompt)
            konteks_text = "\n\n".join([doc.page_content for doc in konteks]) if konteks else None
            
            # Langkah 2: Generate jawaban
            jawaban = generate_jawaban(prompt, konteks_text)
            
            # Tampilkan sumber
            sumber = st.session_state.current_pdf if konteks else "Pengetahuan Umum"
            
            # Tampilkan jawaban
            with st.chat_message("assistant"):
                st.markdown(jawaban)
                if konteks:
                    st.caption(f"Sumber: {sumber}")
            
            st.session_state.messages.append({"role": "assistant", "content": jawaban})
            
        except Exception as e:
            st.error(f"🚨 Gagal memproses: {str(e)}")
