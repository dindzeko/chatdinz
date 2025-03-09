import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from supabase import create_client
from typing import List
import numpy as np

# Konfigurasi Supabase
sb_url = st.secrets["supabase"]["url"]
sb_key = st.secrets["supabase"]["key"]
supabase = create_client(sb_url, sb_key)

# Model embedding (384 dimensi)
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(file_bytes) -> str:
    """Ekstrak teks dari file PDF"""
    try:
        reader = PdfReader(file_bytes)
        return "\n".join(page.extract_text() for page in reader.pages)
    except Exception as e:
        st.error(f"Gagal membaca PDF: {str(e)}")
        return None

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Pecah teks dengan sliding window"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def generate_embeddings(text_chunks: List[str]) -> List[List[float]]:
    """Buat embedding dengan validasi dimensi"""
    embeddings = EMBEDDING_MODEL.encode(text_chunks).tolist()
    if len(embeddings) == 0:
        raise ValueError("Tidak ada embedding yang dihasilkan")
    if len(embeddings[0]) != 384:
        raise ValueError(f"Dimensi embedding tidak valid: {len(embeddings[0])}")
    return embeddings

def store_embeddings(filename: str, chunks: List[str], embeddings: List[List[float]]):
    """Simpan ke Supabase dengan validasi"""
    try:
        # Validasi panjang data
        if len(chunks) != len(embeddings):
            raise ValueError("Jumlah chunk dan embedding tidak sama")
            
        data = [
            {
                "filename": filename,
                "content": chunk,
                "embedding": np.array(embedding, dtype=np.float32).tolist()  # Pastikan format numerik
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]
        
        response = supabase.table('pdf_embeddings').insert(data).execute()
        
        if hasattr(response, 'error') and response.error:
            st.error(f"Error Supabase: {response.error.message}")
            return None
            
        return response.data
        
    except Exception as e:
        st.error(f"Gagal menyimpan: {str(e)}")
        return None

# ===== Antarmuka Streamlit =====
st.title("📄 PDF to Embedding Processor")
st.subheader("Ekstrak teks, buat embedding, dan simpan ke Supabase")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
chunk_size = st.slider("Ukuran Chunk (karakter)", 100, 1000, 500, 50)
overlap = st.slider("Overlap (karakter)", 0, 500, 100, 25)

if uploaded_file is not None:
    st.write("**File terpilih:**")
    st.write(f"Nama: `{uploaded_file.name}`")
    st.write(f"Ukuran: {round(uploaded_file.size/1024)} KB")
    
    if st.button("Proses PDF", use_container_width=True):
        with st.spinner("Memproses..."):
            # 1. Ekstrak teks
            text = extract_text_from_pdf(uploaded_file)
            if not text:
                st.error("Gagal mengekstrak teks dari PDF")
                st.stop()
                
            # 2. Split teks
            chunks = split_text_into_chunks(text, chunk_size, overlap)
            if len(chunks) == 0:
                st.error("Tidak ada chunk yang dihasilkan")
                st.stop()
                
            # 3. Generate embedding
            try:
                embeddings = generate_embeddings(chunks)
            except Exception as e:
                st.error(f"Error embedding: {str(e)}")
                st.stop()
                
            # 4. Simpan ke Supabase
            result = store_embeddings(
                filename=uploaded_file.name,
                chunks=chunks,
                embeddings=embeddings
            )
            
            if result:
                st.success(f"✅ {len(chunks)} chunks berhasil disimpan!")
                st.balloons()
                with st.expander("Detail Teknis"):
                    st.write(f"- Total karakter: {len(text)}")
                    st.write(f"- Jumlah chunks: {len(chunks)}")
                    st.write(f"- Dimensi embedding: {len(embeddings[0])}")
                    st.write(f"- Tabel tujuan: `pdf_embeddings`")

# Informasi teknis di sidebar
with st.sidebar:
    st.warning("Pastikan sudah membuat tabel dengan:")
    st.code('''
    CREATE EXTENSION IF NOT EXISTS vector;
    
    CREATE TABLE pdf_embeddings (
        id SERIAL PRIMARY KEY,
        filename TEXT NOT NULL,
        content TEXT NOT NULL,
        embedding VECTOR(384) NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    );
    ''')
    
    st.info("Policy RLS harus mengizinkan INSERT untuk role anon")
