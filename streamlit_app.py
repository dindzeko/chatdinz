import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from supabase import create_client
from typing import List
import numpy as np

# ================================================
# KONFIGURASI SUPABASE
# ================================================
sb_url = st.secrets["supabase"]["url"]
sb_key = st.secrets["supabase"]["key"]
supabase = create_client(sb_url, sb_key)

# Model embedding (384 dimensi)
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
EXPECTED_DIMENSIONS = 384

# ================================================
# FUNGSI UTAMA
# ================================================
def extract_text_from_pdf(file_bytes) -> str:
    """Ekstrak teks dari PDF"""
    try:
        reader = PdfReader(file_bytes)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        st.error(f"Gagal membaca PDF: {str(e)}")
        return None

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Pecah teks dengan sliding window"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) >= 10:
            chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

def generate_embeddings(text_chunks: List[str]) -> List[List[float]]:
    """Buat embedding dari teks"""
    embeddings = EMBEDDING_MODEL.encode(text_chunks)
    if embeddings.shape[1] != EXPECTED_DIMENSIONS:
        raise ValueError(f"Dimensi model tidak sesuai: {embeddings.shape[1]}")
    return embeddings.tolist()

def store_embeddings(filename: str, chunks: List[str], embeddings: List[List[float]]):
    """Simpan data ke Supabase"""
    try:
        # Validasi data
        if len(chunks) != len(embeddings):
            raise ValueError("Jumlah chunk dan embedding tidak sama")
        for emb in embeddings:
            if len(emb) != EXPECTED_DIMENSIONS:
                raise ValueError("Dimensi embedding tidak valid")
                
        # Format data
        data = [
            {
                "filename": filename,
                "content": chunk,
                "embedding": np.array(emb, dtype=np.float32).tolist()
            }
            for chunk, emb in zip(chunks, embeddings)
        ]
        
        # Simpan ke Supabase
        response = supabase.table('pdf_embeddings').insert(data).execute()
        
        if response.error:
            if response.error.code == '42501':
                st.error("Error RLS Policy! Pastikan sudah membuat policy INSERT untuk role anon")
            else:
                st.error(f"Error Supabase: {response.error.message}")
            return None
            
        return response.data
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# ================================================
# ANTARMUKA STREAMLIT
# ================================================
st.title("📄 PDF to Embedding Processor")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
chunk_size = st.slider("Ukuran Chunk (karakter)", 100, 1000, 500, 50)
overlap = st.slider("Overlap (karakter)", 0, 500, 100, 25)

if uploaded_file is not None:
    st.write(f"**File:** {uploaded_file.name} ({round(uploaded_file.size/1024)} KB)")
    
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
                st.error("Tidak ada chunk valid yang dihasilkan")
                st.stop()
                
            # 3. Generate embedding
            try:
                embeddings = generate_embeddings(chunks)
            except Exception as e:
                st.error(f"Error embedding: {str(e)}")
                st.stop()
                
            # 4. Simpan ke Supabase
            result = store_embeddings(uploaded_file.name, chunks, embeddings)
            
            if result:
                st.success(f"✅ {len(chunks)} chunks berhasil disimpan!")
                st.balloons()
                with st.expander("Detail"):
                    st.write(f"- Total karakter: {len(text)}")
                    st.write(f"- Contoh embedding: `{embeddings[0][:5]}...`")

# Info troubleshooting
with st.sidebar:
    st.warning("Pastikan sudah membuat:")
    st.code('''
    CREATE TABLE pdf_embeddings (
        id SERIAL PRIMARY KEY,
        filename TEXT NOT NULL,
        content TEXT NOT NULL,
        embedding VECTOR(384) NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    );
    
    -- Policy RLS untuk role anon
    CREATE POLICY insert_policy ON pdf_embeddings 
    FOR INSERT TO anon 
    WITH CHECK (true);
    
    GRANT INSERT ON TABLE pdf_embeddings TO anon;
    ''')
