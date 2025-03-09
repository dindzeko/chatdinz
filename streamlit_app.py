import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from supabase.exceptions import APIError
from typing import List
import numpy as np

# Konfigurasi Supabase
sb_url = st.secrets["supabase"]["url"]
sb_key = st.secrets["supabase"]["key"]
supabase: Client = create_client(sb_url, sb_key)

# Model embedding
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
EXPECTED_DIMENSIONS = 384

def extract_text_from_pdf(file_bytes) -> str:
    try:
        reader = PdfReader(file_bytes)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        st.error(f"Gagal membaca PDF: {str(e)}")
        return None

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
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
    embeddings = EMBEDDING_MODEL.encode(text_chunks)
    if embeddings.shape[1] != EXPECTED_DIMENSIONS:
        raise ValueError(f"Dimensi model tidak sesuai: {embeddings.shape[1]}")
    return embeddings.tolist()

def store_embeddings(filename: str, chunks: List[str], embeddings: List[List[float]]):
    try:
        if len(chunks) != len(embeddings):
            raise ValueError("Jumlah chunk dan embedding tidak sama")
        for emb in embeddings:
            if len(emb) != EXPECTED_DIMENSIONS:
                raise ValueError("Dimensi embedding tidak valid")
                
        data = [
            {
                "filename": filename,
                "content": chunk,
                "embedding": np.array(emb, dtype=np.float32).tolist()
            }
            for chunk, emb in zip(chunks, embeddings)
        ]
        
        response = supabase.table('pdf_embeddings').insert(data).execute()
        
        # Error handling untuk Supabase v1.0+
        if hasattr(response, 'error') and response.error:
            raise APIError(response.error.message, response.error.code)
            
        return response.data

    except APIError as e:
        if e.code == '42501':
            st.error("🚨 Error RLS Policy!")
            st.markdown('''
            Pastikan sudah:
            1. Membuat policy INSERT untuk role `anon`
            2. Memberikan hak INSERT ke role `anon`
            ''')
        else:
            st.error(f"Supabase Error: [{e.code}] {e.message}")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Antarmuka Streamlit
st.title("📄 PDF to Embedding Processor")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
chunk_size = st.slider("Ukuran Chunk (karakter)", 100, 1000, 500, 50)
overlap = st.slider("Overlap (karakter)", 0, 500, 100, 25)

if uploaded_file is not None:
    st.write(f"**File:** {uploaded_file.name} ({round(uploaded_file.size/1024)} KB)")
    
    if st.button("Proses PDF", use_container_width=True):
        with st.spinner("Memproses..."):
            text = extract_text_from_pdf(uploaded_file)
            if not text:
                st.error("Gagal mengekstrak teks dari PDF")
                st.stop()
                
            chunks = split_text_into_chunks(text, chunk_size, overlap)
            if len(chunks) == 0:
                st.error("Tidak ada chunk valid yang dihasilkan")
                st.stop()
                
            try:
                embeddings = generate_embeddings(chunks)
            except Exception as e:
                st.error(f"Error embedding: {str(e)}")
                st.stop()
                
            result = store_embeddings(uploaded_file.name, chunks, embeddings)
            
            if result:
                st.success(f"✅ {len(chunks)} chunks berhasil disimpan!")
                st.balloons()
                with st.expander("Detail"):
                    st.write(f"- Total karakter: {len(text)}")
                    st.write(f"- Contoh embedding: `{embeddings[0][:5]}...`")

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
    
    CREATE POLICY insert_policy ON pdf_embeddings 
    FOR INSERT TO anon 
    WITH CHECK (true);
    
    GRANT INSERT ON TABLE pdf_embeddings TO anon;
    ''')
