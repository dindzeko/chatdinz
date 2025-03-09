import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from supabase import create_client
from typing import List
import os

# Konfigurasi Supabase dari secrets
sb_url = st.secrets["supabase"]["url"]
sb_key = st.secrets["supabase"]["key"]
supabase = create_client(sb_url, sb_key)

# Model embedding
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# Fungsi ekstrak teks dari PDF
def extract_text_from_pdf(file_bytes) -> str:
    try:
        reader = PdfReader(file_bytes)
        return "\n".join(page.extract_text() for page in reader.pages)
    except Exception as e:
        st.error(f"Gagal membaca PDF: {str(e)}")
        return None

# Fungsi split teks dengan sliding window
def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

# Fungsi generate embedding
def generate_embeddings(text_chunks: List[str]) -> List[List[float]]:
    return EMBEDDING_MODEL.encode(text_chunks).tolist()

# Fungsi simpan ke Supabase
def store_embeddings(filename: str, chunks: List[str], embeddings: List[List[float]]):
    try:
        data = [
            {
                "filename": filename,
                "content": chunk,
                "embedding": embedding
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]
        response = supabase.table('pdf_embeddings').insert(data).execute()
        return response.data
    except Exception as e:
        st.error(f"Gagal menyimpan ke Supabase: {str(e)}")
        return None

# Antarmuka Streamlit
st.title("📄 PDF to Embedding Processor")
st.subheader("Ekstrak teks, buat embedding, dan simpan ke Supabase")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
chunk_size = st.slider("Ukuran Chunk (karakter)", 100, 1000, 500, 50)
overlap = st.slider("Overlap (karakter)", 0, 500, 100, 25)

if uploaded_file is not None:
    # Tampilkan info file
    st.write("File terpilih:")
    st.write(f"**Nama**: {uploaded_file.name}")
    st.write(f"**Ukuran**: {round(uploaded_file.size/1024)} KB")
    
    if st.button("Proses PDF", use_container_width=True):
        with st.spinner("Memproses..."):
            # 1. Ekstrak teks
            text = extract_text_from_pdf(uploaded_file)
            if not text:
                st.error("Tidak dapat mengekstrak teks dari PDF")
                st.stop()
            
            # 2. Split teks
            chunks = split_text_into_chunks(
                text,
                chunk_size=chunk_size,
                overlap=overlap
            )
            
            # 3. Generate embedding
            embeddings = generate_embeddings(chunks)
            
            # 4. Simpan ke Supabase
            result = store_embeddings(
                filename=uploaded_file.name,
                chunks=chunks,
                embeddings=embeddings
            )
            
            if result:
                st.success(f"Berhasil menyimpan {len(chunks)} chunks!")
                st.balloons()
                with st.expander("Lihat detail"):
                    st.write(f"Total karakter: {len(text)}")
                    st.write(f"Jumlah chunks: {len(chunks)}")
                    st.write(f"Ukuran embedding: {len(embeddings[0])} dimensi")

# Info teknis di sidebar
with st.sidebar:
    st.info("Pastikan sudah membuat tabel di Supabase dengan perintah:")
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
