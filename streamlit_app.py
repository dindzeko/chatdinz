import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from supabase import create_client
from typing import List
import numpy as np

# ================================================
# KONFIGURASI AWAL
# ================================================
# Pastikan sudah membuat tabel dengan:
# CREATE EXTENSION IF NOT EXISTS vector;
# CREATE TABLE pdf_embeddings (
#     id SERIAL PRIMARY KEY,
#     filename TEXT NOT NULL,
#     content TEXT NOT NULL,
#     embedding VECTOR(384) NOT NULL,
#     created_at TIMESTAMP DEFAULT NOW()
# );
#
# Policy RLS untuk INSERT:
# CREATE POLICY insert_policy ON pdf_embeddings FOR INSERT TO anon WITH CHECK (true);
# GRANT INSERT ON TABLE pdf_embeddings TO anon;
# ================================================

# Konfigurasi Supabase
sb_url = st.secrets["supabase"]["url"]
sb_key = st.secrets["supabase"]["key"]
supabase = create_client(sb_url, sb_key)

# Model embedding (384 dimensi)
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
EXPECTED_DIMENSIONS = 384

def extract_text_from_pdf(file_bytes) -> str:
    """Ekstrak teks dari PDF dengan validasi"""
    try:
        reader = PdfReader(file_bytes)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        if not text.strip():
            raise ValueError("PDF tidak mengandung teks")
        return text
    except Exception as e:
        st.error(f"Gagal membaca PDF: {str(e)}")
        return None

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Pecah teks dengan sliding window dan validasi"""
    if len(text) < chunk_size:
        raise ValueError("Ukuran teks lebih kecil dari chunk size")
        
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
    """Buat embedding dengan validasi dimensi"""
    embeddings = EMBEDDING_MODEL.encode(text_chunks)
    
    # Validasi dimensi
    if embeddings.shape[1] != EXPECTED_DIMENSIONS:
        raise ValueError(f"Dimensi model tidak sesuai: {embeddings.shape[1]}")
        
    return embeddings.tolist()

def store_embeddings(filename: str, chunks: List[str], embeddings: List[List[float]]):
    """Simpan ke Supabase dengan validasi ketat"""
    try:
        # Validasi korespondensi data
        if len(chunks) != len(embeddings):
            raise ValueError(f"Jumlah chunk ({len(chunks)}) != embedding ({len(embeddings)})")
            
        # Validasi dimensi embedding
        for idx, emb in enumerate(embeddings):
            if len(emb) != EXPECTED_DIMENSIONS:
                raise ValueError(f"Chunk {idx+1} dimensi tidak valid: {len(emb)} (harus {EXPECTED_DIMENSIONS})")

        # Format data untuk Supabase
        data = [
            {
                "filename": filename,
                "content": chunk,
                "embedding": np.array(emb, dtype=np.float32).tolist()
            }
            for chunk, emb in zip(chunks, embeddings)
        ]
        
        # Insert ke Supabase dengan timeout
        response = supabase.table('pdf_embeddings').insert(data, timeout=20).execute()

        # Handle error response
        if hasattr(response, 'error') and response.error:
            error_code = response.error.code
            error_msg = response.error.message
            
            # Handle error RLS khusus
            if error_code == '42501':
                st.error("🚨 Error RLS Policy:")
                st.markdown(f"""
                ```sql
                -- Pastikan sudah menjalankan:
                CREATE POLICY insert_policy ON pdf_embeddings 
                FOR INSERT TO anon 
                WITH CHECK (true);
                
                GRANT INSERT ON TABLE pdf_embeddings TO anon;
                ```
                """)
            else:
                st.error(f"Supabase Error [{error_code}]: {error_msg}")
            return None

        return response.data

    except ValueError as ve:
        st.error(f"⚠️ Validasi Error: {str(ve)}")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

# ================================================
# ANTARMUKA STREAMLIT
# ================================================
st.title("📄 PDF to Embedding Processor")
st.warning("Pastikan sudah setup tabel & policy RLS di Supabase!")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
chunk_size = st.slider("Ukuran Chunk (karakter)", 100, 1000, 500, 50)
overlap = st.slider("Overlap (karakter)", 0, 500, 100, 25)

if uploaded_file is not None:
    st.write("**File Info:**")
    st.write(f"- Nama: `{uploaded_file.name}`")
    st.write(f"- Ukuran: {round(uploaded_file.size/1024)} KB")
    
    if st.button("Proses PDF", use_container_width=True):
        with st.spinner("Processing..."):
            # 1. Ekstrak teks
            text = extract_text_from_pdf(uploaded_file)
            if not text:
                st.error("Gagal mengekstrak teks dari PDF")
                st.stop()
                
            # 2. Split teks
            try:
                chunks = split_text_into_chunks(text, chunk_size, overlap)
                if len(chunks) == 0:
                    raise ValueError("Tidak ada chunk valid")
            except Exception as e:
                st.error(f"Split Error: {str(e)}")
                st.stop()
                
            # 3. Generate embedding
            try:
                embeddings = generate_embeddings(chunks)
            except Exception as e:
                st.error(f"Embedding Error: {str(e)}")
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
                    st.write(f"- Contoh embedding: `{embeddings[0][:5]}...`")

# Troubleshooting di sidebar
with st.sidebar:
    st.header("🛠 Troubleshooting")
    st.info("""
    1. Pastikan sudah membuat tabel dengan SQL yang disediakan
    2. Cek policy RLS untuk role anon di Supabase
    3. Verifikasi koneksi internet dan credentials
    4. Coba insert manual via SQL Editor Supabase:
       ```sql
       INSERT INTO pdf_embeddings 
       (filename, content, embedding) 
       VALUES ('test.pdf', 'test', '[0.1,0.2,...,0.384]');
       ```
    """)
    
    if st.checkbox("Show Model Info"):
        st.write("Model:", EMBEDDING_MODEL)
        st.write("Dimensi:", EMBEDDING_MODEL.get_sentence_embedding_dimension())
