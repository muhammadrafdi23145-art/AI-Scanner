import streamlit as st
import os
import tempfile
import uuid
import time
from pydantic import BaseModel, Field
from typing import List

# Import LangChain & AI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser

st.set_page_config(page_title="Elite Global Ranking", layout="wide")

st.title("Ranking Selection System")
st.markdown("*Sistem Seleksi Data Scientist Kelas Dunia berbasis AI (RAG)*")

# ==============================================================================
# 1. SIDEBAR & KONFIGURASI
# ==============================================================================
with st.sidebar:
    st.header("Konfigurasi Sistem")
    api_key = st.text_input("Google API Key", type="password", help="Masukkan API Key Gemini Anda")
    model_name = st.selectbox("Pilih Model AI", ["gemini-3-flash-preview", "gemini-1.5-flash", "gemini-1.5-pro"])
    
    st.markdown("---")
    st.markdown("**Kriteria Pencarian:**")
    kriteria_default = """Mencari Data Scientist kelas dunia:
- Mahir Python, Cloud Computing, dan Deep Learning.
- Paham lifecycle AI dari riset hingga deployment (MLOps).
- Memiliki latar belakang pendidikan dari universitas dengan peringkat global yang tinggi."""
    kriteria = st.text_area("Detail Kriteria", value=kriteria_default, height=200)

# ==============================================================================
# 2. SKEMA JSON
# ==============================================================================
class StrategiInterview(BaseModel):
    alasan_layak: str = Field(description="Alasan mengapa kandidat layak")
    celah_teknis: str = Field(description="Satu celah atau tantangan teknis")
    pertanyaan_lanjutan: List[str] = Field(description="Daftar 3 pertanyaan wawancara tingkat lanjut")

class Kandidat(BaseModel):
    peringkat: int = Field(description="Peringkat kandidat")
    nama_file: str = Field(description="Nama file CV")
    nama_kandidat: str = Field(description="Nama lengkap")
    skor_global: int = Field(description="Skor keseluruhan (0-100)")
    universitas: str = Field(description="Universitas asal")
    estimasi_rank_dunia: str = Field(description="Estimasi peringkat QS World")
    alasan_utama: str = Field(description="Alasan utama skor")
    strategi_interview: StrategiInterview = Field(description="Strategi wawancara")

class RankingOutput(BaseModel):
    kandidat: List[Kandidat] = Field(description="Daftar kandidat yang dievaluasi")

# ==============================================================================
# 3. FUNGSI CORE AI
# ==============================================================================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def elite_global_ranking(pdf_paths, criteria, api_key, model_name):
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.1)
    embeddings = load_embeddings()

    all_docs = []
    for path in pdf_paths:
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["candidate_name"] = os.path.basename(path)
            all_docs.extend(docs)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")

    if not all_docs:
        return {"error": "Tidak ada dokumen yang berhasil diproses."}

    dynamic_collection_name = f"eval_session_{uuid.uuid4().hex}"
    vectorstore = Chroma.from_documents(documents=all_docs, embedding=embeddings, collection_name=dynamic_collection_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) 
    parser = JsonOutputParser(pydantic_object=RankingOutput)

    template = """
    Tugas: Anda adalah Global Talent Headhunter.
    Tujuan: Menyeleksi kandidat Data Science terbaik dunia.

    {context}

    Kriteria: {question}

    Logika Penilaian:
    1. Global Academic Prestige (50%): Lulusan Top 100 Global dapat nilai maksimal.
    2. Technical Prowess (50%): Fokus Python, SOTA AI Models, MLOps.

    TIDAK BOLEH ADA TEKS LAIN SELAIN JSON. PATUHI FORMAT BERIKUT:
    {format_instructions}
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"], partial_variables={"format_instructions": parser.get_format_instructions()})
    chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | parser)

    try:
        return chain.invoke(criteria)
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            vectorstore.delete_collection()
        except:
            pass

# ==============================================================================
# 4. ANTARMUKA UTAMA (MAIN UI)
# ==============================================================================
uploaded_files = st.file_uploader("Unggah CV Kandidat (Bisa lebih dari 1 PDF)", type="pdf", accept_multiple_files=True)

if st.button("Mulai Analisis Ranking", type="primary", use_container_width=True):
    if not api_key:
        st.warning("Masukkan Google API Key di sidebar sebelah kiri terlebih dahulu.")
    elif not uploaded_files:
        st.warning("Silakan unggah minimal 1 file CV (PDF).")
    else:
        # Menyimpan file upload ke temporary folder di cloud
        temp_dir = tempfile.mkdtemp()
        saved_paths = []
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            saved_paths.append(file_path)

        with st.spinner(f"AI sedang menganalisis {len(saved_paths)} CV. Mohon tunggu..."):
            hasil = elite_global_ranking(saved_paths, kriteria, api_key, model_name)
            
            if "error" not in hasil and "kandidat" in hasil:
                kandidat_list = hasil["kandidat"]
                kandidat_list.sort(key=lambda x: x.get('skor_global', 0), reverse=True)
                for urutan, kand in enumerate(kandidat_list):
                    kand['peringkat'] = urutan + 1

                st.success("Analisis Selesai! Berikut adalah daftar ranking kandidat:")
                
                # Tabel Ringkasan
                ringkasan = [{"Peringkat": k['peringkat'], "Kandidat": k['nama_kandidat'], "Skor": k['skor_global'], "Universitas": k['universitas']} for k in kandidat_list]
                st.table(ringkasan)

                # Detail Accordion
                st.markdown("### Detail Kandidat & Strategi Wawancara")
                for kand in kandidat_list:
                    with st.expander(f"Rank {kand['peringkat']} - {kand['nama_kandidat']} (Skor: {kand['skor_global']})"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Universitas:** {kand['universitas']} *(Estimasi QS: {kand['estimasi_rank_dunia']})*")
                            st.write(f"**Alasan Utama:** {kand['alasan_utama']}")
                        with col2:
                            st.write("**Strategi Interview:**")
                            st.success(f"**Kelebihan:** {kand['strategi_interview']['alasan_layak']}")
                            st.warning(f"**Celah Teknis:** {kand['strategi_interview']['celah_teknis']}")
                            
                        st.write("**Pertanyaan Lanjutan:**")
                        for q in kand['strategi_interview']['pertanyaan_lanjutan']:
                            st.write(f"- {q}")
            else:
                st.error(f"Terjadi kesalahan saat pemrosesan: {hasil.get('error')}")
