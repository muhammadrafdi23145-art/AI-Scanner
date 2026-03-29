import streamlit as st
import os
import tempfile
import uuid
import time
import re
from pydantic import BaseModel, Field
from typing import List
from duckduckgo_search import DDGS

# Import LangChain & AI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser

st.set_page_config(page_title="Ranking Selection System", layout="wide")

st.title("Ranking Selection System")
st.markdown("*Sistem Seleksi Data Scientist Kelas Dunia berbasis AI & Open-Source Intelligence (OSINT)*")

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
    celah_teknis: str = Field(description="Satu celah atau tantangan teknis atau red flag dari sosmed")
    pertanyaan_lanjutan: List[str] = Field(description="Daftar 3 pertanyaan wawancara tingkat lanjut")

class Kandidat(BaseModel):
    peringkat: int = Field(description="Peringkat kandidat")
    nama_file: str = Field(description="Nama file CV")
    nama_kandidat: str = Field(description="Nama lengkap")
    skor_global: int = Field(description="Skor keseluruhan (0-100)")
    universitas: str = Field(description="Universitas asal")
    estimasi_rank_dunia: str = Field(description="Estimasi peringkat QS World")
    jejak_digital: str = Field(description="Analisis komprehensif rekam jejak digital berdasarkan input manual rekruter dan penelusuran internet")
    alasan_utama: str = Field(description="Alasan utama skor")
    strategi_interview: StrategiInterview = Field(description="Strategi wawancara")

class RankingOutput(BaseModel):
    kandidat: List[Kandidat] = Field(description="Daftar kandidat yang dievaluasi")

# ==============================================================================
# 3. FUNGSI CORE AI & SMART OSINT
# ==============================================================================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ekstrak_username_dari_teks(teks_cv):
    usernames = []
    linkedin = re.search(r'linkedin\.com/in/([a-zA-Z0-9_-]+)', teks_cv, re.IGNORECASE)
    if linkedin: usernames.append(f"linkedin.com/in/{linkedin.group(1)}")
    github = re.search(r'github\.com/([a-zA-Z0-9_-]+)', teks_cv, re.IGNORECASE)
    if github: usernames.append(f"github.com/{github.group(1)}")
    kaggle = re.search(r'kaggle\.com/([a-zA-Z0-9_-]+)', teks_cv, re.IGNORECASE)
    if kaggle: usernames.append(f"kaggle.com/{kaggle.group(1)}")
    return usernames

def cari_jejak_digital(nama_file, teks_cv, input_manual_rekruter):
    """Mencari jejak digital dengan menggabungkan input manual dan internet search"""
    hasil_teks = ""
    
    # 1. Prioritaskan Input Manual dari Rekruter
    if input_manual_rekruter:
        hasil_teks += f"[CATATAN REKRUTER / MEDIA SOSIAL MANUAL]: {input_manual_rekruter}\n\n"
        kueri = f'"{input_manual_rekruter}"'
    else:
        # 2. Fallback ke URL di CV atau Nama Kandidat
        usernames_ditemukan = ekstrak_username_dari_teks(teks_cv)
        if usernames_ditemukan:
            kueri = " OR ".join(usernames_ditemukan)
        else:
            nama_bersih = nama_file.replace(".pdf", "").replace("-", " ").replace("_", " ")
            kueri = f'"{nama_bersih}" (LinkedIn OR GitHub OR Kaggle)'
            
    hasil_teks += f"[HASIL PENELUSURAN OTOMATIS]:\n"
    
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(kueri, max_results=3)]
            if results:
                for r in results:
                    hasil_teks += f"- [{r['title']}] {r['body']}\n"
            else:
                hasil_teks += "Tidak ada data publik relevan yang ditemukan di mesin pencari.\n"
    except Exception as e:
        hasil_teks += f"Pencarian otomatis gagal: {str(e)}\n"
        
    return hasil_teks

def elite_global_ranking(pdf_paths, criteria, api_key, model_name, manual_inputs_dict):
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.1)
    embeddings = load_embeddings()

    all_docs = []
    data_internet_tambahan = "" 
    
    for path in pdf_paths:
        nama_file = os.path.basename(path)
        input_manual = manual_inputs_dict.get(nama_file, "")
        
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            teks_lengkap_cv = " ".join([doc.page_content for doc in docs])
            
            for doc in docs:
                doc.metadata["candidate_name"] = nama_file
            all_docs.extend(docs)
            
            # Analisis jejak digital (menggabungkan Catatan Manual & Pencarian Web)
            jejak = cari_jejak_digital(nama_file, teks_lengkap_cv, input_manual)
            data_internet_tambahan += f"\nData Investigasi untuk {nama_file}:\n{jejak}\n"
            
        except Exception as e:
            st.error(f"Gagal memproses file {nama_file}: {e}")

    if not all_docs:
        return {"error": "Tidak ada dokumen yang berhasil diproses."}

    dynamic_collection_name = f"eval_session_{uuid.uuid4().hex}"
    vectorstore = Chroma.from_documents(documents=all_docs, embedding=embeddings, collection_name=dynamic_collection_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) 
    parser = JsonOutputParser(pydantic_object=RankingOutput)

    template = """
    Tugas: Anda adalah Global Talent Headhunter.
    Tujuan: Menyeleksi kandidat Data Science terbaik dunia.

    KONTEKS DARI CV (PDF):
    {context}

    KONTEKS INVESTIGASI (CATATAN MANUAL & INTERNET):
    {internet_data}

    Kriteria: {question}

    Logika Penilaian:
    1. Global Academic Prestige (40%): Lulusan Top 100 Global dapat nilai maksimal.
    2. Technical Prowess (40%): Fokus Python, SOTA AI Models, MLOps.
    3. Jejak Digital & Budaya (20%): Evaluasi rekam jejak online. BACA DENGAN TELITI 'CATATAN REKRUTER' jika ada. Catatan rekruter mengenai perilaku sosial media kandidat (Instagram, TikTok, dll) adalah sumber kebenaran tertinggi. Evaluasi apakah pola kebiasaan, portofolio, dan etika profesional mereka sesuai standar industri tinggi. Berikan penalti jika ada perilaku tidak profesional.

    TIDAK BOLEH ADA TEKS LAIN SELAIN JSON. PATUHI FORMAT BERIKUT:
    {format_instructions}
    """
    prompt = PromptTemplate(
        template=template, 
        input_variables=["context", "question"], 
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "internet_data": data_internet_tambahan
        }
    )
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
st.markdown("### 1. Unggah Dokumen CV")
uploaded_files = st.file_uploader("Unggah CV Kandidat (PDF) - Anda bisa mengunggah lebih dari 1 file", type="pdf", accept_multiple_files=True)

manual_inputs = {}

# Menampilkan kolom input manual hanya jika ada file yang diunggah
if uploaded_files:
    st.markdown("---")
    st.markdown("### 2. Validasi Media Sosial & Catatan Tambahan (Opsional)")
    st.info("Masukkan link media sosial yang akurat, ATAU tulis catatan manual mengenai kebiasaan/profil kandidat berdasarkan temuan Anda di sosmed agar AI memberikan penilaian karakter yang presisi.")
    
    for file in uploaded_files:
        manual_inputs[file.name] = st.text_area(
            f"Catatan Sosmed/Portofolio untuk kandidat: {file.name}", 
            placeholder="Contoh: IG @budi (Sering posting konten AI, tapi bahasanya kurang profesional) ATAU linkedin.com/in/budi",
            height=68,
            key=file.name
        )
        
    st.markdown("---")
    if st.button("Mulai Analisis Ranking & Profiling", type="primary", use_container_width=True):
        if not api_key:
            st.warning("Masukkan Google API Key di sidebar sebelah kiri terlebih dahulu.")
        else:
            temp_dir = tempfile.mkdtemp()
            saved_paths = []
            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                saved_paths.append(file_path)

            with st.spinner(f"AI sedang menganalisis {len(saved_paths)} CV dan memproses catatan media sosial..."):
                hasil = elite_global_ranking(saved_paths, kriteria, api_key, model_name, manual_inputs)
                
                if "error" not in hasil and "kandidat" in hasil:
                    kandidat_list = hasil["kandidat"]
                    kandidat_list.sort(key=lambda x: x.get('skor_global', 0), reverse=True)
                    for urutan, kand in enumerate(kandidat_list):
                        kand['peringkat'] = urutan + 1

                    st.success("Analisis & Pengecekan Karakter Selesai. Berikut adalah hasilnya:")
                    
                    ringkasan = [{"Peringkat": k['peringkat'], "Kandidat": k['nama_kandidat'], "Skor": k['skor_global'], "Universitas": k['universitas']} for k in kandidat_list]
                    st.table(ringkasan)

                    st.markdown("### Detail Kandidat, Investigasi Karakter & Strategi Wawancara")
                    for kand in kandidat_list:
                        with st.expander(f"Rank {kand['peringkat']} - {kand['nama_kandidat']} (Skor: {kand['skor_global']})"):
                            st.write(f"**Alasan Utama:** {kand['alasan_utama']}")
                            st.info(f"**Hasil Investigasi Profil (Sosmed & Internet):**\n{kand['jejak_digital']}")
                            st.markdown("---")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Universitas:** {kand['universitas']} *(Estimasi QS: {kand['estimasi_rank_dunia']})*")
                                st.write("**Strategi Interview:**")
                                st.success(f"**Kelebihan/Potensi:** {kand['strategi_interview']['alasan_layak']}")
                                st.warning(f"**Celah/Red Flags:** {kand['strategi_interview']['celah_teknis']}")
                                
                            with col2:
                                st.write("**Pertanyaan Lanjutan:**")
                                for q in kand['strategi_interview']['pertanyaan_lanjutan']:
                                    st.write(f"- {q}")
                else:
                    st.error(f"Terjadi kesalahan saat pemrosesan: {hasil.get('error')}")
