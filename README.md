# Ranking Selection System (AI & OSINT)

**An Intelligent Recruitment Pipeline for World-Class Data Scientist Selection**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](https://your-app-link.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/Powered%20by-LangChain-brightgreen)](https://python.langchain.com/)

## 🚀 Overview
The **Ranking Selection System** is an advanced AI-driven recruitment tool designed to identify, evaluate, and rank elite Data Scientists. Unlike traditional ATS (Applicant Tracking Systems), this tool combines **Retrieval-Augmented Generation (RAG)** for deep CV analysis and **Open-Source Intelligence (OSINT)** to validate a candidate's digital footprint across GitHub, LinkedIn, and Kaggle.

This project was built to eliminate human bias and significantly reduce the time required to perform comprehensive technical and background evaluations.

---

## 🛠️ Key Features

### 1. Automated RAG Pipeline
Utilizes **ChromaDB** and **HuggingFace Embeddings** to ingest and query multiple PDF resumes. It performs context-aware analysis to find technical skills that go beyond simple keyword matching.

### 2. OSINT Investigation Engine
Integrated with **DuckDuckGo Search**, the system automatically:
- Extracts social handles (LinkedIn, GitHub, Kaggle) from the CV.
- Performs real-time web searches to gather public data regarding the candidate's professional behavior and contributions.
- Provides a "Digital Footprint Analysis" to detect red flags or unprofessional conduct.

### 3. Structured Global Ranking
Uses **Google Gemini API** (Flash/Pro) to generate an objective "Global Score" (0-100) based on:
- **Academic Prestige (40%):** University ranking and global reputation.
- **Technical Prowess (40%):** Proficiency in Python, Deep Learning, and MLOps.
- **Cultural/Social Fit (20%):** Professionalism and ethics based on OSINT data.

### 4. Dynamic Interview Strategy
For every candidate, the system generates:
- **Strengths & Technical Gaps:** Clear pros and cons.
- **Bespoke Interview Questions:** Three high-level technical questions tailored specifically to the candidate's experience and detected gaps.

---

## Tech Stack
- **Framework:** Streamlit (UI/UX)
- **Orchestration:** LangChain
- **LLM:** Google Gemini API (`gemini-1.5-flash`, `gemini-1.5-pro`)
- **Vector Store:** ChromaDB
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
- **OSINT Tool:** DuckDuckGo Search (DDGS)
- **Data Schema:** Pydantic (Structured JSON Output)

---

## Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/yourusername/ranking-selection-system.git](https://github.com/yourusername/ranking-selection-system.git)
   cd ranking-selection-system

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py

6. **Usage:**
    - 
