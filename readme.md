# CV Summarizer - Multi-Agent Pipeline

A powerful, production-ready system that extracts structured data, analyzes, and generates professional LinkedIn-style summaries from CVs (PDF, DOCX, TXT) using either **local LLMs** or **OpenAI ChatGPT**.

## Features
- Supports **Local LLM** (llama.cpp / GGUF) and **OpenAI API**
- Multi-agent workflow: Extractor → Analyzer → Writer
- Fully configurable via CLI and environment variables
- Clean JSON output with metadata and statistics
- No hard-coded paths

## Setup

### 1. Clone & Install
```bash
git clone <your-repo>
cd cv-summarizer
pip install -r requirements.txt

2. requirements.txt
txtlanggraph
llama-cpp-python
openai
PyMuPDF
python-docx
3. Environment Variables (Recommended)
Local LLM:
envMODEL_PATH=models/granite-4.0-h-micro-Q6_K.gguf
CV_FOLDER=CVs
OUTPUT_FILE=output/cv_summaries.json
OpenAI:
envOPENAI_API_KEY=sk-...
CV_FOLDER=CVs
OUTPUT_FILE=output/cv_summaries.json
