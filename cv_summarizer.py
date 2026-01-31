import os
import time
import subprocess
import json
import argparse
from pathlib import Path
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from abc import ABC, abstractmethod

# --- LLM Provider Abstraction ---
class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        pass

    def reset(self):
        pass  # Only needed for local LLM


class LocalLLM(BaseLLM):
    def __init__(self, model_path: str):
        from llama_cpp import Llama
        print(f"Loading local model: {model_path}")
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_gpu_layers=35,
                n_batch=512,
                verbose=False,
                use_mmap=True,
                use_mlock=False,
                n_threads=6,
                n_threads_batch=6,
            )
            print("Local model loaded successfully!")
        except Exception as e:
            print(f"GPU offload failed: {e}")
            print("Falling back to CPU mode...")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_gpu_layers=0,
                n_batch=512,
                verbose=False,
            )

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        self.reset()
        response = self.llm(prompt, max_tokens=max_tokens, temperature=temperature, stop=["\n\n", "Rules:", "CV TEXT:"])
        text = response["choices"][0]["text"].strip() if isinstance(response, dict) else str(response).strip()
        return text

    def reset(self):
        self.llm.reset()


class OpenAILLM(BaseLLM):
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        print(f"OpenAI client initialized ({model})")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()


# ========================
# Configuration
# ========================
def get_config():
    parser = argparse.ArgumentParser(description="Multi-Agent CV Summarizer")
    parser.add_argument("--provider", type=str, choices=["local", "openai"], default="local",
                        help="LLM provider: 'local' or 'openai' (default: local)")
    parser.add_argument("--model-path", type=str, help="Path to GGUF model (for local provider)")
    parser.add_argument("--cv-folder", type=str, default=os.getenv("CV_FOLDER", "CVs"),
                        help="Folder containing CV files (default: CVs)")
    parser.add_argument("--output", type=str, default=os.getenv("OUTPUT_FILE", "output/cv_summaries.json"),
                        help="Output JSON file path (default: output/cv_summaries.json)")
    parser.add_argument("--limit", type=int, help="Limit number of CVs to process")
    parser.add_argument("--openai-model", type=str, default="gpt-4o-mini", help="OpenAI model name (default: gpt-4o-mini)")

    args = parser.parse_args()

    config = {
        "provider": args.provider,
        "model_path": args.model_path or os.getenv("MODEL_PATH"),
        "cv_folder": args.cv_folder,
        "output_file": args.output,
        "limit": args.limit,
        "openai_model": args.openai_model,
    }

    if config["provider"] == "local" and not config["model_path"]:
        raise ValueError("MODEL_PATH environment variable or --model-path argument is required for local provider")
    if config["provider"] == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider")

    return config


def log_gpu_stats():
    """Log current GPU usage (only for local)"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            util, used, total = result.stdout.strip().split(', ')
            print(f"GPU: {util}% utilization, {used}/{total} MiB VRAM")
        else:
            print("GPU: nvidia-smi failed")
    except Exception:
        print("GPU: Not available (CPU mode)")


class CVSummaryState(TypedDict):
    cv_text: str
    cv_filename: str
    extracted_info: dict          # Structured data from Extractor
    analysis: dict                # Insights from Analyzer
    professional_summary: str
    processing_time: float
    total_processing_time: float  # Added at the end


# ========================
# Agent 1: Extractor
# ========================
def extract_cv_info(state: CVSummaryState, llm: BaseLLM):
    """Extract structured information from CV"""
    start_time = time.time()
    
    print(f"  [Extractor] Processing {state['cv_filename']}...")

    prompt = f"""You are an expert CV parser. Extract information from the CV below and return ONLY a valid JSON object with exactly these keys:

{{
  "skills": string array of main technical and soft skills (max 10, most important only),
  "years_experience": integer number of years of professional experience (estimate if not explicit),
  "education_level": string with highest degree and field (e.g., "Bachelor's in Computer Science"),
  "key_achievements": string array of 2-4 most impressive achievements,
  "current_role": string with most recent or current job title
}}

Rules:
- Do not add any extra text, explanations, or markdown.
- Return ONLY the JSON object.
- If information is missing, use null or empty array appropriately.
- Be accurate — do not hallucinate.

CV TEXT (first 2500 characters):
{state["cv_text"][:2500]}

Return ONLY the JSON:
"""

    try:
        text = llm.generate(prompt, max_tokens=300, temperature=0.1)
        
        # Clean up response
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        extracted_data = json.loads(text)
        print(f"  [Extractor] Extraction successful ({time.time() - start_time:.1f}s)")
        
    except json.JSONDecodeError as e:
        print(f"  [Extractor] JSON parsing failed: {e}")
        print(f"  Raw output: {text[:200]}...")
        extracted_data = {
            "skills": ["Data extraction failed"],
            "years_experience": 0,
            "education_level": "Unknown",
            "key_achievements": ["Parsing error occurred"],
            "current_role": "Unknown"
        }
    except Exception as e:
        print(f"  [Extractor] Inference failed: {e}")
        extracted_data = {
            "skills": ["Error during extraction"],
            "years_experience": 0,
            "education_level": "Error",
            "key_achievements": ["System error"],
            "current_role": "Error"
        }

    return {
        "extracted_info": extracted_data,
        "processing_time": time.time() - start_time
    }


# ========================
# Agent 2: Analyzer
# ========================
def analyze_cv(state: CVSummaryState, llm: BaseLLM):
    """Analyze extracted CV data for insights"""
    start_time = time.time()
    
    print(f"  [Analyzer] Analyzing {state['cv_filename']}...")

    extracted = state["extracted_info"]
    
    # Create clean summary for analysis
    extracted_summary = f"""
Skills: {", ".join(extracted.get("skills", [])[:8])}
Years of Experience: {extracted.get("years_experience", 0)}
Education: {extracted.get("education_level") or "Not specified"}
Current/Most Recent Role: {extracted.get("current_role") or "Not specified"}
Key Achievements:
{chr(10).join(f"- {ach}" for ach in extracted.get("key_achievements", [])[:3])}
"""

    prompt = f"""You are a senior recruiter with 15+ years of experience evaluating candidates.

Based on the extracted CV data below, provide deep analysis.

Return ONLY a valid JSON object with exactly these keys:

{{
  "seniority_level": "Junior" | "Mid-level" | "Senior" | "Lead/Principal" | "Executive",
  "primary_domain": string describing main expertise area (e.g., "Full-Stack Web Development", "DevOps & Cloud"),
  "strengths": array of 3-5 bullet-point style strengths (as strings),
  "standout_factor": single most impressive thing about this candidate (one sentence),
  "potential_concerns": array of concerns (or empty array if none),
  "suggested_angle": one-sentence suggestion for how to position this candidate in a professional summary
}}

Rules:
- Base everything strictly on the provided data.
- Do not hallucinate.
- Be insightful but fair.
- Return ONLY the JSON.

EXTRACTED CV DATA:
{extracted_summary}

Return ONLY the JSON:
"""

    try:
        text = llm.generate(prompt, max_tokens=400, temperature=0.2)
        
        # Clean up response
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        analysis_data = json.loads(text)
        print(f"  [Analyzer] Analysis successful ({time.time() - start_time:.1f}s)")
        
    except json.JSONDecodeError as e:
        print(f"  [Analyzer] JSON parsing failed: {e}")
        print(f"  Raw output: {text[:200]}...")
        analysis_data = {
            "seniority_level": "Mid-level",
            "primary_domain": "General Professional",
            "strengths": ["Solid professional experience", "Reliable work history"],
            "standout_factor": "Consistent career progression",
            "potential_concerns": [],
            "suggested_angle": "Highlight well-rounded skills and experience"
        }
    except Exception as e:
        print(f"  [Analyzer] Inference failed: {e}")
        analysis_data = {
            "seniority_level": "Unknown",
            "primary_domain": "Error in analysis",
            "strengths": ["Analysis failed"],
            "standout_factor": "Unable to analyze",
            "potential_concerns": ["System error during analysis"],
            "suggested_angle": "Manual review required"
        }

    return {
        "analysis": analysis_data,
        "processing_time": time.time() - start_time
    }


# ========================
# Agent 3: Writer
# ========================
def write_professional_summary(state: CVSummaryState, llm: BaseLLM):
    """Generate professional summary based on extracted data and analysis"""
    start_time = time.time()
    
    print(f"  [Writer] Writing summary for {state['cv_filename']}...")

    extracted = state["extracted_info"]
    analysis = state["analysis"]

    # Create comprehensive brief for writing
    brief = f"""CANDIDATE PROFILE BRIEF

Role: {extracted.get("current_role", "Professional")}
Seniority: {analysis.get("seniority_level", "Mid-level")}
Primary Domain: {analysis.get("primary_domain", "General")}

Key Skills: {", ".join(extracted.get("skills", [])[:8])}
Years of Experience: {extracted.get("years_experience", 0)}
Education: {extracted.get("education_level") or "Not specified"}

Top Achievements:
{chr(10).join(f"- {ach}" for ach in extracted.get("key_achievements", [])[:3])}

Strengths:
{chr(10).join(f"- {strength}" for strength in analysis.get("strengths", [])[:4])}

Standout Factor: {analysis.get("standout_factor", "Solid contributor")}
Suggested Positioning: {analysis.get("suggested_angle", "Highlight experience and skills")}
"""

    prompt = f"""You are an expert executive resume writer and recruiter ghostwriter.

Write a compelling, professional third-person summary (exactly 95-105 words) for this candidate's LinkedIn profile or job application.

Requirements:
- Start with strong opening: [Seniority] [Domain Expert] with [X] years...
- Highlight technical skills, leadership, and impact naturally.
- Weave in 2-3 key achievements with results.
- Mention education briefly if relevant.
- End with value proposition (what they bring to employers).
- Use confident, professional tone — no clichés like "passionate" or "hardworking".
- Do not hallucinate or add unlisted skills/tools.
- Write in third person (he/she/they).

Use ONLY the information in the brief below.

BRIEF:
{brief}

Write the summary now (95-105 words). Start directly — no labels or quotes:

SUMMARY:"""

    try:
        text = llm.generate(prompt, max_tokens=200, temperature=0.4)
        
        # Clean up response
        if text.startswith("SUMMARY:"):
            text = text[len("SUMMARY:"):].strip()
        
        # Fallback if generation failed
        if not text or len(text.split()) < 50:
            text = f"{analysis.get('seniority_level', 'Experienced')} professional with {extracted.get('years_experience', 'several')} years in {analysis.get('primary_domain', 'their field')}. Skilled in {', '.join(extracted.get('skills', ['relevant technologies'])[:5])}. Known for delivering impactful results and driving business growth through technical expertise and strategic thinking."

        print(f"  [Writer] Summary generated ({time.time() - start_time:.1f}s)")
        
    except Exception as e:
        print(f"  [Writer] Summary generation failed: {e}")
        text = f"Professional with experience in {analysis.get('primary_domain', 'various domains')}. Skilled in {', '.join(extracted.get('skills', ['multiple areas'])[:3])}. Seeking to contribute expertise and drive results in new opportunities."

    return {
        "professional_summary": text,
        "processing_time": time.time() - start_time
    }


# ========================
# CV Reading Utilities
# ========================
def read_cv_file(filepath: Path) -> str | None:
    """Read CV file content from various formats"""
    try:
        if filepath.suffix.lower() == '.txt':
            return filepath.read_text(encoding='utf-8', errors='ignore')
        
        elif filepath.suffix.lower() == '.pdf':
            import fitz  # PyMuPDF
            doc = fitz.open(filepath)
            text = "\n".join(page.get_text("text") for page in doc)
            doc.close()
            return text if text.strip() else None
        
        elif filepath.suffix.lower() == '.docx':
            from docx import Document
            doc = Document(str(filepath))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        
        else:
            print(f"  Unsupported format: {filepath.name}")
            return None
    except Exception as e:
        print(f"  Error reading {filepath.name}: {e}")
        return None


# ========================
# Workflow Definition
# ========================
def build_workflow(llm: BaseLLM):
    workflow = StateGraph(CVSummaryState)

    workflow.add_node("extract", lambda state: extract_cv_info(state, llm))
    workflow.add_node("analyze", lambda state: analyze_cv(state, llm))
    workflow.add_node("summarize", lambda state: write_professional_summary(state, llm))

    workflow.add_edge("extract", "analyze")
    workflow.add_edge("analyze", "summarize")

    workflow.set_entry_point("extract")
    workflow.set_finish_point("summarize")

    return workflow.compile()


# ========================
# Processing Functions
# ========================
def process_single_cv(filepath: Path, app, config):
    """Process a single CV through the multi-agent pipeline"""
    print(f"\nProcessing: {filepath.name}")
    
    cv_text = read_cv_file(filepath)
    if not cv_text or len(cv_text.strip()) < 100:
        print(f"  Skipped (empty or too short): {filepath.name}")
        return None
    
    if config["provider"] == "local":
        log_gpu_stats()
    
    start_time = time.time()
    try:
        result = app.invoke({
            "cv_text": cv_text,
            "cv_filename": filepath.name,
            "extracted_info": {},
            "analysis": {},
            "professional_summary": "",
            "processing_time": 0,
            "total_processing_time": 0
        })
        
        result["total_processing_time"] = time.time() - start_time
        print(f"SUCCESS: {result['cv_filename']} - {result['total_processing_time']:.1f}s")
        print(f"Summary: {result['professional_summary'][:120]}...")
        return result
        
    except Exception as e:
        print(f"FAILED: {filepath.name} - Error: {e}")
        return None


def batch_process_cvs(config, llm: BaseLLM):
    """Process multiple CVs in batch"""
    cv_folder = Path(config["cv_folder"])
    if not cv_folder.exists():
        print(f"Error: CV folder not found: {config['cv_folder']}")
        return
    
    # Find all CV files
    cv_files = []
    for ext in ['*.txt', '*.pdf', '*.docx']:
        cv_files.extend(cv_folder.glob(ext))
    
    print(f"Found {len(cv_files)} CV files.")
    if config["limit"]:
        cv_files = cv_files[:config["limit"]]
        print(f"Limited to {config['limit']} files.")
    
    results = []
    failed_count = 0
    total_start = time.time()
    
    print(f"\n{'='*70}")
    print("STARTING BATCH PROCESSING")
    print(f"{'='*70}")
    
    if config["provider"] == "local":
        log_gpu_stats()
    
    app = build_workflow(llm)
    
    for i, cv_file in enumerate(cv_files, 1):
        print(f"\n[{i}/{len(cv_files)}] ", end="")
        result = process_single_cv(cv_file, app, config)
        
        if result:
            results.append(result)
            print("─" * 70)
        else:
            failed_count += 1
    
    total_time = time.time() - total_start
    
    # Prepare output data
    output_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "provider": config["provider"],
            "model": config.get("openai_model") or Path(config.get("model_path", "")).name,
            "workflow": "Multi-Agent (Extractor → Analyzer → Writer)",
            "environment": "Docker" if os.path.exists("/.dockerenv") else "Local"
        },
        "statistics": {
            "total_files_found": len(cv_files),
            "processed_successfully": len(results),
            "failed_or_skipped": failed_count,
            "success_rate_percent": round((len(results) / len(cv_files)) * 100, 1) if cv_files else 0,
            "total_time_seconds": round(total_time, 1),
            "total_time_minutes": round(total_time / 60, 1),
            "average_time_per_cv_seconds": round(total_time / len(results), 1) if results else 0,
        },
        "results": results
    }
    
    # Save results
    Path(config["output_file"]).parent.mkdir(parents=True, exist_ok=True)
    with open(config["output_file"], "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Display final summary
    print(f"\n{'='*70}")
    print("BATCH PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"Processed Successfully: {len(results)} CVs")
    print(f"Failed/Skipped: {failed_count} CVs")
    print(f"Success Rate: {output_data['statistics']['success_rate_percent']}%")
    print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Average per CV: {output_data['statistics']['average_time_per_cv_seconds']}s")
    print(f"Results saved to: {config['output_file']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    config = get_config()
    
    print("=" * 80)
    print("CV SUMMARIZER - Multi-Agent Pipeline (Docker Compatible)")
    print("=" * 80)
    print(f"Provider: {config['provider'].upper()}")
    if config["provider"] == "local":
        print(f"Model: {Path(config['model_path']).name}")
    else:
        print(f"Model: {config['openai_model']}")
    print(f"Input Folder: {config['cv_folder']}")
    print(f"Output File: {config['output_file']}")
    print(f"Pipeline: Extractor → Analyzer → Writer")
    print(f"Environment: {'Docker' if os.path.exists('/.dockerenv') else 'Local'}")
    print("=" * 80)
    
    # Load LLM
    if config["provider"] == "local":
        llm = LocalLLM(config["model_path"])
    else:
        llm = OpenAILLM(config["openai_model"])
    
    # Run batch processing
    batch_process_cvs(config, llm)
