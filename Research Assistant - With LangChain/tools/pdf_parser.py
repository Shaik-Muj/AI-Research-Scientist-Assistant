"""
PDF parsing and summarization tools.
"""
import logging
from pathlib import Path
from typing import Optional
import PyPDF2
import google.generativeai as genai
from config import config
from tools.arxiv_search import download_paper


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text or None if failed
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text
            
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return None


def summarize_paper(pdf_input: str, focus: str = "methodology and key findings") -> str:
    """
    Summarize a research paper using LLM.
    
    Args:
        pdf_input: Either an arXiv ID (e.g., "2301.12345") OR a path to a PDF file
        focus: What to focus on in the summary
        
    Returns:
        Summary text
    """
    logger.info(f"Summarizing paper: {pdf_input}")
    
    # Check if input is an arXiv ID or a file path
    if not pdf_input.endswith('.pdf') and not Path(pdf_input).exists():
        # It's likely an arXiv ID - download it first
        logger.info(f"Detected arXiv ID: {pdf_input}, downloading...")
        pdf_path = download_paper(pdf_input)
        if not pdf_path:
            return f"Failed to download paper {pdf_input}"
    else:
        # It's already a file path
        pdf_path = pdf_input
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        return "Failed to extract text from PDF"
    
    # Truncate if too long (LLM has token limits)
    max_chars = 30000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... truncated ...]"
    
    # Use Groq to summarize
    from langchain_groq import ChatGroq
    
    llm = ChatGroq(
        model=config.llm.model,
        api_key=config.llm.api_key,
        temperature=config.llm.temperature
    )
    
    prompt = f"""Summarize this research paper, focusing on {focus}.

Paper content:
{text}

Provide a structured summary including:
1. Main contribution/thesis
2. Methodology
3. Key findings/results
4. Limitations (if mentioned)
5. Relevance to ML/AI research

Keep the summary concise (2-3 paragraphs)."""
    
    try:
        response = llm.invoke(prompt)
        summary = response.content
        logger.info("Paper summarized successfully")
        return summary
        
    except Exception as e:
        logger.error(f"Error summarizing paper: {str(e)}")
        return f"Error generating summary: {str(e)}"


def extract_methodology(pdf_path: str) -> str:
    """
    Extract methodology section from a paper.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Methodology description
    """
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        return "Failed to extract text"
    
    # Use LLM to extract methodology
    genai.configure(api_key=config.llm.api_key)
    model = genai.GenerativeModel(config.llm.model)
    
    prompt = f"""Extract and summarize the methodology from this research paper.

Paper content:
{text[:20000]}

Focus on:
- Experimental setup
- Model architectures used
- Training procedures
- Datasets
- Evaluation metrics

Provide a clear, concise summary of the methodology."""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error extracting methodology: {str(e)}"
