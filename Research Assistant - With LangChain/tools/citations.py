"""
Citation management tools for extracting and formatting citations.
"""
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def extract_bibtex(paper: Dict) -> str:
    """
    Generate BibTeX citation from paper metadata.
    
    Args:
        paper: Paper metadata dictionary
        
    Returns:
        BibTeX formatted citation
    """
    # Extract fields
    title = paper.get('title', 'Unknown Title')
    authors = paper.get('authors', [])
    year = paper.get('year', paper.get('published', datetime.now().year))
    
    # Handle different metadata formats
    if isinstance(authors, list):
        if len(authors) > 0 and isinstance(authors[0], dict):
            author_names = [a.get('name', '') for a in authors]
        else:
            author_names = authors
    else:
        author_names = [str(authors)]
    
    authors_str = ' and '.join(author_names)
    
    # Generate citation key
    first_author = author_names[0].split()[-1] if author_names else 'Unknown'
    cite_key = f"{first_author}{year}"
    
    # Get additional fields
    journal = paper.get('venue', paper.get('journal', ''))
    arxiv_id = paper.get('arxiv_id', '')
    url = paper.get('url', '')
    
    # Build BibTeX
    bibtex = f"@article{{{cite_key},\n"
    bibtex += f"  title={{{title}}},\n"
    bibtex += f"  author={{{authors_str}}},\n"
    bibtex += f"  year={{{year}}},\n"
    
    if journal:
        bibtex += f"  journal={{{journal}}},\n"
    if arxiv_id:
        bibtex += f"  eprint={{{arxiv_id}}},\n"
        bibtex += f"  archivePrefix={{arXiv}},\n"
    if url:
        bibtex += f"  url={{{url}}},\n"
    
    bibtex += "}"
    
    return bibtex


def extract_apa(paper: Dict) -> str:
    """
    Generate APA citation from paper metadata.
    
    Args:
        paper: Paper metadata dictionary
        
    Returns:
        APA formatted citation
    """
    # Extract fields
    title = paper.get('title', 'Unknown Title')
    authors = paper.get('authors', [])
    year = paper.get('year', paper.get('published', datetime.now().year))
    
    # Handle different metadata formats
    if isinstance(authors, list):
        if len(authors) > 0 and isinstance(authors[0], dict):
            author_names = [a.get('name', '') for a in authors]
        else:
            author_names = authors
    else:
        author_names = [str(authors)]
    
    # Format authors (APA style)
    if len(author_names) == 1:
        authors_str = author_names[0]
    elif len(author_names) == 2:
        authors_str = f"{author_names[0]} & {author_names[1]}"
    elif len(author_names) > 2:
        authors_str = f"{author_names[0]}, et al."
    else:
        authors_str = "Unknown"
    
    # Get additional fields
    journal = paper.get('venue', paper.get('journal', 'arXiv preprint'))
    url = paper.get('url', '')
    
    # Build APA citation
    apa = f"{authors_str} ({year}). {title}. "
    
    if journal:
        apa += f"*{journal}*. "
    
    if url:
        apa += f"Retrieved from {url}"
    
    return apa.strip()


def format_citation(paper: Dict, format_type: str = "bibtex") -> str:
    """
    Format citation in specified style.
    
    Args:
        paper: Paper metadata
        format_type: Citation format ('bibtex' or 'apa')
        
    Returns:
        Formatted citation
    """
    if format_type.lower() == "bibtex":
        return extract_bibtex(paper)
    elif format_type.lower() == "apa":
        return extract_apa(paper)
    else:
        return f"Unsupported format: {format_type}"
