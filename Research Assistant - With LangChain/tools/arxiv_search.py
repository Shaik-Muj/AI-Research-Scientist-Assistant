"""
arXiv API integration for searching and fetching research papers.
"""
import arxiv
import logging
from pathlib import Path
from typing import List, Dict, Optional
from config import config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def search_arxiv(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search arXiv for papers matching the query.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of paper dictionaries with metadata
    """
    logger.info(f"Searching arXiv for: {query}")
    
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in search.results():
            paper = {
                "arxiv_id": result.entry_id.split('/')[-1],
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "abstract": result.summary,
                "published": result.published.isoformat(),
                "url": result.entry_id,
                "pdf_url": result.pdf_url,
                "categories": result.categories
            }
            papers.append(paper)
        
        logger.info(f"Found {len(papers)} papers")
        return papers
        
    except Exception as e:
        logger.error(f"Error searching arXiv: {str(e)}")
        return []


def download_paper(arxiv_id: str, download_dir: Optional[Path] = None) -> Optional[str]:
    """
    Download a paper PDF from arXiv.
    
    Args:
        arxiv_id: arXiv ID (e.g., "2301.12345")
        download_dir: Directory to save PDF (defaults to config.paths.papers_dir)
        
    Returns:
        Path to downloaded PDF or None if failed
    """
    if download_dir is None:
        download_dir = config.paths.papers_dir
    
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading paper: {arxiv_id}")
    
    try:
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        
        # Download PDF
        pdf_path = download_dir / f"{arxiv_id.replace('/', '_')}.pdf"
        paper.download_pdf(filename=str(pdf_path))
        
        logger.info(f"Downloaded to: {pdf_path}")
        return str(pdf_path)
        
    except Exception as e:
        logger.error(f"Error downloading paper {arxiv_id}: {str(e)}")
        return None


def get_paper_metadata(arxiv_id: str) -> Optional[Dict]:
    """
    Get metadata for a specific arXiv paper.
    
    Args:
        arxiv_id: arXiv ID
        
    Returns:
        Paper metadata dictionary or None if not found
    """
    try:
        search = arxiv.Search(id_list=[arxiv_id])
        result = next(search.results())
        
        return {
            "arxiv_id": arxiv_id,
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "abstract": result.summary,
            "published": result.published.isoformat(),
            "url": result.entry_id,
            "pdf_url": result.pdf_url,
            "categories": result.categories
        }
        
    except Exception as e:
        logger.error(f"Error fetching metadata for {arxiv_id}: {str(e)}")
        return None
