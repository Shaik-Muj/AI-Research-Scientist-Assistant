"""
Semantic Scholar API integration for academic paper search.
"""
import requests
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"


def search_semantic_scholar(query: str, limit: int = 10, year_filter: str = None) -> List[Dict]:
    """
    Search Semantic Scholar for papers.
    
    Args:
        query: Search query
        limit: Maximum number of results
        year_filter: Year filter (e.g., "2020-" for 2020 onwards)
        
    Returns:
        List of paper dictionaries
    """
    logger.info(f"Searching Semantic Scholar for: {query}")
    
    try:
        url = f"{SEMANTIC_SCHOLAR_API}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "paperId,title,abstract,authors,year,citationCount,influentialCitationCount,url,venue"
        }
        
        if year_filter:
            params["year"] = year_filter
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        papers = data.get("data", [])
        
        logger.info(f"Found {len(papers)} papers on Semantic Scholar")
        return papers
        
    except Exception as e:
        logger.error(f"Error searching Semantic Scholar: {str(e)}")
        return []


def get_paper_details(paper_id: str) -> Optional[Dict]:
    """
    Get detailed information about a specific paper.
    
    Args:
        paper_id: Semantic Scholar paper ID
        
    Returns:
        Paper details dictionary
    """
    try:
        url = f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}"
        params = {
            "fields": "paperId,title,abstract,authors,year,citationCount,influentialCitationCount,references,citations,url,venue,publicationTypes"
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        return response.json()
        
    except Exception as e:
        logger.error(f"Error getting paper details: {str(e)}")
        return None


def get_paper_citations(paper_id: str, limit: int = 10) -> List[Dict]:
    """
    Get papers that cite this paper.
    
    Args:
        paper_id: Semantic Scholar paper ID
        limit: Maximum number of citations
        
    Returns:
        List of citing papers
    """
    try:
        url = f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}/citations"
        params = {
            "limit": limit,
            "fields": "paperId,title,year,citationCount"
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        return data.get("data", [])
        
    except Exception as e:
        logger.error(f"Error getting citations: {str(e)}")
        return []


def find_similar_papers(paper_id: str, limit: int = 5) -> List[Dict]:
    """
    Find papers similar to the given paper.
    
    Args:
        paper_id: Semantic Scholar paper ID
        limit: Maximum number of similar papers
        
    Returns:
        List of similar papers
    """
    try:
        url = f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}/recommendations"
        params = {
            "limit": limit,
            "fields": "paperId,title,abstract,year,citationCount"
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        return data.get("recommendedPapers", [])
        
    except Exception as e:
        logger.error(f"Error finding similar papers: {str(e)}")
        return []
