"""
Cross-paper comparison and analysis tools.
"""
import logging
from typing import List, Dict
import json

logger = logging.getLogger(__name__)


def compare_papers(papers: List[Dict], aspect: str = "methods") -> str:
    """
    Compare multiple papers on a specific aspect.
    
    Args:
        papers: List of paper dictionaries
        aspect: Aspect to compare ('methods', 'results', 'datasets')
        
    Returns:
        Comparison summary as markdown
    """
    if not papers:
        return "No papers to compare"
    
    comparison = f"# Paper Comparison: {aspect.title()}\n\n"
    
    for i, paper in enumerate(papers, 1):
        title = paper.get('title', f'Paper {i}')
        year = paper.get('year', 'Unknown')
        authors = paper.get('authors', [])
        
        # Format authors
        if isinstance(authors, list) and len(authors) > 0:
            if isinstance(authors[0], dict):
                author_str = authors[0].get('name', 'Unknown')
            else:
                author_str = str(authors[0])
        else:
            author_str = 'Unknown'
        
        comparison += f"## {i}. {title}\n"
        comparison += f"**Authors:** {author_str} et al. ({year})\n\n"
        
        # Add abstract or summary if available
        if 'abstract' in paper and paper['abstract']:
            comparison += f"**Summary:** {paper['abstract'][:200]}...\n\n"
        
        if 'citationCount' in paper:
            comparison += f"**Citations:** {paper['citationCount']}\n\n"
        
        comparison += "---\n\n"
    
    return comparison


def create_comparison_table(papers: List[Dict]) -> str:
    """
    Generate a markdown comparison table for papers.
    
    Args:
        papers: List of paper dictionaries
        
    Returns:
        Markdown table
    """
    if not papers:
        return "No papers to compare"
    
    table = "| Title | Year | Citations | Venue |\n"
    table += "|-------|------|-----------|-------|\n"
    
    for paper in papers:
        title = paper.get('title', 'Unknown')[:50] + "..."
        year = paper.get('year', 'N/A')
        citations = paper.get('citationCount', 'N/A')
        venue = paper.get('venue', 'N/A')
        
        table += f"| {title} | {year} | {citations} | {venue} |\n"
    
    return table


def identify_trends(papers: List[Dict]) -> str:
    """
    Identify trends across multiple papers.
    
    Args:
        papers: List of paper dictionaries
        
    Returns:
        Trend analysis summary
    """
    if not papers:
        return "No papers to analyze"
    
    # Sort by year
    sorted_papers = sorted(papers, key=lambda x: x.get('year', 0))
    
    trends = "# Research Trends Analysis\n\n"
    
    # Year distribution
    years = [p.get('year') for p in papers if p.get('year')]
    if years:
        trends += f"## Temporal Distribution\n"
        trends += f"- Earliest: {min(years)}\n"
        trends += f"- Latest: {max(years)}\n"
        trends += f"- Total papers: {len(papers)}\n\n"
    
    # Citation analysis
    citations = [p.get('citationCount', 0) for p in papers]
    if citations:
        trends += f"## Citation Analysis\n"
        trends += f"- Average citations: {sum(citations) / len(citations):.1f}\n"
        trends += f"- Most cited: {max(citations)}\n"
        trends += f"- Least cited: {min(citations)}\n\n"
    
    # Venue distribution
    venues = {}
    for paper in papers:
        venue = paper.get('venue', 'Unknown')
        venues[venue] = venues.get(venue, 0) + 1
    
    if venues:
        trends += f"## Top Venues\n"
        for venue, count in sorted(venues.items(), key=lambda x: x[1], reverse=True)[:5]:
            trends += f"- {venue}: {count} papers\n"
    
    return trends


def synthesize_findings(papers: List[Dict], summaries: List[str] = None) -> str:
    """
    Synthesize findings from multiple papers.
    
    Args:
        papers: List of paper dictionaries
        summaries: Optional list of paper summaries
        
    Returns:
        Synthesis summary
    """
    synthesis = "# Multi-Paper Synthesis\n\n"
    synthesis += f"**Based on {len(papers)} papers**\n\n"
    
    synthesis += "## Key Papers\n\n"
    
    # Sort by citation count
    sorted_papers = sorted(papers, key=lambda x: x.get('citationCount', 0), reverse=True)
    
    for i, paper in enumerate(sorted_papers[:5], 1):
        title = paper.get('title', 'Unknown')
        year = paper.get('year', 'N/A')
        citations = paper.get('citationCount', 'N/A')
        
        synthesis += f"{i}. **{title}** ({year}) - {citations} citations\n"
        
        if summaries and i-1 < len(summaries):
            synthesis += f"   - {summaries[i-1][:150]}...\n"
        
        synthesis += "\n"
    
    return synthesis
