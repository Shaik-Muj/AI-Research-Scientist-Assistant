"""
Research Orchestrator - Simplified single-agent workflow.
"""
import logging
from typing import Any, Dict
from datetime import datetime
from pathlib import Path
import json
import hashlib

from langchain.agents import AgentExecutor
from memory import SharedMemory
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    """Orchestrates research workflow using a single intelligent agent."""
    
    def __init__(self, query: str, output_dir: Path = None):
        """
        Initialize the research orchestrator.
        
        Args:
            query: Research query or question
            output_dir: Directory to save outputs
        """
        self.query = query
        self.memory = SharedMemory()
        
        # Set up output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("outputs") / f"research_{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store query in memory
        self.memory.set("research_query", query)
        self.memory.set("output_dir", str(self.output_dir))
        
        logger.info(f"Initialized ResearchOrchestrator for: {query}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def run(self) -> Dict[str, Any]:
        """
        Execute research query using the research agent.
        
        Returns:
            Dictionary containing the research results
        """
        logger.info("=" * 80)
        logger.info(f"RESEARCH QUERY: {self.query}")
        logger.info("=" * 80)
        
        # Check cache
        cache_key = hashlib.md5(self.query.encode()).hexdigest()
        cached_result = self.memory.get(f"research_{cache_key}")
        
        if cached_result:
            logger.info("✅ Found cached result! Using cached response.")
            self._save_result(cached_result)
            return cached_result
        
        logger.info("No cache found. Running research agent...")
        
        # Create and run research agent
        from agents.research_agent import create_research_agent
        
        # Use the first available model
        model = config.llm.available_models[0]
        logger.info(f"Using model: {model}")
        
        agent = create_research_agent(self.memory, model=model)
        
        # Execute query
        result = agent.invoke({"input": self.query})
        
        # Cache result
        self.memory.set(f"research_{cache_key}", result)
        logger.info("✓ Result cached for future use")
        
        # Save result
        self._save_result(result)
        
        logger.info("=" * 80)
        logger.info("RESEARCH COMPLETED")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("=" * 80)
        
        return result
    
    def _save_result(self, result: Dict[str, Any]):
        """Save research result to file."""
        # Save as markdown
        result_path = self.output_dir / "research_result.md"
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(f"# Research Result\n\n")
            f.write(f"**Query:** {self.query}\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            f.write(result.get('output', ''))
        
        # Save as JSON (only serializable parts)
        json_path = self.output_dir / "research_result.json"
        serializable_result = {
            'query': self.query,
            'output': result.get('output', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2)
        
        logger.info(f"✓ Results saved to {result_path}")
