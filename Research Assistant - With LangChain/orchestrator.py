"""
Multi-Agent Orchestrator using LangChain for coordinating the research workflow.
"""
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json

from langchain.agents import AgentExecutor
from memory import SharedMemory
from config import config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    """
    Orchestrates the multi-agent research workflow using LangChain agents.
    """
    
    def __init__(self, research_question: str, output_dir: Optional[Path] = None):
        """Initialize the research orchestrator."""
        self.research_question = research_question
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = config.paths.outputs_dir / f"research_{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Shared memory for all agents
        self.memory = SharedMemory()
        
        # Store research question in memory
        self.memory.set("research_question", research_question)
        self.memory.set("output_dir", str(self.output_dir))
        
        # Initialize agents (will be created lazily)
        self.agents: Dict[str, AgentExecutor] = {}
        
        logger.info(f"Initialized ResearchOrchestrator (LangChain) for: {research_question}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _get_agent(self, agent_type: str) -> AgentExecutor:
        """Get or create a LangChain agent by type."""
        if agent_type in self.agents:
            return self.agents[agent_type]
        
        if agent_type == "literature":
            from agents.literature_agent import create_literature_agent
            agent = create_literature_agent(self.memory)
        else:
            raise ValueError(f"Agent type {agent_type} not yet implemented in LangChain version")
        
        self.agents[agent_type] = agent
        return agent
    
    def run_literature_review(self) -> Dict[str, Any]:
        """Run literature review phase using LangChain agent."""
        logger.info("\n📚 PHASE 1: Literature Review (LangChain)")
        
        agent = self._get_agent("literature")
        
        task = f"""Conduct a literature review for the research question: "{self.research_question}"

Please:
1. Search for relevant papers on arXiv
2. Identify the top 5 most relevant papers
3. Summarize key findings and methodologies
4. Identify baselines and prior work

Provide a comprehensive literature review summary."""
        
        result = agent.invoke({"input": task})
        
        # Save literature review
        lit_review_path = self.output_dir / "literature_review.md"
        with open(lit_review_path, 'w') as f:
            f.write(f"# Literature Review (LangChain Version)\n\n")
            f.write(f"**Research Question:** {self.research_question}\n\n")
            f.write(result.get('output', ''))
        
        logger.info(f"✓ Literature review saved to {lit_review_path}")
        
        return result
    
    def run_full_workflow(self) -> Dict[str, Any]:
        """Run the research workflow (currently only literature review implemented)."""
        logger.info("=" * 80)
        logger.info(f"STARTING RESEARCH WORKFLOW (LangChain): {self.research_question}")
        logger.info("=" * 80)
        
        results = {}
        
        # Phase 1: Literature Review
        lit_result = self.run_literature_review()
        results["literature"] = lit_result
        
        # Save results
        results_path = self.output_dir / "workflow_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                "research_question": self.research_question,
                "output_dir": str(self.output_dir),
                "results": {
                    "literature": {
                        "output": lit_result.get('output', ''),
                        "intermediate_steps": str(lit_result.get('intermediate_steps', []))[:500]
                    }
                },
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"WORKFLOW COMPLETE (LangChain)!")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"{'='*80}\n")
        
        return {
            "success": True,
            "results": results,
            "output_dir": str(self.output_dir)
        }
