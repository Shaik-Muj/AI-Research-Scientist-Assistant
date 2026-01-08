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
        elif agent_type == "experiment":
            from agents.experiment_agent import create_experiment_agent
            agent = create_experiment_agent(self.memory)
        elif agent_type == "analysis":
            from agents.analysis_agent import create_analysis_agent
            agent = create_analysis_agent(self.memory)
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
    
    def run_experiment_design(self) -> Dict[str, Any]:
        """Run experiment design phase using LangChain agent."""
        logger.info("\n🔬 PHASE 2: Experiment Design (LangChain)")
        
        # Get the experiment agent
        agent = self._get_agent("experiment")
        
        # Create the task
        task = f"""Based on the literature review for "{self.research_question}", design and run experiments.

Please:
1. Design appropriate experiments to test the research question
2. Run the experiments using available models and datasets
3. Store the results for analysis

Focus on comparing different approaches mentioned in the literature."""
        
        # Run the agent
        result = agent.invoke({"input": task})
        
        # Save experiment report
        exp_report_path = self.output_dir / "experiment_report.md"
        with open(exp_report_path, 'w') as f:
            f.write(f"# Experiment Report (LangChain Version)\n\n")
            f.write(f"**Research Question:** {self.research_question}\n\n")
            f.write(result.get('output', ''))
        
        logger.info(f"✓ Experiment report saved to {exp_report_path}")
        
        return result
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run analysis phase using LangChain agent."""
        logger.info("\n📊 PHASE 3: Analysis (LangChain)")
        
        # Get the analysis agent
        agent = self._get_agent("analysis")
        
        # Create the task
        task = f"""Analyze the experimental results for "{self.research_question}".

Please:
1. Search for experiment results in memory
2. Perform statistical analysis
3. Generate visualizations
4. Identify trends and patterns
5. Generate insights and recommendations

Provide a comprehensive analysis report."""
        
        # Run the agent
        result = agent.invoke({"input": task})
        
        # Save analysis report
        analysis_report_path = self.output_dir / "analysis_report.md"
        with open(analysis_report_path, 'w') as f:
            f.write(f"# Analysis Report (LangChain Version)\n\n")
            f.write(f"**Research Question:** {self.research_question}\n\n")
            f.write(result.get('output', ''))
        
        logger.info(f"✓ Analysis report saved to {analysis_report_path}")
        
        return result
    
    def run_full_workflow(self, run_experiments: bool = False, run_analysis: bool = False) -> Dict[str, Any]:
        """
        Run the research workflow.
        
        Args:
            run_experiments: Whether to run experiment design phase
            run_analysis: Whether to run analysis phase
        """
        logger.info("=" * 80)
        logger.info(f"STARTING RESEARCH WORKFLOW (LangChain): {self.research_question}")
        logger.info("=" * 80)
        
        results = {}
        
        # Phase 1: Literature Review
        lit_result = self.run_literature_review()
        results["literature"] = lit_result
        
        # Phase 2: Experiment Design (optional)
        if run_experiments:
            exp_result = self.run_experiment_design()
            results["experiments"] = exp_result
        
        # Phase 3: Analysis (optional)
        if run_analysis:
            analysis_result = self.run_analysis()
            results["analysis"] = analysis_result
        
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
                    },
                    "experiments": {
                        "output": results.get("experiments", {}).get('output', 'Not run'),
                        "intermediate_steps": str(results.get("experiments", {}).get('intermediate_steps', []))[:500]
                    } if run_experiments else {"status": "skipped"},
                    "analysis": {
                        "output": results.get("analysis", {}).get('output', 'Not run'),
                        "intermediate_steps": str(results.get("analysis", {}).get('intermediate_steps', []))[:500]
                    } if run_analysis else {"status": "skipped"}
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

