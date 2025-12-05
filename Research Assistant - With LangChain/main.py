"""
AI Research Scientist Agent - Main Entry Point (LangChain Version)
"""
import argparse
import logging
from pathlib import Path
import sys

from config import config
from orchestrator import ResearchOrchestrator


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the AI Research Scientist Agent (LangChain version)."""
    
    parser = argparse.ArgumentParser(
        description="AI Research Scientist Agent - LangChain Implementation"
    )
    
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Research question to investigate"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize configuration
    try:
        config.initialize()
        logger.info("✓ Configuration initialized")
    except ValueError as e:
        logger.error(f"✗ Configuration error: {e}")
        sys.exit(1)
    
    # Print banner
    print("\n" + "=" * 80)
    print("AI RESEARCH SCIENTIST AGENT (LangChain Version)")
    print("Autonomous ML Research System using LangChain")
    print("=" * 80)
    print(f"\n📋 Research Question: {args.question}")
    print("\n" + "=" * 80 + "\n")
    
    # Create orchestrator
    output_dir = Path(args.output_dir) if args.output_dir else None
    orchestrator = ResearchOrchestrator(
        research_question=args.question,
        output_dir=output_dir
    )
    
    # Run workflow
    try:
        results = orchestrator.run_full_workflow()
        
        # Print summary
        print("\n" + "=" * 80)
        print("✓ RESEARCH WORKFLOW COMPLETED (LangChain)")
        print("=" * 80)
        print(f"\n📁 Output Directory: {results['output_dir']}")
        print(f"\n✓ Check the output directory for:")
        print("  - literature_review.md")
        print("  - workflow_results.json")
        print("\n" + "=" * 80 + "\n")
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Workflow interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n\n✗ Error during workflow: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
