"""
AI Research Assistant - Simplified version focused on literature research.
"""
import argparse
import logging
from pathlib import Path

from orchestrator import ResearchOrchestrator
from config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the research assistant."""
    parser = argparse.ArgumentParser(
        description="AI Research Assistant - Literature Research Tool"
    )
    
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        required=True,
        help="Research query or question (e.g., 'What is the difference between GRU and LSTM?')"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Validate API key
    if not config.llm.validate_api_key():
        logger.error("❌ GROQ_API_KEY not found in environment variables")
        logger.error("Please set it: $env:GROQ_API_KEY='your-key-here'")
        return
    
    logger.info("✓ Configuration initialized")
    
    # Print header
    print("\n" + "=" * 80)
    print("AI RESEARCH ASSISTANT")
    print("Intelligent Literature Research Tool")
    print("=" * 80)
    print(f"\n📋 Query: {args.query}\n")
    print("=" * 80 + "\n")
    
    # Create orchestrator and run
    output_dir = Path(args.output) if args.output else None
    orchestrator = ResearchOrchestrator(args.query, output_dir=output_dir)
    
    try:
        result = orchestrator.run()
        
        print("\n" + "=" * 80)
        print("✅ RESEARCH COMPLETED")
        print(f"📁 Results saved to: {orchestrator.output_dir}")
        print("=" * 80 + "\n")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Research interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Error during research: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
