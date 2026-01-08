"""
Basic RAG Query System - Simple Q&A using research papers.
"""
import argparse
import logging
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from config import config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BasicRAG:
    """Simple RAG system for querying research papers."""
    
    def __init__(self, persist_directory: str = "./cache/memory/papers"):
        """Initialize the RAG system."""
        logger.info("Initializing Basic RAG system...")
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        
        # Load vector store
        self.vectorstore = Chroma(
            collection_name="papers",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=config.llm.model,
            temperature=0.3,  # Lower temperature for factual answers
            max_tokens=config.llm.max_tokens
        )
        
        logger.info("✓ Basic RAG system initialized")
    
    def query(self, question: str, n_results: int = 5) -> dict:
        """
        Answer a question using retrieved papers.
        
        Args:
            question: User question
            n_results: Number of papers to retrieve
            
        Returns:
            Dictionary with answer and sources
        """
        logger.info(f"Processing query: {question}")
        
        # Retrieve relevant papers
        logger.info(f"Retrieving top {n_results} relevant papers...")
        docs = self.vectorstore.similarity_search(question, k=n_results)
        
        if not docs:
            logger.warning("No relevant papers found")
            return {
                "answer": "I couldn't find any relevant papers to answer your question.",
                "sources": [],
                "num_sources": 0
            }
        
        logger.info(f"Found {len(docs)} relevant papers")
        
        # Build context from retrieved papers
        context = "\n\n".join([
            f"Paper {i+1}:\n{doc.page_content}\n"
            f"Metadata: {doc.metadata}"
            for i, doc in enumerate(docs)
        ])
        
        # Create prompt
        prompt = f"""Based on the following research papers, answer the question.
Be specific and cite which papers support your answer.

Research Papers:
{context}

Question: {question}

Answer (cite papers by number):"""
        
        # Generate answer
        logger.info("Generating answer...")
        response = self.llm.invoke(prompt)
        answer = response.content
        
        # Extract sources
        sources = [
            {
                "title": doc.metadata.get('title', 'Unknown'),
                "paper_id": doc.metadata.get('paper_id', 'Unknown'),
                "url": doc.metadata.get('url', '')
            }
            for doc in docs
        ]
        
        logger.info("✓ Answer generated")
        
        return {
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources)
        }


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Basic RAG Query System - Ask questions about research papers"
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask"
    )
    parser.add_argument(
        "--num-results",
        type=int,
        default=5,
        help="Number of papers to retrieve (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Initialize RAG system
    rag = BasicRAG()
    
    # Query
    print("\n" + "="*80)
    print(f"Question: {args.question}")
    print("="*80 + "\n")
    
    result = rag.query(args.question, n_results=args.num_results)
    
    # Display answer
    print("Answer:")
    print("-" * 80)
    print(result['answer'])
    print("\n" + "-" * 80)
    
    # Display sources
    print(f"\nSources ({result['num_sources']} papers):")
    for i, source in enumerate(result['sources'], 1):
        print(f"\n{i}. {source['title']}")
        print(f"   Paper ID: {source['paper_id']}")
        if source['url']:
            print(f"   URL: {source['url']}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
