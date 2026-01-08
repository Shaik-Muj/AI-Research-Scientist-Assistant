"""
Agentic RAG System - Intelligent multi-step research queries using ReAct agents.
"""
import argparse
import logging
from pathlib import Path
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

from config import config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgenticRAG:
    """Agentic RAG system with multi-step reasoning capabilities."""
    
    def __init__(self, persist_directory: str = "./cache/memory/papers"):
        """Initialize the Agentic RAG system."""
        logger.info("Initializing Agentic RAG system...")
        
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
            temperature=0.5,
            max_tokens=config.llm.max_tokens
        )
        
        # Create agent
        self.agent = self._create_agent()
        
        logger.info("✓ Agentic RAG system initialized")
    
    def _search_papers(self, query: str, n_results: int = 5) -> str:
        """Search for papers and return formatted results."""
        logger.info(f"Searching papers: {query}")
        docs = self.vectorstore.similarity_search(query, k=n_results)
        
        if not docs:
            return "No relevant papers found."
        
        results = []
        for i, doc in enumerate(docs, 1):
            results.append(
                f"\nPaper {i}:\n"
                f"Title: {doc.metadata.get('title', 'Unknown')}\n"
                f"Content: {doc.page_content[:500]}...\n"
                f"Paper ID: {doc.metadata.get('paper_id', 'Unknown')}"
            )
        
        return "\n".join(results)
    
    def _get_paper_details(self, paper_id: str) -> str:
        """Get detailed information about a specific paper."""
        logger.info(f"Getting details for paper: {paper_id}")
        
        # Search by paper ID in metadata
        docs = self.vectorstore.get(where={"paper_id": paper_id})
        
        if not docs or not docs.get('documents'):
            return f"Paper {paper_id} not found."
        
        doc = docs['documents'][0]
        metadata = docs['metadatas'][0] if docs.get('metadatas') else {}
        
        details = f"""
Paper ID: {paper_id}
Title: {metadata.get('title', 'Unknown')}
Authors: {metadata.get('authors', 'Unknown')}
URL: {metadata.get('url', 'Unknown')}

Content:
{doc}
"""
        return details
    
    def _compare_papers(self, topic: str) -> str:
        """Compare papers on a specific topic."""
        logger.info(f"Comparing papers on topic: {topic}")
        
        # Search for papers on the topic
        docs = self.vectorstore.similarity_search(topic, k=5)
        
        if len(docs) < 2:
            return "Not enough papers found for comparison."
        
        comparison = f"Comparison of papers on '{topic}':\n\n"
        for i, doc in enumerate(docs, 1):
            comparison += f"{i}. {doc.metadata.get('title', 'Unknown')}\n"
            comparison += f"   Key points: {doc.page_content[:200]}...\n\n"
        
        return comparison
    
    def _create_agent(self) -> AgentExecutor:
        """Create the RAG agent with tools."""
        
        # Define tools
        tools = [
            Tool(
                name="search_papers",
                func=lambda q: self._search_papers(q, n_results=5),
                description="Search for research papers by query. Input should be a search query string. Returns top 5 relevant papers with titles and content excerpts."
            ),
            Tool(
                name="get_paper_details",
                func=self._get_paper_details,
                description="Get detailed information about a specific paper. Input should be a paper ID. Returns full paper details including title, authors, and content."
            ),
            Tool(
                name="compare_papers",
                func=self._compare_papers,
                description="Compare multiple papers on a topic. Input should be a topic or research question. Returns comparison of relevant papers."
            ),
            Tool(
                name="search_specific",
                func=lambda q: self._search_papers(q, n_results=3),
                description="Search for papers with a specific, narrow query. Input should be a very specific search term. Returns top 3 most relevant papers."
            )
        ]
        
        # Create prompt template
        template = """You are a Research Assistant Agent with access to a database of research papers.

Your goal is to answer complex research questions by:
1. Breaking down the question into sub-questions
2. Searching for relevant papers multiple times if needed
3. Comparing and synthesizing information from multiple sources
4. Providing comprehensive, well-cited answers

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Important:
- Use multiple searches if the question is complex
- Compare different papers when relevant
- Cite specific papers in your final answer
- Be thorough and comprehensive

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
        
        prompt = PromptTemplate.from_template(template)
        
        # Create agent
        agent = create_react_agent(self.llm, tools, prompt)
        
        # Create executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=15,
            max_execution_time=300,
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )
        
        return agent_executor
    
    def query(self, question: str) -> dict:
        """
        Answer a complex question using agentic reasoning.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and reasoning steps
        """
        logger.info(f"Processing agentic query: {question}")
        
        # Run agent
        result = self.agent.invoke({"input": question})
        
        # Extract reasoning steps
        steps = []
        if result.get('intermediate_steps'):
            for action, observation in result['intermediate_steps']:
                steps.append({
                    "action": action.tool,
                    "input": action.tool_input,
                    "observation": str(observation)[:200] + "..."
                })
        
        return {
            "answer": result.get('output', 'No answer generated'),
            "reasoning_steps": steps,
            "num_steps": len(steps)
        }


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Agentic RAG System - Complex research queries with multi-step reasoning"
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Research question to ask"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed reasoning steps"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize Agentic RAG
    rag = AgenticRAG()
    
    # Query
    print("\n" + "="*80)
    print(f"Research Question: {args.question}")
    print("="*80 + "\n")
    
    print("🤔 Agent is thinking and searching...\n")
    
    result = rag.query(args.question)
    
    # Display reasoning steps
    if args.verbose and result['reasoning_steps']:
        print("\nReasoning Steps:")
        print("-" * 80)
        for i, step in enumerate(result['reasoning_steps'], 1):
            print(f"\nStep {i}: {step['action']}")
            print(f"Input: {step['input']}")
            print(f"Result: {step['observation']}")
        print("\n" + "-" * 80)
    
    # Display answer
    print("\nFinal Answer:")
    print("-" * 80)
    print(result['answer'])
    print("\n" + "-" * 80)
    
    print(f"\n✓ Completed in {result['num_steps']} reasoning steps")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
