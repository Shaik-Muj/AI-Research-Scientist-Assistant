"""
Research Agent using LangChain - Main intelligent agent for literature research.
"""
from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

from memory import SharedMemory
from config import config
from tools.arxiv_search import search_arxiv, download_paper
from tools.pdf_parser import summarize_paper
from tools.semantic_scholar import search_semantic_scholar, find_similar_papers
from tools.citations import format_citation
from tools.comparison import compare_papers, create_comparison_table, identify_trends


def create_research_agent(memory: SharedMemory, model: str = None) -> AgentExecutor:
    """Create a LangChain-based research agent."""
    
    # Initialize LLM with specified model or default
    llm = ChatGroq(
        model=model or config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        api_key=config.llm.api_key
    )
    
    # Define tools
    tools = [
        # Literature Discovery
        Tool(
            name="search_arxiv",
            func=lambda query: str(search_arxiv(query, max_results=10)),
            description="Search arXiv for research papers. Input should be a search query string."
        ),
        Tool(
            name="search_semantic_scholar",
            func=lambda query: str(search_semantic_scholar(query, limit=10)),
            description="Search Semantic Scholar for papers with citation counts. Input should be a search query."
        ),
        Tool(
            name="find_similar_papers",
            func=lambda paper_id: str(find_similar_papers(paper_id, limit=5)),
            description="Find papers similar to a given paper. Input should be a Semantic Scholar paper ID."
        ),
        # Paper Analysis
        Tool(
            name="download_paper",
            func=download_paper,
            description="Download a paper PDF from arXiv. Input should be the arXiv ID."
        ),
        Tool(
            name="summarize_paper",
            func=summarize_paper,
            description="Summarize a research paper. Input should be either an arXiv ID or path to a PDF file."
        ),
        # Comparison
        Tool(
            name="compare_papers",
            func=lambda papers_json: compare_papers(eval(papers_json)),
            description="Compare multiple papers. Input should be a JSON string of paper list."
        ),
        Tool(
            name="create_comparison_table",
            func=lambda papers_json: create_comparison_table(eval(papers_json)),
            description="Create a comparison table for papers. Input should be a JSON string of paper list."
        ),
        # Citations
        Tool(
            name="format_citation",
            func=lambda paper_json: format_citation(eval(paper_json), "bibtex"),
            description="Generate BibTeX citation. Input should be a JSON string of paper metadata."
        )
    ]
    
    # Create prompt template
    template = """You are an AI Research Assistant specialized in helping researchers with literature review and paper analysis.

Your capabilities:
1. Search for relevant papers across multiple sources (arXiv, etc.)
2. Download and analyze research papers
3. Extract key findings, methodologies, and contributions
4. Answer questions about papers and research topics
5. Compare different approaches and methods
6. Synthesize information from multiple papers

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

When answering questions like "What's the difference between X and Y?":
1. Search for papers about both X and Y
2. Summarize relevant papers
3. Extract key differences from the summaries
4. Provide a comprehensive comparison

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    
    prompt = PromptTemplate.from_template(template)
    
    # Create agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=config.agent.verbose,
        max_iterations=config.agent.max_iterations,
        max_execution_time=config.agent.max_execution_time,
        return_intermediate_steps=config.agent.return_intermediate_steps,
        handle_parsing_errors=True
    )
    
    return agent_executor
