"""
Literature Search Agent using LangChain.
"""
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

from memory import SharedMemory
from config import config
from tools.arxiv_search import search_arxiv, download_paper
from tools.pdf_parser import summarize_paper


def create_literature_agent(memory: SharedMemory) -> AgentExecutor:
    """Create a LangChain-based literature search agent."""
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens
    )
    
    # Define tools
    tools = [
        Tool(
            name="search_arxiv",
            func=lambda query: str(search_arxiv(query, max_results=10)),
            description="Search arXiv for research papers. Input should be a search query string. Returns list of papers with titles, authors, and abstracts."
        ),
        Tool(
            name="download_paper",
            func=download_paper,
            description="Download a paper PDF from arXiv. Input should be the arXiv ID (e.g., '2301.12345'). Returns path to downloaded PDF."
        ),
        Tool(
            name="summarize_paper",
            func=summarize_paper,
            description="Summarize a research paper PDF. Input should be the path to a PDF file. Returns a structured summary of the paper."
        ),
        Tool(
            name="store_paper",
            func=lambda paper_info: memory.store_paper(**eval(paper_info)) or "Paper stored successfully",
            description="Store a paper in memory. Input should be a dict-like string with keys: paper_id, title, abstract, authors, url."
        )
    ]
    
    # Create prompt template
    template = """You are a Literature Research Agent specialized in finding and analyzing academic papers.

Your responsibilities:
1. Search for relevant papers on arXiv based on research questions
2. Download and analyze the most relevant papers
3. Extract key findings, methodologies, and baselines
4. Synthesize literature reviews
5. Identify gaps in existing research

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
