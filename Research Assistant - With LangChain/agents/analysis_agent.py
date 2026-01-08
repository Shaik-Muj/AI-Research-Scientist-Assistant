"""
Analysis Agent using LangChain for experiment result analysis.
"""
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

from memory import SharedMemory
from config import config
from tools.analysis_tools import (
    analyze_results,
    generate_plots,
    identify_trends,
    generate_insights,
    statistical_comparison
)


def create_analysis_agent(memory: SharedMemory) -> AgentExecutor:
    """Create a LangChain-based analysis agent."""
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens
    )
    
    # Define tools
    tools = [
        Tool(
            name="analyze_results",
            func=analyze_results,
            description="Perform statistical analysis on experiment results. Input should be JSON string with experiment results. Returns analysis with performance metrics and insights."
        ),
        Tool(
            name="generate_plots",
            func=lambda data: generate_plots(data, plot_type="training_curves"),
            description="Generate visualization plots. Input should be JSON string with data to plot. Returns path to saved plot image."
        ),
        Tool(
            name="identify_trends",
            func=identify_trends,
            description="Identify trends across multiple experiments. Input should be JSON string with list of experiments. Returns trends summary including best model type and average performance."
        ),
        Tool(
            name="generate_insights",
            func=generate_insights,
            description="Generate high-level insights from analysis data. Input should be JSON string with analysis results. Returns key findings and recommendations."
        ),
        Tool(
            name="compare_experiments",
            func=statistical_comparison,
            description="Compare two experiments statistically. Input should be two JSON strings separated by '|||'. Returns detailed comparison with winner determination."
        ),
        Tool(
            name="search_experiments",
            func=lambda query: str(memory.search_experiments(query, n_results=10)),
            description="Search for experiments in memory. Input should be a search query. Returns list of relevant experiments with their results."
        ),
        Tool(
            name="store_insight",
            func=lambda insight: memory.store_insight(insight, source="analysis_agent") or "Insight stored successfully",
            description="Store an analysis insight in memory. Input should be the insight text. Returns confirmation."
        )
    ]
    
    # Create prompt template
    template = """You are an Analysis Agent specialized in analyzing ML experiment results and generating insights.

Your responsibilities:
1. Analyze experiment results statistically
2. Generate visualizations and plots
3. Identify trends across multiple experiments
4. Compare different approaches
5. Generate actionable insights and recommendations

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
