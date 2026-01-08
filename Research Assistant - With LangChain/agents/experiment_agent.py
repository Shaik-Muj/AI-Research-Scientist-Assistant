"""
Experiment Agent using LangChain for ML experiment design and execution.
"""
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

from memory import SharedMemory
from config import config
from tools.experiment_tools import (
    design_experiment,
    run_experiment,
    compare_experiments,
    evaluate_model
)


def create_experiment_agent(memory: SharedMemory) -> AgentExecutor:
    """Create a LangChain-based experiment agent."""
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens
    )
    
    # Define tools
    tools = [
        Tool(
            name="design_experiment",
            func=design_experiment,
            description="Design an ML experiment based on a natural language description. Input should be a description like 'Compare LSTM and GRU on small MNIST dataset'. Returns JSON experiment configuration."
        ),
        Tool(
            name="run_experiment",
            func=run_experiment,
            description="Run an ML experiment based on a JSON configuration. Input should be a JSON string with experiment config. Returns JSON with results including accuracy and loss."
        ),
        Tool(
            name="compare_experiments",
            func=compare_experiments,
            description="Compare multiple experiments. Input should be comma-separated experiment IDs. Returns comparison summary."
        ),
        Tool(
            name="evaluate_model",
            func=evaluate_model,
            description="Evaluate a trained model on a dataset. Input should be 'model_path,dataset_name'. Returns evaluation metrics."
        ),
        Tool(
            name="store_experiment",
            func=lambda exp_data: memory.store_experiment(**eval(exp_data)) or "Experiment stored successfully",
            description="Store experiment results in memory. Input should be a dict-like string with keys: experiment_id, config, results."
        ),
        Tool(
            name="search_experiments",
            func=lambda query: str(memory.search_experiments(query, n_results=5)),
            description="Search for similar experiments in memory. Input should be a search query. Returns list of relevant experiments."
        )
    ]
    
    # Create prompt template
    template = """You are an ML Experiment Agent specialized in designing and running machine learning experiments.

Your responsibilities:
1. Design experiments based on research questions and literature findings
2. Configure model architectures, datasets, and training parameters
3. Run experiments and collect results
4. Compare different approaches
5. Store and retrieve experiment data

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
