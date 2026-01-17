# Usage Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Literature Review](#literature-review)
3. [RAG Query Systems](#rag-query-systems)
4. [Full Research Workflow](#full-research-workflow)
5. [Example Notebooks](#example-notebooks)
6. [Command Line Reference](#command-line-reference)

---

## Getting Started

### Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

Next, configure your Google API key. You can obtain a free API key from [Google AI Studio](https://aistudio.google.com/apikey).

```bash
# On Linux/Mac
export GOOGLE_API_KEY="your-api-key-here"

# On Windows
set GOOGLE_API_KEY=your-api-key-here
```

### Running Your First Literature Review

Once you've set up your environment, you can start conducting literature reviews:

```bash
python main.py --question "Can GRU outperform LSTM for small datasets?"
```

If you want to see detailed logs of what the agent is doing, add the `--verbose` flag:

```bash
python main.py --question "Your question" --verbose
```

---

## Literature Review

The literature review agent searches arXiv for relevant papers, downloads them, and generates a comprehensive review based on your research question.

### Command Line Interface

```bash
python main.py --question "Compare GRU and LSTM performance"
```

This will create an output directory with timestamped results:

- `outputs/research_YYYYMMDD_HHMMSS/literature_review.md` - Human-readable review
- `outputs/research_YYYYMMDD_HHMMSS/workflow_results.json` - Machine-readable data

### Python API

You can also use the orchestrator directly in your Python code:

```python
from orchestrator import ResearchOrchestrator

orchestrator = ResearchOrchestrator("Your research question")
result = orchestrator.run_literature_review()
```

### Interactive Notebooks

For a hands-on introduction, check out `examples/literature_review_demo.ipynb`.

---

## RAG Query Systems

We've implemented two different approaches to querying your research papers, each suited for different scenarios.

### Basic RAG - Quick Answers

This mode is designed for straightforward factual questions where you need a quick answer.

**Command Line:**

```bash
python rag_query.py --question "What is GRU?" --num-results 5
```

**Python:**

```python
from rag_query import BasicRAG

rag = BasicRAG()
result = rag.query("What is GRU?", n_results=5)
print(result['answer'])
```

**When to use Basic RAG:**

- Looking up definitions or basic concepts
- Finding specific information from papers
- Quick fact-checking

**Example queries:**

- "What is GRU?"
- "Who authored this paper?"
- "Which dataset was used in the experiments?"

### Agentic RAG - Complex Analysis

For more sophisticated research questions that require comparing multiple sources or multi-step reasoning, use the agentic RAG system.

**Command Line:**

```bash
python agentic_rag.py --question "Compare GRU and LSTM performance on small datasets" --verbose
```

**Python:**

```python
from agentic_rag import AgenticRAG

rag = AgenticRAG()
result = rag.query("Compare GRU and LSTM performance")
print(result['answer'])
print(f"The agent used {result['num_steps']} reasoning steps")
```

**When to use Agentic RAG:**

- Comparing different approaches or models
- Analyzing trends across multiple papers
- Questions requiring synthesis of information

**Example queries:**

- "How does GRU performance compare to LSTM across different dataset sizes?"
- "What are the main criticisms of transformer models in recent literature?"
- "How have attention mechanisms evolved from 2017 to 2023?"

### Choosing Between RAG Modes

| Aspect        | Basic RAG        | Agentic RAG       |
| ------------- | ---------------- | ----------------- |
| Response Time | Fast             | Slower            |
| Question Type | Factual          | Analytical        |
| Reasoning     | Single retrieval | Multiple steps    |
| Best Used For | Quick lookups    | In-depth analysis |

---

## Full Research Workflow

The complete workflow includes three phases: literature review, experiment design, and analysis.

### Command Line

By default, running the main script only executes the literature review:

```bash
python main.py --question "Your question"
```

To run experiments and analysis, you'll need to use the Python API.

### Python API

```python
from orchestrator import ResearchOrchestrator

orchestrator = ResearchOrchestrator("Your research question")

# Run all three phases
result = orchestrator.run_full_workflow(
    run_experiments=True,
    run_analysis=True
)

# Or run phases individually
lit_result = orchestrator.run_literature_review()
exp_result = orchestrator.run_experiment_design()
analysis_result = orchestrator.run_analysis()
```

### Understanding the Output

After running the workflow, you'll find several files in your output directory:

```
outputs/research_YYYYMMDD_HHMMSS/
├── literature_review.md      # Comprehensive literature review
├── experiment_report.md       # Results from ML experiments
├── analysis_report.md         # Statistical analysis and insights
└── workflow_results.json      # Structured data for programmatic access
```

---

## Example Notebooks

We've prepared four Jupyter notebooks to help you get started with different features.

### 1. Literature Review Demo

**Location:** `examples/literature_review_demo.ipynb`

This notebook walks you through:

- Searching for papers on arXiv
- Downloading and processing PDFs
- Storing papers in the vector database
- Generating a literature review

### 2. Basic RAG Demo

**Location:** `examples/basic_rag_demo.ipynb`

Learn how to:

- Ask simple questions about your papers
- Get quick factual answers
- View source citations

### 3. Agentic RAG Demo

**Location:** `examples/agentic_rag_demo.ipynb`

Explore:

- Complex multi-step reasoning
- Comparative analysis across papers
- How the agent thinks through problems
- Side-by-side comparison with Basic RAG

### 4. Full Workflow Demo

**Location:** `examples/full_workflow_demo.ipynb`

See the complete pipeline in action:

- All three agents working together
- Running individual phases
- Managing output files

---

## Command Line Reference

### Main Application

```bash
python main.py --question "QUESTION" [OPTIONS]

Options:
  --question TEXT       Your research question (required)
  --output-dir PATH     Where to save results (auto-generated by default)
  --verbose            Show detailed logging
```

### Basic RAG

```bash
python rag_query.py --question "QUESTION" [OPTIONS]

Options:
  --question TEXT       Your question (required)
  --num-results INT     How many papers to retrieve (default: 5)
```

### Agentic RAG

```bash
python agentic_rag.py --question "QUESTION" [OPTIONS]

Options:
  --question TEXT       Your research question (required)
  --verbose            Display the agent's reasoning process
```

---

## Advanced Usage

### Specifying a Custom Output Directory

```python
orchestrator = ResearchOrchestrator(
    research_question="Your question",
    output_dir="./my_custom_output"
)
```

### Working with Shared Memory

The shared memory system allows you to search and store data across different sessions:

```python
# Search for papers you've already processed
papers = orchestrator.memory.search_papers("GRU LSTM", n_results=5)

# Store custom metadata
orchestrator.memory.set("my_key", "my_value")
value = orchestrator.memory.get("my_key")
```

### Using Agents Independently

If you want more control, you can work with individual agents:

```python
from memory import SharedMemory
from agents.literature_agent import create_literature_agent
from agents.experiment_agent import create_experiment_agent
from agents.analysis_agent import create_analysis_agent

memory = SharedMemory()

# Initialize the agents you need
lit_agent = create_literature_agent(memory)
exp_agent = create_experiment_agent(memory)
analysis_agent = create_analysis_agent(memory)

# Use them directly
result = lit_agent.invoke({"input": "Your task"})
```

---

## Best Practices

### Start with Literature Review

Before running experiments or queries, conduct a literature review to populate your vector database with relevant papers.

### Choose the Right RAG Mode

Think about your question's complexity:

- Need a quick fact? Use Basic RAG
- Need comparative analysis? Use Agentic RAG

### Enable Verbose Logging When Debugging

If something isn't working as expected, verbose mode helps you understand what's happening:

```bash
python main.py --question "..." --verbose
```

### Skip Optional Phases

You don't always need to run experiments and analysis. For literature-only work:

```python
result = orchestrator.run_full_workflow(
    run_experiments=False,
    run_analysis=False
)
```

### Leverage Persistent Storage

Papers are saved in `./cache/memory/` and persist between sessions. You can query them anytime without re-downloading.

---

## Troubleshooting

### API Key Issues

**Problem:** `Error: Google API key not found`

**Solution:** Make sure you've set the `GOOGLE_API_KEY` environment variable. Check that it's set in your current terminal session.

### No Papers Found

**Problem:** `Warning: No relevant papers found`

**Solution:** Try rephrasing your query or using broader search terms. The arXiv search is quite literal, so synonyms might help.

### ChromaDB Connection Errors

**Problem:** `Error: Failed to connect to ChromaDB`

**Solution:** Check that the `./cache/memory/` directory exists and has proper read/write permissions.

### Memory Issues During Experiments

**Problem:** `Error: CUDA out of memory`

**Solution:** Either reduce the batch size in your experiment configuration or switch to CPU-only mode.

---

## Next Steps

1. Start with the `literature_review_demo.ipynb` notebook to get familiar with the system
2. Try the full workflow demo to see all components working together
3. Experiment with both RAG modes to understand their strengths
4. Customize the agent prompts and tools for your specific research needs

---

## Getting Help

If you run into issues:

1. Review this guide for common solutions
2. Check the example notebooks for working code
3. Consult the README for architectural details
4. Look at the inline code documentation
