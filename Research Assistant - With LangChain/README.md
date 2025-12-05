# AI Research Scientist Agent (LangChain Version)

A **LangChain-based** implementation of an autonomous AI research system that conducts end-to-end ML research experiments.

## 🌟 Key Differences from Non-LangChain Version

This version uses **LangChain** framework for:

- ✅ **AgentExecutor** - LangChain's agent execution framework
- ✅ **ReAct Pattern** - Reasoning and Acting agent pattern
- ✅ **LangChain Tools** - Standardized tool interface
- ✅ **Chroma Integration** - LangChain's vector store wrapper
- ✅ **ConversationBufferMemory** - LangChain's memory management

## 🏗️ Architecture

```
LangChain Components:
- ChatGoogleGenerativeAI (LLM)
- AgentExecutor (Agent runner)
- ReAct Agent (Reasoning pattern)
- Chroma VectorStore (Memory)
- LangChain Tools (Tool interface)
```

## 📋 Prerequisites

- Python 3.8+
- Google AI API key (free tier available)
- LangChain and LangChain-Google-Genai packages

## 🚀 Installation

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Google AI API key**

   ```bash
   # Windows
   set GOOGLE_API_KEY=your_api_key_here

   # Linux/Mac
   export GOOGLE_API_KEY=your_api_key_here
   ```

## 💡 Usage

### Basic Usage

```bash
python main.py --question "Can GRU outperform LSTM for small datasets?"
```

### Verbose Mode

```bash
python main.py --question "Your research question" --verbose
```

## 📊 LangChain Implementation Details

### Agent Pattern

Uses **ReAct (Reasoning + Acting)** pattern:

```
Question → Thought → Action → Observation → ... → Final Answer
```

### Tools

Tools are wrapped in LangChain's `Tool` class:

```python
Tool(
    name="search_arxiv",
    func=search_arxiv,
    description="Search arXiv for papers..."
)
```

### Memory

Uses LangChain's memory components:

- `Chroma` - Vector store for semantic search
- `ConversationBufferMemory` - Conversation history
- `GoogleGenerativeAIEmbeddings` - Text embeddings

### Agent Execution

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=15,
    handle_parsing_errors=True
)

result = agent_executor.invoke({"input": task})
```

## 🔄 Comparison with Non-LangChain Version

| Feature               | LangChain Version      | Non-LangChain Version |
| --------------------- | ---------------------- | --------------------- |
| **Framework**         | LangChain              | Direct Gemini API     |
| **Agent Pattern**     | ReAct (predefined)     | Custom autonomous     |
| **Dependencies**      | 13 packages            | 11 packages           |
| **Code Complexity**   | Framework abstractions | Direct implementation |
| **Flexibility**       | Within LangChain       | Unlimited             |
| **Industry Standard** | ✅ Yes                 | Custom                |
| **Learning Curve**    | Learn LangChain API    | Understand internals  |

## 🎯 When to Use This Version

**Use LangChain version when:**

- ✅ Working in teams familiar with LangChain
- ✅ Need standard patterns (RAG, chains, etc.)
- ✅ Want framework support and community
- ✅ Building on existing LangChain infrastructure
- ✅ Resume/portfolio needs LangChain keywords

**Use Non-LangChain version when:**

- ✅ Need maximum control and customization
- ✅ Want minimal dependencies
- ✅ Prefer direct API access
- ✅ Building custom agentic behaviors
- ✅ Educational/research purposes

## 📁 Project Structure

```
Research Assistant - With LangChain/
├── main.py                    # CLI entry point
├── orchestrator.py            # LangChain-based orchestrator
├── memory.py                  # LangChain memory (Chroma)
├── config.py                  # Configuration
├── requirements.txt           # LangChain dependencies
│
├── agents/                    # LangChain agents
│   └── literature_agent.py    # AgentExecutor with tools
│
├── ml/                        # ML infrastructure (same)
│   ├── models.py
│   ├── experiment_framework.py
│   └── data_pipeline.py
│
└── tools/                     # Tools (same)
    ├── arxiv_search.py
    └── pdf_parser.py
```

## 🔧 Extending with LangChain

### Add New Tools

```python
from langchain.tools import Tool

new_tool = Tool(
    name="my_tool",
    func=my_function,
    description="What this tool does..."
)
```

### Add Chains

```python
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(input)
```

### Add Memory Types

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)
```

## 📚 LangChain Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain Agents Guide](https://python.langchain.com/docs/modules/agents/)
- [Google Gemini Integration](https://python.langchain.com/docs/integrations/llms/google_ai/)

## 🎓 Learning Value

This implementation demonstrates:

- ✅ LangChain agent patterns
- ✅ Tool integration with LangChain
- ✅ Vector store usage
- ✅ Agent execution flow
- ✅ Industry-standard practices

## 🤝 Comparison Project

This is part of a dual-implementation project:

- **Non-LangChain Version**: Custom autonomous agents
- **LangChain Version**: Framework-based agents (this one)

Both solve the same problem with different approaches, showcasing architectural decision-making.

---

**Built with LangChain + Google Gemini 1.5 Flash (free tier) 🚀**
