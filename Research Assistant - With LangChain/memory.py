"""
Shared memory system using LangChain's memory components and ChromaDB.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document


class SharedMemory:
    """
    Shared memory system for multi-agent collaboration using LangChain.
    """
    
    def __init__(self, persist_directory: str = "./cache/memory"):
        """Initialize shared memory with LangChain components."""
        self.persist_dir = Path(persist_directory)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Create vector stores for different types of data
        self.papers_store = Chroma(
            collection_name="papers",
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir / "papers")
        )
        
        self.experiments_store = Chroma(
            collection_name="experiments",
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir / "experiments")
        )
        
        self.insights_store = Chroma(
            collection_name="insights",
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir / "insights")
        )
        
        # Conversation memory for each agent
        self.agent_memories: Dict[str, ConversationBufferMemory] = {}
        
        # Working memory (in-memory for fast access)
        self.working_memory: Dict[str, Any] = {}
    
    # ========== Paper Management ==========
    
    def store_paper(self, paper_id: str, title: str, abstract: str, 
                   authors: List[str], url: str, metadata: Optional[Dict] = None):
        """Store a research paper in vector store."""
        doc = Document(
            page_content=f"{title}\n\n{abstract}",
            metadata={
                "paper_id": paper_id,
                "title": title,
                "authors": json.dumps(authors),
                "url": url,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
        )
        
        self.papers_store.add_documents([doc])
        
    def search_papers(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant papers using semantic search."""
        results = self.papers_store.similarity_search_with_score(query, k=n_results)
        
        papers = []
        for doc, score in results:
            papers.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': score
            })
        
        return papers
    
    # ========== Experiment Management ==========
    
    def store_experiment(self, experiment_id: str, config: Dict, 
                        results: Optional[Dict] = None, code: Optional[str] = None):
        """Store experiment configuration and results."""
        doc_text = f"Experiment: {config.get('name', experiment_id)}\n"
        doc_text += f"Description: {config.get('description', '')}\n"
        if results:
            doc_text += f"Results: {json.dumps(results)}"
        
        doc = Document(
            page_content=doc_text,
            metadata={
                "experiment_id": experiment_id,
                "config": json.dumps(config),
                "results": json.dumps(results) if results else None,
                "code": code,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        self.experiments_store.add_documents([doc])
    
    def search_experiments(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant experiments."""
        results = self.experiments_store.similarity_search_with_score(query, k=n_results)
        
        experiments = []
        for doc, score in results:
            metadata = doc.metadata
            experiments.append({
                'experiment_id': metadata['experiment_id'],
                'config': json.loads(metadata['config']),
                'results': json.loads(metadata['results']) if metadata.get('results') else None,
                'timestamp': metadata['timestamp'],
                'score': score
            })
        
        return experiments
    
    # ========== Insights & Analysis ==========
    
    def store_insight(self, insight_text: str, source: str, metadata: Optional[Dict] = None):
        """Store an insight or analysis finding."""
        doc = Document(
            page_content=insight_text,
            metadata={
                "source": source,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
        )
        
        self.insights_store.add_documents([doc])
        return str(uuid.uuid4())
    
    def search_insights(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant insights."""
        results = self.insights_store.similarity_search_with_score(query, k=n_results)
        
        insights = []
        for doc, score in results:
            insights.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': score
            })
        
        return insights
    
    # ========== Agent Memory Management ==========
    
    def get_agent_memory(self, agent_name: str) -> ConversationBufferMemory:
        """Get or create conversation memory for an agent."""
        if agent_name not in self.agent_memories:
            self.agent_memories[agent_name] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        return self.agent_memories[agent_name]
    
    # ========== Working Memory ==========
    
    def set(self, key: str, value: Any):
        """Set a value in working memory."""
        self.working_memory[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from working memory."""
        return self.working_memory.get(key, default)
    
    def update(self, data: Dict[str, Any]):
        """Update multiple values in working memory."""
        self.working_memory.update(data)
