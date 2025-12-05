"""
Configuration settings for the AI Research Scientist Agent system (LangChain version).
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for the LLM (Google Gemini via LangChain)."""
    model: str = "gemini-2.0-flash"
    temperature: float = 0.7
    max_tokens: int = 8192
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    
    def validate_api_key(self) -> bool:
        """Check if API key is available."""
        return self.api_key is not None and len(self.api_key) > 0


class AgentConfig(BaseModel):
    """Configuration for LangChain agents."""
    max_iterations: int = 15
    max_execution_time: float = 300.0
    verbose: bool = True
    return_intermediate_steps: bool = True


class ExperimentConfig(BaseModel):
    """Configuration for ML experiments."""
    default_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    device: str = "cuda"
    random_seed: int = 42
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"


class PathsConfig(BaseModel):
    """Configuration for file paths and directories."""
    base_dir: Path = Field(default_factory=lambda: Path.cwd())
    outputs_dir: Path = Field(default_factory=lambda: Path.cwd() / "outputs")
    cache_dir: Path = Field(default_factory=lambda: Path.cwd() / "cache")
    papers_dir: Path = Field(default_factory=lambda: Path.cwd() / "cache" / "papers")
    datasets_dir: Path = Field(default_factory=lambda: Path.cwd() / "datasets")
    
    def create_directories(self):
        """Create all necessary directories."""
        for dir_path in [self.outputs_dir, self.cache_dir, self.papers_dir, self.datasets_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


class SystemConfig(BaseModel):
    """Main system configuration."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    
    def initialize(self):
        """Initialize the system configuration."""
        if not self.llm.validate_api_key():
            raise ValueError(
                "Google API key not found. Please set GOOGLE_API_KEY environment variable.\n"
                "Get your free API key at: https://aistudio.google.com/apikey"
            )
        
        self.paths.create_directories()
        return self


# Global configuration instance
config = SystemConfig()
