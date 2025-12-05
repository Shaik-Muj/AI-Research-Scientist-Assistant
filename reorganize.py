"""
Script to reorganize the Research Assistant project into two versions.
"""
import shutil
import os
from pathlib import Path

# Define paths
base_dir = Path(r"c:\Users\mesmh\OneDrive\Desktop\KMIT\MySpace\Codes\Projects\Research Assistant")
no_langchain_dir = base_dir / "Research Assistant - No LangChain"
with_langchain_dir = base_dir / "Research Assistant - With LangChain"

# Create target directories
no_langchain_dir.mkdir(exist_ok=True)
with_langchain_dir.mkdir(exist_ok=True)

# Files and folders to copy
items_to_copy = [
    ".gitignore",
    "README.md",
    "agent_core.py",
    "config.py",
    "main.py",
    "memory.py",
    "orchestrator.py",
    "requirements.txt",
    "verify_setup.py",
    "agents",
    "examples",
    "ml",
    "tools"
]

# Folders to exclude
exclude_dirs = {
    "Research Assistant - No LangChain",
    "Research Assistant - With LangChain",
    "cache",
    "outputs",
    "datasets",
    "checkpoints",
    "results",
    "__pycache__"
}

print("Copying files to 'Research Assistant - No LangChain'...")

for item in items_to_copy:
    src = base_dir / item
    dst = no_langchain_dir / item
    
    if not src.exists():
        print(f"  Skipping {item} (not found)")
        continue
    
    try:
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            print(f"  ✓ Copied directory: {item}")
        else:
            shutil.copy2(src, dst)
            print(f"  ✓ Copied file: {item}")
    except Exception as e:
        print(f"  ✗ Error copying {item}: {e}")

print("\n✓ Reorganization complete!")
print(f"\nNo LangChain version: {no_langchain_dir}")
print(f"With LangChain version: {with_langchain_dir} (will be created next)")
