
# ⚡ PocketRAG

**Privacy-first, local RAG engine. Run LLM search on your documents without the cloud.**

PocketRAG turns your local documents into an intelligent, searchable database. Built for speed, privacy, and minimal resource usage (optimized for Qwen3.5 0.8B).

![PixelArt](assets/lr.jpeg)

## 🚀 Why PocketRAG?

- **100% Offline:** Your data never leaves your machine.
- **Ultra-Light:** Designed to run on consumer laptops (no massive GPUs required).
- **Zero-Setup:** Single-command CLI.
- **Smart Retrieval:** Uses vector embeddings to find exact answers, not just keywords.
- **Production Ready:** Modular architecture, proper error handling, and logging.

## 🛠️ Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/Qeuph/PocketRag
cd pocketrag
pip install -e .

# Or install directly from PyPI (when published)
# pip install pocketrag
```

### Usage

```bash
# 1. Initialize the database
pocketrag init

# 2. Add documents to index
pocketrag add ./my_documents

# 3. Search documents
pocketrag search "your query here"

# 4. Chat with your documents
pocketrag chat

# Check status anytime
pocketrag status
```

### Advanced Usage

```bash
# Index non-recursively
pocketrag add ./docs --no-recursive

# Use a different model for chat
pocketrag chat --model llama3.2:1b

# Adjust retrieval count
pocketrag chat --top-k 10

# Search with custom result count
pocketrag search "query" --top-k 3

# Clear all indexed data
pocketrag clear --yes
```

## 🏗️ Architecture

PocketRAG uses a modular, production-ready architecture:

```
pocketrag/
├── config/          # Configuration management
├── core/            # Core RAG components
│   ├── chunker.py   # Text chunking strategies
│   ├── embedding.py # Embedding generation
│   ├── indexer.py   # Document indexing pipeline
│   ├── search.py    # Vector search & retrieval
│   └── chat.py      # LLM chat integration
├── utils/           # Utility functions
│   └── document_loader.py  # Multi-format document parsing
└── cli.py           # Command-line interface
```

### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Vector Store | LanceDB | Serverless vector storage |
| Embeddings | SentenceTransformers | High-performance local embeddings |
| LLM | Ollama | Local LLM inference |
| CLI | Typer + Rich | Beautiful command-line interface |

## 📁 Supported File Formats

- **Documents:** PDF, TXT, MD
- **Code:** Python (.py), JavaScript (.js), TypeScript (.ts/.tsx), Java (.java), C/C++ (.c/.cpp/.h), Go (.go), Rust (.rs), Ruby (.rb), PHP (.php)
- **Web:** HTML, CSS, JSON

## 🔧 Configuration

PocketRAG uses sensible defaults but can be customized:

```python
from pocketrag import Config, Indexer, ChatEngine

# Custom configuration
config = Config(
    db_path="./custom_db",
    chunk_size=1000,
    chunk_overlap=100,
    embedding_model="all-MiniLM-L6-v2",
    default_top_k=5,
)

# Programmatic usage
indexer = Indexer(
    db_path=config.db_path,
    chunk_size=config.chunk_size,
)
indexer.index_directory("./docs")

chat = ChatEngine(model_name="llama3.2:1b")
response = chat.chat("What's in my documents?")
```

## 🤝 Contributing

Built for developers who value privacy. PRs are welcome!

### Development Setup

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black pocketrag/

# Lint
ruff check pocketrag/
```

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

If this project saved you time, please ⭐ star the repo!
