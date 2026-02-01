# MediAI - Medical RAG Assistant

> An intelligent medical question-answering system powered by Retrieval-Augmented Generation (RAG) for supporting medical professionals.

## 🎯 Overview

MediAI is a production-ready medical AI assistant that combines vector database retrieval with large language models to provide accurate, evidence-based medical information. The system is designed to support medical professionals (medics, nurses, healthcare providers) with reliable medical knowledge retrieval and question answering.

**Key Features:**

- 🤖 **Theme-Aware Response Generation** - Automatically classifies questions into 10 medical themes
- 📚 **RAG Pipeline** - Retrieves relevant context from medical documents before generating answers
- 🔍 **Source Attribution** - Cites sources and indicates information provenance
- 📝 **Comprehensive Logging** - Tracks model thinking and decision-making process
- ⚡ **Batch Processing** - Efficiently handles multiple questions
- 🎨 **Theme-Specific Prompts** - Tailored responses for different medical question types

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Testing](#testing)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [Contributing](#contributing)

## 🏗️ Architecture

```
User Question
     ↓
Theme Detector (ministral-3:8b)
     ↓
Vector Search (Pinecone)
     ↓
Context Evaluation
     ↓
Prompt Selection (Theme-specific)
     ↓
Response Generation (deepseek-v3.1:671b-cloud)
     ↓
Structured MedicalAnswer
```

### Core Components

- **Theme Detection**: Classifies questions into 10 medical categories (anatomy, physiology, pathology, pharmacology, symptoms, diagnosis, treatment, prevention, lifestyle, general)
- **Vector Database**: Pinecone for semantic search across medical documents
- **Embedding Model**: nomic-embed-text for document vectorization
- **Generation Models**: Ollama-powered LLMs for response generation
- **RAG Pipeline**: Orchestrates retrieval and generation workflow

## 📦 Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running
- Pinecone API key
- UV package manager (recommended)

### Required Ollama Models

Pull the required models:

```bash
ollama pull ministral-3:8b          # Theme detection
ollama pull deepseek-v3.1:671b-cloud # Main generation
ollama pull nomic-embed-text:latest  # Embeddings
```

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/mediAi.git
   cd mediAi
   ```

2. **Create virtual environment and install dependencies:**

   ```bash
   # Using UV (recommended)
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt

   # Or using pip
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**

   ```bash
   cp .env.example .env
   # Edit .env and add your PINECONE_API_KEY
   ```

4. **Add medical documents:**
   ```bash
   # Place your medical PDF documents in the data/ directory
   cp /path/to/medical/pdfs/*.pdf data/
   ```

## 🚀 Quick Start

### Streamlit App (Interactive UI)

The easiest way to use the assistant:

```bash
streamlit run app.py
```

- **Real-time Streaming**: Watch the AI "think" and generate answers.
- **Model Selection**: Switch between available Ollama models.
- **Reference Sources**: View citations.

### Python API

```python
from src.rag_pipeline import Med icalRAGPipeline
from src.vector_utils import VectorStore, EmbeddingManager
from src.enums import ModelType
import os

# Initialize
embeddings = EmbeddingManager.get_embeddings(ModelType.EMBEDDING.value)
vectorstore = VectorStore.load_vectorstore(embeddings, "mediai-bot")
rag_pipeline = MedicalRAGPipeline(vectorstore)

# Ask a question
answer = rag_pipeline.process_question("What is hypertension?")

# Access results
print(f"Theme: {answer.theme}")
print(f"Answer: {answer.answer}")
print(f"Sources: {answer.sources}")
print(f"Confidence: {answer.confidence_score}")
```

### Batch Processing

```python
questions = [
    "What is diabetes?",
    "How does insulin work?",
    "What are symptoms of hypertension?"
]

answers = rag_pipeline.batch_process_questions(questions, search_k=3)

for q, a in zip(questions, answers):
    print(f"Q: {q}")
    print(f"A: {a.answer[:200]}...\n")
```

## 📂 Project Structure

```
mediAi/
├── src/                        # Core source code
│   ├── enums.py               # Enumerations (themes, models, sources)
│   ├── models.py              # Pydantic data models
│   ├── prompts.py             # Theme-specific prompt templates
│   ├── logger.py              # Logging configuration
│   ├── vector_utils.py        # Document processing & vector operations
│   ├── model_utils.py         # Model management & inference
│   └── rag_pipeline.py        # Main RAG orchestrator
├── data/                       # Medical PDF documents
├── research/                   # Research notebooks
│   └── trials.ipynb           # Comprehensive testing notebook
├── logs/                       # Application logs
├── requirements.txt           # Python dependencies
├── .env                       # Environment configuration
└── README.md                  # This file
```

## 💻 Usage

### Command Line Interface

```bash
# Run the Streamlit app (if available)
streamlit run app.py

# Or use the FastAPI backend
uvicorn api:app --reload
```

### Testing Notebook

The comprehensive testing notebook validates all functionality:

```bash
cd mediAi
jupyter notebook research/trials.ipynb
```

The notebook includes:

- ✅ Environment setup and validation
- ✅ Module import verification
- ✅ Document loading and processing
- ✅ Vector store initialization
- ✅ Simple and complex question testing
- ✅ Theme detection validation (all 10 themes)
- ✅ Batch processing demonstration
- ✅ Logging and performance metrics

## 🧪 Testing

### Run the Comprehensive Test Suite

1. **Open the testing notebook:**

   ```bash
   jupyter notebook research/trials.ipynb
   ```

2. **Execute all cells sequentially**

3. **Review results:**
   - Simple question: "What is hypertension?"
   - Complex questions with multi-faceted medical topics
   - Theme detection accuracy across all categories
   - Batch processing performance
   - Logging output in `logs/mediai_YYYYMMDD.log`

### Expected Test Results

- ✅ 8+ PDF documents loaded
- ✅ 100+ document chunks created
- ✅ Vector store successfully initialized
- ✅ Theme detection >80% accuracy
- ✅ Comprehensive medical answers with sources
- ✅ Batch processing <5s per question
- ✅ Complete logging of model thinking process

## 🚀 Deployment

### Production Deployment Checklist

1. **Environment Setup**
   - [ ] Configure production Pinecone index
   - [ ] Set up Ollama in production environment
   - [ ] Configure environment variables
   - [ ] Set up logging infrastructure

2. **API Deployment**
   - [ ] Create FastAPI endpoints
   - [ ] Implement rate limiting
   - [ ] Add authentication/authorization
   - [ ] Set up health check endpoints
   - [ ] Configure CORS policies

3. **Monitoring**
   - [ ] Set up metrics collection (Prometheus/Grafana)
   - [ ] Configure alerting for errors and latency
   - [ ] Implement distributed tracing
   - [ ] Track model performance metrics

4. **Scaling**
   - [ ] Implement response caching
   - [ ] Set up load balancing
   - [ ] Configure horizontal scaling
   - [ ] Optimize vector store queries

### Docker Deployment (Example)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with:

```env
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here

# Ollama Configuration (if remote)
OLLAMA_HOST=http://localhost:11434

# Logging Configuration
LOG_LEVEL=INFO
LOG_DIR=./logs

# Application Configuration
INDEX_NAME=mediai-bot
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Model Configuration

Edit `src/enums.py` to configure models:

```python
class ModelType(str, Enum):
    THEME_DETECTOR = "ministral-3:8b"
    MAIN_GENERATOR = "deepseek-v3.1:671b-cloud"
    EMBEDDING = "nomic-embed-text:latest"
```

## 📊 Question Themes

The system automatically detects and routes questions to theme-specific prompts:

| Theme            | Description                     | Example Question                                       |
| ---------------- | ------------------------------- | ------------------------------------------------------ |
| **Anatomy**      | Body structure and anatomy      | "What is the structure of the human heart?"            |
| **Physiology**   | How body systems work           | "How does blood circulation work?"                     |
| **Pathology**    | Diseases and conditions         | "What is diabetes mellitus?"                           |
| **Pharmacology** | Medications and drugs           | "What is metformin used for?"                          |
| **Symptoms**     | Medical symptoms and signs      | "What causes chest pain?"                              |
| **Diagnosis**    | Diagnostic tests and procedures | "What does an ECG measure?"                            |
| **Treatment**    | Treatment options               | "What are treatments for hypertension?"                |
| **Prevention**   | Disease prevention              | "How can I prevent heart disease?"                     |
| **Lifestyle**    | Lifestyle and health habits     | "How does exercise affect health?"                     |
| **General**      | General medical questions       | "What's the difference between type 1 and 2 diabetes?" |

## 📝 Logging

The system includes comprehensive logging that tracks:

- ✅ Module initialization
- ✅ Document processing steps
- ✅ Vector search operations
- ✅ **Theme detection reasoning**
- ✅ **Model thinking and decision-making process**
- ✅ Response generation details
- ✅ Performance metrics
- ✅ Error conditions

Logs are stored in `logs/mediai_YYYYMMDD.log`

## 🔒 Security & Compliance

### Important Disclaimers

> **⚠️ MEDICAL DISCLAIMER**: This system is designed to support medical professionals with information retrieval. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.

### HIPAA Compliance

If handling Protected Health Information (PHI):

- Implement data encryption at rest and in transit
- Add comprehensive audit logging
- Implement access controls and authentication
- Ensure secure API endpoints
- Follow HIPAA guidelines for data handling

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Medical documents from various open-source medical textbooks
- Powered by [Ollama](https://ollama.ai/) for local LLM inference
- Vector storage by [Pinecone](https://www.pinecone.io/)
- Built with [LangChain](https://www.langchain.com/)

## 📧 Support

For support, please:

- Open an issue in the GitHub repository
- Check the [documentation](docs/)
- Review the [testing notebook](research/trials.ipynb) for examples

---

**MediAI** - Empowering medical professionals with AI-assisted knowledge retrieval 🚀

Made with ❤️ for the medical community
