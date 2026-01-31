from setuptools import setup, find_packages

setup(
    name="mediAI",
    version="0.1.0",
    author="Macowen Keru",
    author_email="macowenkeru@gmail.com",
    description="An AI-powered medical learning and assistant tool.",
    packages=find_packages(),
    install_requires=[
    "fastapi>=0.128.0",
    "ipykernel>=7.1.0",
    "langchain>=1.2.7",
    "langchain-community>=0.4.1",
    "langchain-core>=1.2.7",
    "langchain-ollama>=1.0.1",
    "langchain-openai>=1.1.7",
    "langchain-pinecone>=0.2.13",
    "langchain-text-splitters>=1.1.0",
    "pydantic>=2.12.5",
    "pypdf>=6.6.2",
    "python-dotenv>=1.2.1",
    "sentence-transformers>=5.2.2",
    "uvicorn>=0.40.0",
    ],
)