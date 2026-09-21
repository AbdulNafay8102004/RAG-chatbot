# RAG-chatbot
A Retrieval-Augmented Generation (RAG) document chatbot built with LangChain, Sentence-Transformers, ChromaDB, and Ollama. It uses semantic embeddings and vector search to retrieve relevant document content and generate context-grounded answers using a local LLM.

Before running file ingest.py, create folder named documents and place your document files in that folder and use these files in ingest.py to create chunks.
Download and install OLLAMA if you dont want to use api-key

commands :
pip install -r requirements.txt
python ingest.py

ollama --version
ollama pull llama3.2
ollama run llama3.2
/bye (after testing ollama)
pip install langchain-ollama
python chatbot.py
